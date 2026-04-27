"""
=============================================================================
Baseline Comparison Framework — LIGHT VERSION (M2 MacBook)
Orthogonal Basis Network (OBN) Dissertation
=============================================================================
PURPOSE:
  Runs the same experiments as baseline_comparison_v2.py but scaled down
  for fast iteration on your local machine. Each run takes ~3-5 minutes.

KNOWN BUGS (confirmed by code analysis — to be fixed one at a time):
  BUG 1 [CRITICAL]: function_range=(2,4) — PyGAD int gene_space is
          exclusive of high, so pow=4 is NEVER selected. Only sin and cos
          are ever used. Fix: function_range=(2,5)
  BUG 2 [CRITICAL]: operation_range=(0,1) — same issue. Multiplication
          (op=1) is NEVER selected. All expressions are always pure sums.
          Fix: operation_range=(0,2)
  BUG 3 [CRITICAL]: allow_duplicate_genes=False corrupts gene sampling
          for a mixed-type chromosome. Distorts function type distribution.
          Fix: remove the flag (defaults to False only for permutation genes,
          or set allow_duplicate_genes=True)
  BUG 4 [MAJOR]: np.radians() in sin/cos compresses frequency information.
          Fix: remove np.radians() (already tracked as v1 in roadmap)
  BUG 5 [MAJOR]: weight_range=(-1,1) too narrow for periodic functions.
          Fix: expand to (-10,10) (already tracked as v2 in roadmap)
  BUG 6 [MAJOR]: v3 coefficient expansion breaks fitness landscape.
          1/(MAE+eps) returns near-zero for all chromosomes when outputs
          are large. Fix: normalize predictions inside fitness function.
  BUG 7 [MODERATE]: pow computation is data-dependent — same chromosome
          evaluates differently on positive vs negative inputs.
  BUG 8 [MODERATE]: Fitness uses MAE, test metrics use MSE. Different optima.
  BUG 9 [MODERATE]: MinMaxScaler fragile for Rosenbrock (extreme outliers).

CURRENT STATE: v0 only (original bugs intact for baseline measurement)
HOW TO USE:
  1. Run as-is to get your buggy baseline numbers
  2. Fix one bug at a time in this file and re-run
  3. Compare R2 before and after each fix
=============================================================================
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import pygad
    PYGAD_AVAILABLE = True
except ImportError:
    PYGAD_AVAILABLE = False
    print("ERROR: pip install pygad")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: pip install torch")

warnings.filterwarnings('ignore')


# ============================================================================
# LIGHT CONFIG — change these to control speed vs thoroughness
# ============================================================================

LIGHT_CONFIG = {
    #'functions':    ['sphere', 'sine_composite', 'rastrigin'],  # 3 of 7
    'functions':    ['sine_composite'],  # 3 of 7
    'dimensions':   [2, 5],                                      # skip 10D
    'n_samples':    500,                                         # vs 1000
    'n_runs':       2,                                           # vs 3
    'ga_pop':       80,                                          # vs 200
    'ga_gens':      150,                                         # vs 500
    'torch_epochs': 200,                                         # vs 500
    'ga_versions':  ['v1', 'v2'],                                # add versions as you fix bugs
}


# ============================================================================
# SECTION 1: BENCHMARK FUNCTIONS (unchanged from v2)
# ============================================================================

def sphere(X):
    return np.sum(X**2, axis=1)

def rosenbrock(X):
    result = np.zeros(X.shape[0])
    for i in range(X.shape[1] - 1):
        result += 100 * (X[:, i+1] - X[:, i]**2)**2 + (1 - X[:, i])**2
    return result

def rastrigin(X):
    d = X.shape[1]
    return 10 * d + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1)

def griewank(X):
    sum_term = np.sum(X**2, axis=1) / 4000
    prod_term = np.prod(
        np.cos(X / np.sqrt(np.arange(1, X.shape[1] + 1))), axis=1
    )
    return 1 + sum_term - prod_term

def ackley(X):
    d = X.shape[1]
    sum1 = np.sum(X**2, axis=1)
    sum2 = np.sum(np.cos(2 * np.pi * X), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e

def sine_composite(X):
    d = X.shape[1]
    coeffs = np.arange(1, d + 1, dtype=float)
    freqs  = np.arange(1, d + 1, dtype=float)
    return np.sum(coeffs * np.sin(freqs * X), axis=1)

def fourier_mixture(X):
    d = X.shape[1]
    result = np.zeros(X.shape[0])
    for i in range(d):
        result += (i + 1) * np.sin((i + 1) * X[:, i])
        result += 0.5 * (i + 1) * np.cos(2 * (i + 1) * X[:, i])
    return result

BENCHMARKS = {
    'sphere':          {'func': sphere,          'domain': (-5.12, 5.12),       'category': 'polynomial'},
    'rosenbrock':      {'func': rosenbrock,       'domain': (-2.048, 2.048),     'category': 'polynomial'},
    'rastrigin':       {'func': rastrigin,        'domain': (-5.12, 5.12),       'category': 'mixed'},
    'griewank':        {'func': griewank,         'domain': (-600, 600),         'category': 'periodic'},
    'ackley':          {'func': ackley,           'domain': (-5, 5),             'category': 'mixed'},
    'sine_composite':  {'func': sine_composite,   'domain': (-np.pi, np.pi),     'category': 'periodic'},
    'fourier_mixture': {'func': fourier_mixture,  'domain': (-np.pi, np.pi),     'category': 'periodic'},
}

def generate_dataset(func_name, n_dims, n_samples=500, seed=42):
    rng = np.random.RandomState(seed)
    info = BENCHMARKS[func_name]
    low, high = info['domain']
    X = rng.uniform(low, high, size=(n_samples, n_dims))
    y = info['func'](X)
    return X, y


# ============================================================================
# SECTION 2: GA CODE (original — bugs preserved for baseline measurement)
# ============================================================================

def parse_solution_one_output(solution, num_inputs, num_functions):
    """Parse GA chromosome. Original code, unchanged."""
    weights_total    = num_inputs * num_functions
    weights          = solution[:weights_total]

    functions_start  = weights_total
    function_types   = solution[functions_start:functions_start + num_functions]

    operations_start = functions_start + num_functions
    num_operations   = num_functions - 1
    operations       = solution[operations_start:operations_start + num_operations]

    coefficients_start = operations_start + num_operations
    coefficients       = solution[coefficients_start:coefficients_start + num_functions]

    parsed_solution = []
    for i in range(num_functions):
        code = function_types[i]
        # BUG 1: function_range=(2,4) means high=4 is exclusive in PyGAD.
        # pow=4 is never reached. Only sin=2 and cos=3 are ever selected.
        function_type = (
            'sin'  if code == 2 else
            'cos'  if code == 3 else
            'pow'  if code == 4 else
            'none'
        )
        parsed_solution.append({
            'function':    function_type,
            'coefficient': coefficients[i],
            'weights':     weights[i * num_inputs:(i + 1) * num_inputs],
        })

    # BIAS: last gene in chromosome
    bias_start = coefficients_start + num_functions
    bias = float(solution[bias_start])

    return parsed_solution, operations, bias

def apply_operations_one_output(terms, operations):
    """Apply +/* operations between terms. Original code, unchanged."""
    # BUG 2: operation_range=(0,1) means high=1 is exclusive in PyGAD.
    # op=1 (multiply) is never reached. All expressions are pure sums.
    operations = np.array(operations)
    if operations.size == 0:
        return terms[0]

    result_terms    = []
    current_product = terms[0]

    for term, op in zip(terms[1:], operations):
        if op == 1:    # multiply
            current_product *= term
        elif op == 0:  # sum
            result_terms.append(current_product)
            current_product = term
        else:
            raise ValueError(f"Unsupported operation: {op}")

    result_terms.append(current_product)
    return sum(result_terms)

def activation_function_one_output(inputs, solution, num_inputs, num_functions, version='v0'):
    """
    Compute model output for all input rows simultaneously (vectorized).
    version='v0' : original code with np.radians (BUG 4)
    version='v1' : np.radians removed, integer exponents {1,2,3,4}
    version='v2' : fractional exponents, elementwise integer/fractional handling
    """
    inputs  = np.array(inputs, dtype=float)   # shape: (n_samples, n_inputs)

    parsed_solution, operations, bias = parse_solution_one_output(
        solution, num_inputs, num_functions
    )

    terms = []  # each entry will be shape (n_samples,)

    for func in parsed_solution:
        ft          = func['function']
        coefficient = func['coefficient']
        weights     = np.array(func['weights'], dtype=float)  # shape: (n_inputs,)

        if ft == 'sin':
            # inputs @ weights: (n_samples, n_inputs) @ (n_inputs,) = (n_samples,)
            if version == 'v0':
                term = coefficient * np.sin(np.radians(inputs @ weights))
            else:
                term = coefficient * np.sin(inputs @ weights)

        elif ft == 'cos':
            if version == 'v0':
                term = coefficient * np.cos(np.radians(inputs @ weights))
            else:
                term = coefficient * np.cos(inputs @ weights)

        elif ft == 'pow':
            if version == 'v2':
                # Fractional exponents — elementwise per dimension
                # is_integer shape: (n_inputs,) — broadcasts across all rows
                # base shape: (n_samples, n_inputs)
                # powered shape: (n_samples, n_inputs) → sum to (n_samples,)
                exponents  = np.abs(weights)
                is_integer = np.floor(exponents) == exponents
                base       = np.where(is_integer, inputs, np.abs(inputs) + 1e-10)
                term       = coefficient * np.sum(np.power(base, exponents), axis=1)

            elif version == 'v1':
                exponents = np.clip(np.round(np.abs(weights)), 1, 4).astype(int)
                term      = coefficient * np.sum(np.power(inputs, exponents), axis=1)

            else:  # v0
                term = coefficient * np.sum(np.power(inputs, np.abs(weights)), axis=1)

        elif ft == 'none':
            term = np.zeros(len(inputs))

        else:
            raise ValueError(f"Unsupported function type: {ft}")

        terms.append(term)

    # apply_operations_one_output works unchanged — *= and sum() both
    # operate elementwise on numpy arrays automatically
    result = apply_operations_one_output(terms, operations)

    # add bias to every prediction
    return (np.array(result) + bias).tolist()


def solution_to_string(solution, num_inputs, num_functions):
    """Convert chromosome to readable math expression."""
    
    parsed_solution, operations, bias = parse_solution_one_output(
        solution, num_inputs, num_functions
    )

    terms = []
    for func in parsed_solution:
        ft, c, w = func['function'], func['coefficient'], func['weights']
        if ft == 'sin':
            inner = " + ".join(f"{wi:.3f}*x{i+1}" for i, wi in enumerate(w))
            terms.append(f"{c:.3f}*sin({inner})")
        elif ft == 'cos':
            inner = " + ".join(f"{wi:.3f}*x{i+1}" for i, wi in enumerate(w))
            terms.append(f"{c:.3f}*cos({inner})")
        elif ft == 'pow':
            w_arr = np.array(w, dtype=float)
            exponents = np.abs(w_arr)
            inner = " + ".join(f"x{i+1}^{e:.3f}" for i, e in enumerate(exponents))
            terms.append(f"{c:.3f}*({inner})")
        elif ft == 'none':
            terms.append("0")
        else:
            raise ValueError(f"Unsupported function type: {ft}")    

    expression = terms[0] if terms else "0"
    for term, op in zip(terms[1:], operations):
        expression += (" + " if op == 0 else " * ") + term
    expression += f" + {bias:.3f}"   # BIAS: always shown as final additive term
    return expression

def identify_gene_types(num_weights, num_functions, num_operations, num_coefficients):
    return (
        [[float, 2]] * num_weights      +
        [int]        * num_functions    +
        [int]        * num_operations   +
        [[float, 2]] * num_coefficients +
        [[float, 2]] * 1                # BIAS: one constant offset gene
    )


def identify_gene_ranges(num_weights, weight_range,
                          num_functions, function_range,
                          num_operations, operation_range,
                          num_coefficients, coefficient_range,
                          bias_range):
    return (
        [{'low': weight_range[0],      'high': weight_range[1]}]       * num_weights      +
        [{'low': function_range[0],    'high': function_range[1]}]     * num_functions    +
        [{'low': operation_range[0],   'high': operation_range[1]}]    * num_operations   +
        [{'low': coefficient_range[0], 'high': coefficient_range[1]}]  * num_coefficients +
        [{'low': bias_range[0],        'high': bias_range[1]}]         * 1                   # BIAS
    )

def set_elite_parents(population_size, elite_ratio=0.1, parents_ratio=0.5):
    elite   = round(population_size * elite_ratio)
    parents = round(population_size * parents_ratio)
    if parents % 2 != 0:
        parents += 1
    if elite + parents > population_size:
        parents = population_size - elite
        if parents % 2 != 0:
            parents -= 1
    return elite, parents


# ============================================================================
# GA CONFIGURATIONS — one per version being tested
# ============================================================================

GA_CONFIGS = {
    'v0': {
        'description':      'Original code — all bugs present',
        'weight_range':     (-1, 1),
        #'function_range':   (2, 4),    # BUG 1: pow unreachable
        'function_range':   (2, 5),     # FIXED A1: pow=4 now reachable
        #'operation_range':  (0, 1),    # BUG 2: multiply unreachable
        'operation_range':  (0, 2),     # FIXED A2: multiply=1 now reachable
        'coefficient_range':(-1, 1),
        #'coefficient_range':(-10, 10),  #revert # FIXED: allows proper output scaling
        'version':          'v0',       # uses np.radians (BUG 4)
    },
    
    'v1': {
        'description': 'All structural fixes: A1+A2+A3+B1+B2+B4+B3/B5. Coefficient range (-1,1).',
        'weight_range':     (-10, 10),  # FIXED B2: periodic functions need this 
        'function_range':   (2, 5),     # FIXED A1: pow=4 now reachable 
        'operation_range':  (0, 2),     # FIXED A2: multiply=1 now reachable
        'coefficient_range':(-1, 1),    # keep coefficients small to avoid fitness explosion (BUG 6)
        'version':          'v1',       # FIXED B1: np.radians removed 
        'bias_range': (-1, 1),
    },

    'v2': {
    'description': 'v1 + fractional exponents in pow via elementwise integer/fractional handling',
    'weight_range':     (-10, 10),
    'function_range':   (2, 5),
    'operation_range':  (0, 2),
    'coefficient_range':(-1, 1),
    'version': 'v2',                    # activates fractional exponents in pow branch
    'bias_range': (-1, 1),
},

}


def run_ga_experiment(X_train, y_train, X_test, y_test,
                      num_functions=6, sol_per_pop=80,
                      num_generations=150, ga_config_name='v0'):
    if not PYGAD_AVAILABLE:
        return {'MSE': float('nan'), 'MAE': float('nan'), 'R2': float('nan'),
                'RMSE': float('nan'), 'train_time': float('nan'),
                'n_params': 0, 'expression': 'pygad not installed'}

    config       = GA_CONFIGS[ga_config_name]
    num_inputs   = X_train.shape[1]
    num_ops      = num_functions - 1
    num_weights  = num_inputs * num_functions
    num_coeffs   = num_functions
    y_train_flat = y_train.flatten()

    def fitness_function(ga_instance, solution, solution_idx):
        y_pred = np.array(activation_function_one_output(
            X_train, solution, num_inputs, num_functions,
            version=config['version']
        )).reshape(-1)
        if not np.all(np.isfinite(y_pred)):
            return 0.0
        # FIXED B3+B5: R²-based fitness
        # rewards both correct shape AND correct scale
        # directly optimizes what we report as our metric
        ss_res = np.sum((y_pred - y_train_flat) ** 2)
        ss_tot = np.sum((y_train_flat - np.mean(y_train_flat)) ** 2) + 1e-10
        r2 = 1.0 - ss_res / ss_tot
        # convert to strictly positive fitness for PyGAD
        return 1.0 / (1.0 - r2 + 1e-7)

    gene_types  = identify_gene_types(num_weights, num_functions, num_ops, num_coeffs)
    gene_ranges = identify_gene_ranges(
        num_weights,  config['weight_range'],
        num_functions, config['function_range'],
        num_ops,       config['operation_range'],
        num_coeffs,    config['coefficient_range'],
        config['bias_range'], 
    )

    chrom_len = num_weights + num_functions + num_ops + num_coeffs + 1  # +1 for bias
   
    keep_elitism, num_parents_mating = set_elite_parents(sol_per_pop)

    start = time.time()
    ga = pygad.GA(
        sol_per_pop          = sol_per_pop,
        num_generations      = num_generations,
        num_parents_mating   = num_parents_mating,
        num_genes            = chrom_len,
        gene_type            = gene_types,
        gene_space           = gene_ranges,
        #allow_duplicate_genes= False,   # BUG 3: corrupts mixed-type chromosome sampling
        allow_duplicate_genes= True, # 
        fitness_func         = fitness_function,
        parent_selection_type= "tournament",
        K_tournament         = 3,
        keep_elitism         = keep_elitism,
        crossover_type       = "single_point",
        crossover_probability= 0.9,
        mutation_type        = "random",
        mutation_by_replacement = True,
        mutation_probability = 0.05,
        save_solutions       = False,
        save_best_solutions  = True,
        suppress_warnings    = True,
    )
    ga.run()
    train_time = time.time() - start

    solution, best_fitness, _ = ga.best_solution()
    y_pred = np.array(activation_function_one_output(
        X_test, solution, num_inputs, num_functions,
        version=config['version']
    )).reshape(-1)

    return {
        'MSE':            mean_squared_error(y_test, y_pred),
        'MAE':            mean_absolute_error(y_test, y_pred),
        'R2':             r2_score(y_test, y_pred),
        'RMSE':           np.sqrt(mean_squared_error(y_test, y_pred)),
        'train_time':     train_time,
        'n_params':       chrom_len,
        'expression':     solution_to_string(solution, num_inputs, num_functions),
        'best_generation':ga.best_solution_generation,
        'best_fitness':   best_fitness,
        'fitness_history':list(ga.best_solutions_fitness),
    }


# ============================================================================
# SECTION 3: PYTORCH BASELINES (scaled down for local runs)
# ============================================================================

if TORCH_AVAILABLE:

    class SinActivation(nn.Module):
        def forward(self, x): return torch.sin(x)

    class MLPBaseline(nn.Module):
        def __init__(self, input_dim, hidden_dims=[32, 16], activation='relu'):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                if activation == 'relu': layers.append(nn.ReLU())
                elif activation == 'sin': layers.append(SinActivation())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)
        def forward(self, x): return self.network(x).squeeze(-1)

    class ChebyKANLayer(nn.Module):
        def __init__(self, input_dim, output_dim, degree=4):
            super().__init__()
            self.degree = degree
            self.coeffs = nn.Parameter(torch.randn(output_dim, input_dim, degree + 1) * 0.1)
        def forward(self, x):
            x_norm = torch.tanh(x)
            cheb = [torch.ones_like(x_norm), x_norm]
            for n in range(2, self.degree + 1):
                cheb.append(2 * x_norm * cheb[-1] - cheb[-2])
            return torch.einsum('oid,bid->bo', self.coeffs, torch.stack(cheb, dim=-1))

    class FourierKANLayer(nn.Module):
        def __init__(self, input_dim, output_dim, num_frequencies=5):
            super().__init__()
            self.num_freq  = num_frequencies
            self.sin_coeffs = nn.Parameter(torch.randn(output_dim, input_dim, num_frequencies) * 0.1)
            self.cos_coeffs = nn.Parameter(torch.randn(output_dim, input_dim, num_frequencies) * 0.1)
            self.bias       = nn.Parameter(torch.zeros(output_dim))
        def forward(self, x):
            freqs   = torch.arange(1, self.num_freq + 1, dtype=x.dtype, device=x.device)
            freq_x  = x.unsqueeze(-1) * freqs
            return (torch.einsum('oif,bif->bo', self.sin_coeffs, torch.sin(freq_x)) +
                    torch.einsum('oif,bif->bo', self.cos_coeffs, torch.cos(freq_x)) +
                    self.bias)

    class ChebyKANNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dims=[8, 4], degree=4):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_dims:
                layers.append(ChebyKANLayer(prev, h, degree))
                prev = h
            layers.append(ChebyKANLayer(prev, 1, degree))
            self.layers = nn.ModuleList(layers)
        def forward(self, x):
            for layer in self.layers: x = layer(x)
            return x.squeeze(-1)

    class FourierKANNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dims=[8, 4], num_frequencies=5):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_dims:
                layers.append(FourierKANLayer(prev, h, num_frequencies))
                prev = h
            layers.append(FourierKANLayer(prev, 1, num_frequencies))
            self.layers = nn.ModuleList(layers)
        def forward(self, x):
            for layer in self.layers: x = layer(x)
            return x.squeeze(-1)

    def train_torch_model(model, X_train, y_train, epochs=200, lr=0.001, batch_size=64):
        # M2 Mac: use MPS if available, else CPU
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model  = model.to(device)
        X_t    = torch.FloatTensor(X_train).to(device)
        y_t    = torch.FloatTensor(y_train).to(device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        opt    = optim.Adam(model.parameters(), lr=lr)
        sched  = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.5)
        crit   = nn.MSELoss()
        start  = time.time()
        for _ in range(epochs):
            model.train()
            eloss = 0
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                eloss += loss.item()
            sched.step(eloss / len(loader))
        return model, time.time() - start

    def eval_torch_model(model, X_test, y_test):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
        return {
            'MSE':  mean_squared_error(y_test, pred),
            'MAE':  mean_absolute_error(y_test, pred),
            'R2':   r2_score(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        }


# ============================================================================
# SECTION 4: EXPERIMENT RUNNER
# ============================================================================

def run_comparison(config=None):
    if config is None:
        config = LIGHT_CONFIG

    all_results = []

    for fname in config['functions']:
        for ndim in config['dimensions']:
            print(f"\n{'='*55}")
            print(f"  {fname.upper()} | {ndim}D | {BENCHMARKS[fname]['category']}")
            print(f"{'='*55}")

            X, y = generate_dataset(fname, ndim, config['n_samples'])
            Xs   = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
            # BUG 9: MinMaxScaler is sensitive to outliers (bad for Rosenbrock)
            ys   = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()
            Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42)

            # --- GA versions ---
            for gv in config['ga_versions']:
                print(f"\n  GA-OBN ({gv}): {GA_CONFIGS[gv]['description']}")
                run_r2s = []
                for run in range(config['n_runs']):
                    res = run_ga_experiment(
                        Xtr, ytr, Xte, yte,
                        num_functions  = 6,
                        sol_per_pop    = config['ga_pop'],
                        num_generations= config['ga_gens'],
                        ga_config_name = gv,
                    )
                    run_r2s.append(res['R2'])
                    all_results.append({
                        'Function': fname, 'Dims': ndim,
                        'Model': f'GA-OBN ({gv})', 'Run': run,
                        **{k: res[k] for k in ['MSE','MAE','R2','RMSE','train_time','n_params']},
                    })
                    if run == 0:
                        print(f"    Expression: {res['expression']}")
                print(f"    R2 per run: {[f'{r:.4f}' for r in run_r2s]}  |  avg: {np.mean(run_r2s):.4f}")

            # --- PyTorch baselines ---
            if TORCH_AVAILABLE:
                torch_models = {
                    'MLP-ReLU':   lambda d: MLPBaseline(d, [32, 16], 'relu'),
                    'MLP-Sin':    lambda d: MLPBaseline(d, [32, 16], 'sin'),
                    'ChebyKAN':   lambda d: ChebyKANNetwork(d, [8, 4], degree=4),
                    'FourierKAN': lambda d: FourierKANNetwork(d, [8, 4], num_frequencies=5),
                }
                for mname, mfactory in torch_models.items():
                    run_r2s = []
                    for run in range(config['n_runs']):
                        model  = mfactory(ndim)
                        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        model, ttime = train_torch_model(model, Xtr, ytr, config['torch_epochs'])
                        metrics = eval_torch_model(model, Xte, yte)
                        run_r2s.append(metrics['R2'])
                        all_results.append({
                            'Function': fname, 'Dims': ndim,
                            'Model': mname, 'Run': run,
                            'MSE': metrics['MSE'], 'MAE': metrics['MAE'],
                            'R2': metrics['R2'], 'RMSE': metrics['RMSE'],
                            'train_time': ttime, 'n_params': nparams,
                        })
                    print(f"  {mname:<12} R2 avg: {np.mean(run_r2s):.4f}  params: {nparams}")

    return pd.DataFrame(all_results)


def print_report(df):
    summary = df.groupby(['Function', 'Dims', 'Model']).agg(
        R2_mean  =('R2',  'mean'),
        R2_std   =('R2',  'std'),
        MSE_mean =('MSE', 'mean'),
        Params   =('n_params', 'first'),
        Time_s   =('train_time', 'mean'),
    ).reset_index().sort_values(['Function', 'Dims', 'R2_mean'], ascending=[True, True, False])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for fname in summary['Function'].unique():
        print(f"\n--- {fname.upper()} ---")
        for dim in sorted(summary[summary['Function']==fname]['Dims'].unique()):
            sub = summary[(summary['Function']==fname) & (summary['Dims']==dim)]
            print(f"\n  {dim}D:  {'Model':<22} {'R2 (mean±std)':>16}  {'MSE':>10}  {'Params':>7}  {'Time':>6}")
            print(f"        {'-'*22} {'-'*16}  {'-'*10}  {'-'*7}  {'-'*6}")
            for _, row in sub.iterrows():
                std_str = f"±{row['R2_std']:.3f}" if not np.isnan(row['R2_std']) else ""
                print(f"        {row['Model']:<22} {row['R2_mean']:>8.4f}{std_str:<8}  "
                      f"{row['MSE_mean']:>10.5f}  {int(row['Params']):>7}  {row['Time_s']:>5.1f}s")

    print("\n" + "=" * 80)
    print("BEST MODEL PER FUNCTION")
    print("=" * 80)
    best = summary.loc[summary.groupby(['Function','Dims'])['R2_mean'].idxmax()]
    for _, row in best.iterrows():
        print(f"  {row['Function']:>15} ({int(row['Dims'])}D):  {row['Model']:<22}  R2={row['R2_mean']:.4f}")

    return summary


# ============================================================================
# SECTION 5: MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("OBN BASELINE COMPARISON — LIGHT VERSION")
    print(f"Functions : {LIGHT_CONFIG['functions']}")
    print(f"Dimensions: {LIGHT_CONFIG['dimensions']}")
    print(f"GA        : pop={LIGHT_CONFIG['ga_pop']}, gens={LIGHT_CONFIG['ga_gens']}, runs={LIGHT_CONFIG['n_runs']}")
    print(f"Torch     : epochs={LIGHT_CONFIG['torch_epochs']}")
    print(f"Versions  : {LIGHT_CONFIG['ga_versions']}")
    print("=" * 55)

    df = run_comparison(LIGHT_CONFIG)
    summary = print_report(df)

    out_file = 'results_light.csv'
    df.to_csv(out_file, index=False)
    print(f"\nRaw results saved to {out_file}")

    # -------------------------------------------------------------------------
    # HOW TO ADD A NEW VERSION AFTER FIXING A BUG:
    #
    # Step 1: Add a new entry to GA_CONFIGS (uncomment the v1 block above
    #         and adjust it for the fix you made)
    # Step 2: Add the version name to LIGHT_CONFIG['ga_versions']:
    #         e.g. 'ga_versions': ['v0', 'v1']
    # Step 3: Re-run and compare R2 column between versions
    # -------------------------------------------------------------------------
