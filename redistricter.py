################# Set hashseed for reproducibility #################
HASHSEED = '0'
try:
    notebook
except:
    import os, sys
    if os.getenv('PYTHONHASHSEED') != HASHSEED:
        os.environ['PYTHONHASHSEED'] = HASHSEED
        os.execv(sys.executable, [sys.executable] + sys.argv)

################# Define parameters #################
from src import *
start_time = time.time()

graph_opts = {
    'abbr'             : 'TX',
    'level'            : 'cntyvtd',
    'district_type'    : 'cd',
}

mcmc_opts = {
    'max_steps'            : 5000,
    'pop_diff_exp'          : 2,
    'pop_imbalance_target'  : 0.5,
    'pop_imbalance_stop'    : 'True',
    'anneal'                : 0,
    'report_period'         : 50,
    'new_districts'         : 2,
}

run_opts = {
    'seed_start'      : 0,
    'jobs_per_worker' : 5,
    'workers'         : 80,

}

yes = (True, 't', 'true', 'y', 'yes')
def get_inputs(opts):
    for opt, default in opts.items():
        inp = input(f'{opt.ljust(20," ")} (default={default})')
        if inp != '':
            opts[opt] = inp
    return opts

try:
    skip_inputs = skip_inputs in yes
except:
    try:
        skip_inputs = sys.argv[1].lower() in yes
    except:
        skip_inputs = False

if not skip_inputs:
    graph_opts = get_inputs(graph_opts)
    mcmc_opts  = get_inputs(mcmc_opts)
    run_opts = get_inputs(run_opts)

graph_opts['election_filters'] = (
    "office='President' and race='general'",
    "office='USSen' and race='general'",
    "left(office, 5)='USRep' and race='general'",
)

for opt in ['max_steps', 'pop_diff_exp', 'report_period', 'new_districts']:
    mcmc_opts[opt] = int(mcmc_opts[opt])

for opt in ['pop_imbalance_target', 'anneal']:
    mcmc_opts[opt] = float(mcmc_opts[opt])
    
if mcmc_opts['pop_imbalance_stop'].lower() in yes:
    mcmc_opts['pop_imbalance_stop'] = True
else:
    mcmc_opts['pop_imbalance_stop'] = False
    
for opt in ['seed_start', 'jobs_per_worker', 'workers']:
    run_opts[opt] = int(run_opts[opt])

################# Get Data and Make Graph if necessary #################

from src.graph import *

graph_opts['refresh_all'] = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes',
#     'graph',
)
graph_opts['refresh_tbl'] = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes',
#     'graph',
)

G = Graph(**graph_opts)

################# Run MCMC #################

from src.mcmc import *
import multiprocessing
if os.getenv('PYTHONHASHSEED') == HASHSEED:
    print(f'hashseed == {HASHSEED} so results are ARE reproducible and WILL be saved to BigQuery')
else:
    print(f'hashseed != {HASHSEED} so results are NOT reproducible and will NOT be saved to BigQuery')
    

# timestamp = str(pd.Timestamp.now().round("s")).replace(' ','_').replace('-','_').replace(':','_')
ds = root_bq + '.results'
bqclient.create_dataset(ds, exists_ok=True)
results_stem = G.gpickle.stem[6:]
results_bq = ds + f'.{results_stem}'
results_path = root_path / f'results/{results_stem}'

def f(seed):
    idx = multiprocessing.current_process()._identity[0]
    time.sleep(idx / 100)
    print(f'starting seed {seed}', flush=True)
    
    start = time.time()
    M = MCMC(seed=seed, gpickle=G.gpickle, **mcmc_opts,
             results_bq  =results_bq+f'_{seed}',
             results_path=results_path / seed)
    M.run_chain()
    print(f'finished seed {seed} with pop_imbalance={M.pop_imbalance:.1f} after {M.step} steps and {time_formatter(time.time() - start)}')

    if os.getenv('PYTHONHASHSEED') == HASHSEED:
#         M.save()
        saved = False
        for i in range(1, 60):
            try:
                M.save()
#                 print(f'seed {seed} save try {i} succeeded')
                return M
            except:
#                 rpt(f'seed {seed} try {i} failed')
                time.sleep(1)
        raise Exception(f'I tried to write the result of seed {seed} {i} times without success - giving up')

a = run_opts['seed_start']
b = a + run_opts['jobs_per_worker'] * run_opts['workers']
seeds = [str(s).rjust(4,'0') for s in range(a, b)]

# with multiprocessing.Pool(run_opts['workers']) as pool:
#     print(f'I will run seeds {seeds}', flush=True)
#     pool.map(f, seeds)

################# Analyze #################
    
from src.analysis import *

A = Analysis(nodes=G.nodes.tbl, results_bq=results_bq, seeds=seeds)
A.compute_results()
        
print(f'total time elapsed = {time_formatter(time.time() - start_time)}')