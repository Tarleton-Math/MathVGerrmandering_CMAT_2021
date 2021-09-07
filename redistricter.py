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
    'district_type'    : 'sldu',
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
    'seed_start'      : 800,
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
    if os.getenv('PYTHONHASHSEED') == HASHSEED:
        results_bq   = root_bq   + f'.results.{results_stem}_{seed}'
        results_path = root_path /  f'results/{results_stem}/{seed}'
    else:
        results_bq   = None
        results_path = None
    
    start = time.time()
    M = MCMC(seed=seed, results_bq=results_bq, results_path=results_path, gpickle=G.gpickle, **mcmc_opts)
    M.run_chain()
    print(f'finished seed {seed} with pop_imbalance={M.pop_imbalance:.1f} after {M.step} steps and {time_formatter(time.time() - start)}')

with multiprocessing.Pool(int(run_opts['workers'])) as pool:
    a = int(run_opts['seed_start'])
    b = a + int(run_opts['jobs_per_worker']) * pool._processes
    seeds = [str(s).rjust(4,'0') for s in range(a, b)]
    print(f'I will run seeds {seeds}', flush=True)
    pool.map(f, seeds)
    
# from src.analysis import *
# results = mcmc_opts['results_bq']
# # results = 'cmat-315920.results.TX_2020_cntyvtd_cd_2021_09_06_03_35_51'

# A = Analysis(nodes=G.nodes.tbl, results=results, seeds=seeds)
# A.compute_results()
# for attempt in range(1, 6):
#     rpt(f'analysis attempt {attempt}')
#     try:
#         A.compute_results()
#         print(f'success!')
#         break
#     except:
#         rpt(f'failed')
#         new_tbl = nodes_tbl + '_copy'
#         bqclient.copy_table(nodes_tbl, new_tbl).result()
#         nodes_tbl = new_tbl
        
print(f'total time elapsed = {time_formatter(time.time() - start_time)}')