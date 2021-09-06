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
    'pop_imbalance_target'  : 1.0,
    'pop_imbalance_stop'    : 'True',
    'anneal'                : 0,
    'report_period'         : 50,
    'new_districts'         : 2,
}

run_opts = {
    'seed_start'      : 0,
    'jobs_per_worker' : 5,
    'workers'         : 8,

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
    

timestamp = str(pd.Timestamp.now().round("s")).replace(' ','_').replace('-','_').replace(':','_')
results_stem = G.gpickle.stem[6:]
mcmc_opts['gpickle'] = G.gpickle
mcmc_opts['results_bq']   = root_bq  + f'.results.{results_stem}_{timestamp}'
mcmc_opts['results_path'] = root_path / f'results/{results_stem}/{timestamp}'
mcmc_opts['results_path'].mkdir(parents=True, exist_ok=True)
bqclient.create_dataset(root_bq  + f'.results', exists_ok=True)


def f(seed):
    idx = multiprocessing.current_process()._identity[0]
    time.sleep(idx / 100)
    print(f'starting seed {seed}', flush=True)
    start = time.time()
    M = MCMC(seed=seed, **mcmc_opts)
    M.run_chain()
    print(f'finished seed {seed} with pop_imbalance={M.pop_imbalance:.1f} after {M.step} steps and {time_formatter(time.time() - start)}')

    if os.getenv('PYTHONHASHSEED') == HASHSEED:
        saved = False
        for i in range(1, 120):
            try:
                M.save()
#                 print(f'seed {seed} save try {i} succeeded')
                return M
            except:
#                 rpt(f'seed {seed} try {i} failed')
                time.sleep(1)
        raise Exception(f'I tried to write the result of seed {seed} {i} times without success - giving up')

with multiprocessing.Pool(int(run_opts['workers'])) as pool:
    a = int(run_opts['seed_start'])
    b = a + int(run_opts['jobs_per_worker']) * pool._processes
    seeds = list(range(a, b))
    print(f'I will run seeds {seeds}', flush=True)
    pool.map(f, seeds)
    
from src.analysis import *
results = mcmc_opts['results_bq']
# results = 'cmat-315920.results.TX_2020_cntyvtd_cd_2021_09_06_03_35_51'

A = Analysis(nodes=G.nodes.tbl, results=results, seeds=seeds)
A.compute_results()
for attempt in range(1, 6):
    rpt(f'analysis attempt {attempt}')
    try:
        A.compute_results()
        print(f'success!')
        break
    except:
        rpt(f'failed')
        new_tbl = nodes_tbl + '_copy'
        bqclient.copy_table(nodes_tbl, new_tbl).result()
        nodes_tbl = new_tbl
        
print(f'total time elapsed = {time_formatter(time.time() - start_time)}')