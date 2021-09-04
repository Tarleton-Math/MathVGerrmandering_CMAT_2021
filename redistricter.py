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

graph_opts = {
    'abbr'             : 'TX',
    'level'            : 'cntyvtd',
    'district_type'    : 'cd',
}

mcmc_opts = {
    'user_name'             : 'cook',
    'max_steps'             : 5000,
    'pop_diff_exp'          : 2,
    'pop_imbalance_target'  : 80.0,
    'pop_imbalance_stop'    : 'True',
    'anneal'                : 0,
    'report_period'         : 50,
    'new_districts'         : 2,
}

run_opts = {
    'seed_start'      : 200,
    'jobs_per_worker' : 1,
    'workers'         : 1,
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
from src import *
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
mcmc_opts['gpickle'] = G.gpickle

################# Run MCMC #################

from src.mcmc import *
from src.analysis import *
import multiprocessing
if os.getenv('PYTHONHASHSEED') == HASHSEED:
    print(f'hashseed == {HASHSEED} so results are ARE reproducible and WILL be saved to BigQuery')
else:
    print(f'hashseed != {HASHSEED} so results are NOT reproducible and will NOT be saved to BigQuery')

    
def f(seed):
    idx = multiprocessing.current_process()._identity[0]
    time.sleep(idx / 100)
    print(f'starting seed {seed}', flush=True)
    start = time.time()
    M = MCMC(seed=seed, **mcmc_opts)
    M.run_chain()
    elapsed = time.time() - start
    h, m = divmod(elapsed, 3600)
    m, s = divmod(m, 60)
    print(f'finished seed {seed} with pop_imbalance={M.pop_imbalance:.1f} after {M.step} steps and {int(h)}hrs {int(m)}min {s:.2f}sec')

    if os.getenv('PYTHONHASHSEED') == HASHSEED:
        M.save()
#         A = Analysis(nodes=G.nodes.tbl, tbl=M.tbl)    
    #     print(f'fig {seed}')
    #     fig = A.plot(show=False)
#         A.get_results()
#         print(f'finished analyzing {seed}')
    return M

with multiprocessing.Pool(int(run_opts['workers'])) as pool:
    a = int(run_opts['seed_start'])
    b = a + int(run_opts['jobs_per_worker']) * pool._processes
    seeds = list(range(a, b))
    print(f'I will run seeds {seeds}', flush=True)
    M = pool.map(f, seeds)
    
if os.getenv('PYTHONHASHSEED') == HASHSEED:
    A = Analysis(nodes=G.nodes.tbl, mcmc=M[0].tbl, seeds=seeds)
    A.get_results()

# cmd = f'gsutil -m cp -r {root_path}/results gs://cmat-315920-bucket'
# os.system(cmd)