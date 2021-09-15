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

nodes_opts = {
    'abbr'             : 'TX',
    'level'            : 'cntyvtd',
    'district_type'    : 'cd',
    'contract_thresh'  : 4,
}

mcmc_opts = {
    'max_steps'             : 1000000,
    'pop_diff_exp'          : 2,
    'pop_imbalance_target'  : 0.1,
    'pop_imbalance_stop'    : 'True',
    'anneal'                : 0,
    'report_period'         : 100,
    'save_period'           : 500,
}

run_opts = {
    'seed_start'      : 3005000,
    'jobs_per_worker' : 5,
    'workers'         : 2,
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
    nodes_opts = get_inputs(nodes_opts)
    mcmc_opts  = get_inputs(mcmc_opts)
    run_opts = get_inputs(run_opts)

nodes_opts['election_filters'] = (
    "office='President' and race='general'",
    "office='USSen' and race='general'",
    "left(office, 5)='USRep' and race='general'",
)

for opt in ['max_steps', 'pop_diff_exp', 'report_period']:
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

from src.nodes import *

nodes_opts['refresh_all'] = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes',
#     'graph',
)
nodes_opts['refresh_tbl'] = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes',
#     'graph',
)

N = Nodes(**nodes_opts)
mcmc_opts['nodes'] = N.tbl

################# Run MCMC #################

# from src.mcmc import *
# import multiprocessing

# save = os.getenv('PYTHONHASHSEED') == HASHSEED
# if save:
#     print(f'hashseed == {HASHSEED} so results are ARE reproducible and WILL be saved to BigQuery')
# else:
#     print(f'hashseed != {HASHSEED} so results are NOT reproducible and will NOT be saved to BigQuery')

# def f(seed):    
#     M = MCMC(seed=seed, gpickle=G.gpickle, save=save, **mcmc_opts)
#     M.run_chain()
#     return M
    
# def multi_f(seed):
#     idx = multiprocessing.current_process()._identity[0]
#     time.sleep(idx / 100)
#     return f(seed)

# if run_opts['workers'] <= 1:
#     M = f(seeds[0])
# else:
#     b = run_opts['seed_start']
#     for k in range(run_opts['jobs_per_worker']):
#         a = b
#         b = a + run_opts['workers']
#         seeds = list(range(a, b))
#         print(f'I will run seeds {seeds}', flush=True)
#         with multiprocessing.Pool(run_opts['workers']) as pool:
#             M = pool.map(multi_f, seeds)


# from src.analysis import *
# start = time.time()
# A = Analysis(nodes_tbl=G.nodes.tbl)#, batch_size=2, max_results=20)
# A.compute_results()
# print(f'analysis took {time_formatter(time.time() - start)}')

print(f'total time elapsed = {time_formatter(time.time() - start_time)}')