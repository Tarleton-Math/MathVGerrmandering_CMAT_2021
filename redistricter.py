################# Set hashseed for reproducibility #################
HASHSEED = '0'
try:
    notebook
except:
    import os, sys
    if os.getenv('PYTHONHASHSEED') != HASHSEED:
        os.environ['PYTHONHASHSEED'] = HASHSEED
        os.execv(sys.executable, [sys.executable] + sys.argv)

################# Initial Setup #################
from src import *
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
        
try:
    run_mcmc
except:
    run_mcmc = True
    
################# Get data & make nodes #################
from src.nodes import *

nodes_opts = {
    'district_type'    : 'sldu',
    'contract_thresh'  : 10,
    'level'            : 'cntyvtd',
    'abbr'             : 'TX',
}
    
mcmc_opts = {
    'max_steps'             : 10000000,
    'pop_diff_exp'          : 2,
    'defect_multiplier'     : 1.0,
    'yolo_length'           : 5,
    'pop_deviation_stop'    : False,
    'report_period'         : 25,
    'save_period'           : 500,
}

run_opts = {
    'seed_start'      : 4000000,
    'jobs_per_worker' : 1,
    'workers'         : 80,
}

if not skip_inputs:
    nodes_opts = get_inputs(nodes_opts)
if not skip_inputs:
    mcmc_opts  = get_inputs(mcmc_opts)
if not skip_inputs:
    run_opts = get_inputs(run_opts)

nodes_opts['election_filters'] = (
    "office='President' and race='general'",
    "office='USSen' and race='general'",
    "left(office, 5)='USRep' and race='general'",
)

nodes_opts['refresh_all'] = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes'
)
nodes_opts['refresh_tbl'] = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes'
)

N = Nodes(**nodes_opts)

################# Make graph and run MCMC #################

from src.mcmc import *
import multiprocessing

if nodes_opts['district_type'] == 'cd':
    mcmc_opts['pop_deviation_target'] = 0.01
elif nodes_opts['district_type'] == 'sldu':
    mcmc_opts['pop_deviation_target'] = 10.0
elif nodes_opts['district_type'] == 'sldl':
    mcmc_opts['pop_deviation_target'] = 10.0
else:
    raise Exception(f'unknown district type {nodes_opts["district_type"]}')
    
mcmc_opts['nodes_tbl'] = N.tbl
    
for opt in ['max_steps', 'pop_diff_exp', 'report_period', 'yolo_length']:
    mcmc_opts[opt] = int(mcmc_opts[opt])

for opt in ['pop_deviation_target', 'defect_multiplier']:
    mcmc_opts[opt] = float(mcmc_opts[opt])
    
mcmc_opts['pop_deviation_stop'] = bool(mcmc_opts['pop_deviation_stop'])
    
for opt in ['seed_start', 'jobs_per_worker', 'workers']:
    run_opts[opt] = int(run_opts[opt])

if os.getenv('PYTHONHASHSEED') == HASHSEED:
    mcmc_opts['save'] = True
    print(f'hashseed == {HASHSEED} so results are ARE reproducible and WILL be saved to BigQuery')
else:
    mcmc_opts['save'] = False
    print(f'hashseed != {HASHSEED} so results are NOT reproducible and will NOT be saved to BigQuery')

def f(random_seed):
    M = MCMC(random_seed=random_seed, **mcmc_opts)
    print(f'starting seed {random_seed}')
    M.run_chain()
    return M
    
def multi_f(random_seed):
    idx = multiprocessing.current_process()._identity[0]
    time.sleep(idx / 100)
    return f(random_seed)

a = run_opts['seed_start']
b = a + run_opts['jobs_per_worker'] * run_opts['workers']
random_seeds = np.arange(a, b)

if run_mcmc:
    start_time = time.time()
    if run_opts['workers'] == 1:
        for s in random_seeds:
            M = f(s)
    else:
        with multiprocessing.Pool(run_opts['workers']) as pool:
            M = pool.map(multi_f, random_seeds)
    print(f'total time elapsed = {time_formatter(time.time() - start_time)}')

################# Post-Processing & Analysis #################




# from src.analysis import *
# start = time.time()
# A = Analysis(nodes_tbl=G.nodes.tbl)#, batch_size=2, max_results=20)
# A.compute_results()
# print(f'analysis took {time_formatter(time.time() - start)}')

# print(f'total time elapsed = {time_formatter(time.time() - start_time)}')