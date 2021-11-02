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
from src.mcmc import *
import multiprocessing
opts = {
    'level'                : 'cntyvtd',
    'proposal'             : 'plans2168',
    'contract'             : '0',
    'random_seed'          : 0,
    'max_steps'            : 10,
    'report_period'        : 1,
    'save_period'          : 500,
    'pop_deviation_target' : np.inf,
    'yolo_length'          : 10,
    'defect_cap'           : np.inf,
    'election_filters'     : (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'"),
}
run_opts = {
    'seed_start'      : 0,
    'jobs_per_worker' : 1,
    'workers'         : 1,
}

if opts['proposal'][4] == 'c':
    opts['pop_deviation_target'] = 0.01
    opts['defect_cap'] = 60
elif opts['proposal'][4] == 's':
    opts['pop_deviation_target'] = 10.0
    opts['defect_cap'] = 35
elif opts['proposal'][4] == 'h':
    opts['pop_deviation_target'] = 10.0
    opts['defect_cap'] = 10
else:
    raise Exception(f'unknown proposal {opts["proposal"]}')
    

for opt, val in {**opts, **run_opts}.items():
    print(f'{opt.ljust(22, " ")}: {val}')
ok = input('Using options above - type "q" to quit: ')
if ok.lower() in ['n', 'no', 'stop', 'quit', 'q']:
    raise SystemExit(0)

def f(random_seed):
    print(f'starting seed {random_seed}')
    M = MCMC(**opts)
    M.run_chain()
    return M
    
def multi_f(random_seed):
    time.sleep(multiprocessing.current_process()._identity[0] / 100)
    return f(random_seed)

a = run_opts['seed_start']
b = a + run_opts['jobs_per_worker'] * run_opts['workers']
random_seeds = np.arange(a, b)

start_time = time.time()
if run_opts['workers'] == 1:
    for s in random_seeds:
        M = f(s)
else:
    with multiprocessing.Pool(run_opts['workers']) as pool:
        M = pool.map(multi_f, random_seeds)
print(f'total time elapsed = {time_formatter(time.time() - start_time)}')

################ Post-Processing & Analysis #################
# from src.analysis import *
# start = time.time()
# A = Analysis(nodes_tbl=G.nodes.tbl)#, batch_size=2, max_results=20)
# A.compute_results()
# print(f'analysis took {time_formatter(time.time() - start)}')

# print(f'total time elapsed = {time_formatter(time.time() - start_time)}')