################# Set hashseed for reproducibility #################
import os, sys
hashseed = '0'
if os.getenv('PYTHONHASHSEED') != hashseed:
    os.environ['PYTHONHASHSEED'] = hashseed
    os.execv(sys.executable, [sys.executable] + sys.argv)

################# Define parameters #################
from src import *
from src.graph import *
from src.mcmc import *
from src.analysis import *

graph_opts = {
    'abbr'          : 'TX',
    'level'         : 'cntyvtd',
    'district_type' : 'cd',
    'election_filters' : (
        "office='President' and race='general'",
        "office='USSen' and race='general'",
        "left(office, 5)='USRep' and race='general'",
    ),
}

################# Get Data and Make Graph if necessary #################


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
# del G


################# Run MCMC #################




import multiprocessing

user_name = 'cook'
# u = input(f'user_name (default={user_name})')
# if u != '':
#     user_name = u

max_steps = 10000
# m = input(f'max_steps (default={max_steps})')
# if m != '':
#     max_steps = int(m)

# pop_imbalance_stop = input(f'pop_imbalance_stop (default=True)')
# if pop_imbalance_stop.lower() in ('f', 'false', 'n', 'no'):
#     pop_imbalance_stop = False
# else:
#     pop_imbalance_stop = True

mcmc_opts = {
    'user_name'          : user_name,
#     'random_seed'        : 1,
    'max_steps'          : max_steps,
    'anneal'             : 0,
    'report_period'      : 50,
#     'pop_imbalance_tol'  : 10.0,
#     'pop_imbalance_stop' : pop_imbalance_stop,
    'new_districts'      : 2,
    'num_colors'         : 10,
    'district_type'      : graph_opts['district_type'],
    'gpickle'            : G.gpickle,
#     'gpickle'            : '/home/jupyter/redistricting_data/graph/TX/graph_TX_2020_cntyvtd_cd.gpickle'
}


def f(seed):
    print(f'starting seed {seed}')
    M = MCMC(random_seed=seed, **mcmc_opts)
    assert seed == M.random_seed
    M.run_chain()
    A = Analysis(nodes=G.nodes.tbl, tbl=M.tbl)
#     print(f'fig {seed}')
    fig = A.plot(show=False)
#     print(f'compute {seed}')
    A.get_results()
    print(f'finished seed {seed} after {M.plan} steps with pop_imbalance={M.pop_imbalance:.1f}')

start = time.time()
seed_start = 400
seeds_per_worker = 1

# with multiprocessing.Pool() as pool:
#     seeds = list(range(seed_start, seed_start + seeds_per_worker * pool._processes))
#     pool.map(f, seeds)

# f(seeds[0])
# for seed in seeds:
#     f(seed)

cmd = f'gsutil -m cp -r {root_path}/results gs://cmat-315920-bucket'
print(cmd)
os.system(cmd)