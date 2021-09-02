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
#         "office like 'USRep%' and race='general'",
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
#     'edges',
#     'graph',
)
graph_opts['refresh_tbl'] = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes',
#     'edges',
#     'graph',
)

G = Graph(**graph_opts)
# del G


################# Run MCMC #################


import multiprocessing

user_name = input(f'user_name (default=cook)')
if user_name == '':
    user_name = 'cook'

max_steps = input(f'max_steps (default=100000)')
if max_steps == '':
    max_steps = 100000

pop_imbalance_stop = input(f'pop_imbalance_stop (default=True)')
if pop_imbalance_stop.lower() in ('f', 'false', 'n', 'no'):
    pop_imbalance_stop = False
else:
    pop_imbalance_stop = True

mcmc_opts = {
    'user_name'          : user_name,
#     'random_seed'        : 1,
    'max_steps'          : max_steps,
    'pop_imbalance_tol'  : 10.0,
    'pop_imbalance_stop' : pop_imbalance_stop,
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
    fig = A.plot(show=False)
    A.get_results()
    print(f'finished seed {seed} after {M.plan} steps with pop_imbalance={M.pop_imbalance}')

start = time.time()
seed_start = 200
seeds_per_worker = 8
with multiprocessing.Pool() as pool:
    seeds = list(range(seed_start, seed_start + seeds_per_worker * pool._processes))
    print(seeds)
    pool.map(f, seeds)
elapsed = time.time() - start
h, m = divmod(elapsed, 3600)
m, s = divmod(m, 60)
print(f'{int(h)}hrs {int(m)}min {s:.2f}sec elapsed')