from src import *
from src.graph import *

refresh_all = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes',
#     'edges',
#     'graph',
)

refresh_tbl = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'nodes',
#     'edges',
#     'graph',
)

graph_opts = {
    'abbr'          : 'TX',
    'level'         : 'cntyvtd',
    'district_type' : 'cd',
    'refresh_tbl'   : refresh_tbl,
    'refresh_all'   : refresh_all,
    'election_filters' : (
        "office='President' and race='general'",
        "office='USSen' and race='general'",
#         "office like 'USRep%' and race='general'",
    ),
}
G = Graph(**graph_opts)
gpickle = G.gpickle
del G

mcmc_opts = {
    'num_colors'    : 10,
    'num_steps'     : 1000,
    'district_type' : graph_opts['district_type'],
}

from src.mcmc import *
M = MCMC(gpickle, **mcmc_opts)
start = time.time()
M.run_chain()
elapsed = time.time() - start
h, m = divmod(elapsed, 3600)
m, s = divmod(m, 60)
print(f'{int(h)}hrs {int(m)}min {s:.2f}sec elapsed')