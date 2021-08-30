from src import *


################# Define parameters #################


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

mcmc_opts = {
    'user_name'          : 'cook',
    'random_seed'        : 1,
    'num_steps'          : 10000,
    'pop_imbalance_tol'  : 10.0,
    'pop_imbalance_stop' : True,
    'new_districts'      : 2,
    'num_colors'         : 10,
    'district_type'      : graph_opts['district_type'],
    'gpickle'            : '/home/jupyter/redistricting_data/graph/TX/graph_TX_2020_cntyvtd_cd.gpickle'
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

# from src.graph import *
# G = Graph(**graph_opts)
# del G


################# Run MCMC #################


from src.mcmc import *
for s in range(50):
    mcmc_opts['random_seed'] = s
    print(f"random seed {mcmc_opts['random_seed']}")
    M = MCMC(**mcmc_opts)
    start = time.time()
    M.run_chain()
    elapsed = time.time() - start
    h, m = divmod(elapsed, 3600)
    m, s = divmod(m, 60)
    print(f'{int(h)}hrs {int(m)}min {s:.2f}sec elapsed')