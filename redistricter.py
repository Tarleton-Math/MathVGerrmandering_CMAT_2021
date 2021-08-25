from src import *
from src.gerry import Gerry
from src.crosswalks import Crosswalks

refresh_all = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'votes_all',
#     'votes_hl',
#     'combined',
#     'edges',
#     'nodes',
#     'graph',
)

refresh_tbl = (
#     'crosswalks',
#     'assignments',
#     'shapes',
#     'census',
#     'elections',
#     'votes_all',
#     'votes_hl',
#     'combined',
#     'edges',
#     'nodes',
#     'graph',
)

self = Gerry(abbr = 'TX',
             level = 'cntyvtd',
             district_type='cd',
             simplification=0,
             num_colors=10,
             election_filters=(
                 "office='President' and race='general'",
                 "office='USSen' and race='general'",
#                  "office like 'USRep%' and race='general'",
             ),
             refresh_tbl=refresh_tbl, refresh_all=refresh_all,
             num_steps=10,
            )

start = time.time()

idx = self.nodes.df.nlargest(2, 'total_pop').index
self.nodes.df.loc[idx[0], 'cd'] = '37'
self.nodes.df.loc[idx[1], 'cd'] = '38'

self.MCMC()
# self.agg_plans(agg_polygon_steps=False)
# #                (0,10))#True)#agg_polygon_steps=list(range(3,15)))
# self.stack_plans()

elapsed = time.time() - start
h, m = divmod(elapsed, 3600)
m, s = divmod(m, 60)
print(f'{int(h)}hrs {int(m)}min {s:.2f}sec elapsed')