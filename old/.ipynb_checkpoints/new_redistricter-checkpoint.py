from src import *
from src.data import *
from src.mcmc import *

D = Data()
for district_type, proposals in D.proposals_dict.items():
    for proposal in proposals:
        print(f'\n\n{district_type} {proposal}')
        self = MCMC(level='cntyvtd',
                    district_type=district_type,
                    proposal=proposal,
#                     refresh_tbl = ('nodes', 'graphs')
                   )