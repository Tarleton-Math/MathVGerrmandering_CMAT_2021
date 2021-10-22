from src import *
from src.data import *
from src.space import *
from src.mcmc import *

# https://dvr.capitol.texas.gov/
level    = 'cntyvtd'
proposal = 'planc2122'  # for 2010 enacted, use 'planc2100' for US congress, 'plans2100' for TX Senate, or 'planh2100' for TX House
contract = 2

# D = Data()
S = Space(level=level, proposal=proposal, contract=contract)
M = MCMC(gpickle=S.gpickle, nodes=S.tbls['nodes'],
         max_steps   = 0,
         random_seed = 0,
         defect_cap  = 1000000,
         pop_deviation_target=10)
