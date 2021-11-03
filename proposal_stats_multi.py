################# Set hashseed for reproducibility #################
HASHSEED = '0'
try:
    notebook
except:
    import os, sys
    if os.getenv('PYTHONHASHSEED') != HASHSEED:
        os.environ['PYTHONHASHSEED'] = HASHSEED
        os.execv(sys.executable, [sys.executable] + sys.argv)

from src.mcmc import *
import multiprocessing, itertools as it

proposals_tbl = f'{data_bq}.TX_2020_proposals'
stats_tbl     = proposals_tbl + '_stats'
proposals_df = read_table(proposals_tbl).query('complete').reset_index(drop=True)
Proposals = proposals_df['proposal']
Levels = ['tabblock', 'bg', 'tract', 'cntyvtd']
Contracts = ['0', '1', 'proposal']
P = list(it.product(Levels, Proposals, Contracts))
assert len(P) == 516
delete_table(stats_tbl)

def get_proposal_stats(level='tabblock', proposal='plans2168', contract=0):
    M = MCMC(level=level, proposal=proposal, contract=contract, refresh_all=('nodes', 'districts', 'graph'))
    return M

def f(idx):
    print(f'running {idx}')
    done = False
    for i in range(1, 60):
        try:
            M = get_proposal_stats(*idx)
            done = True
            break
        except:
            time.sleep(1)
    assert done, f'I tried to run {idx} {i} times without success - giving up'

    df = pd.DataFrame()
    for attr in ['level', 'proposal', 'contract', 'pop_deviation', 'intersect_defect', 'whole_defect', 'defect']:
        df.loc[0, attr] = M[attr]
    attr = 'disconnected_districts'
    df.loc[0, attr] = ','.join(f'{i}' for i in sorted(M[attr]))
    df.loc[0, attr+'_count'] = int(len(M[attr]))
    print(df)
    load_table(stats_tbl, df=df, overwrite=False)
    print(f'finished {idx}')

def multi_f(idx):
    time.sleep(5 * multiprocessing.current_process()._identity[0])
    return f(idx)

# for idx in P[:2]:
#     f(idx)
        
with multiprocessing.Pool() as pool:
    pool.map(multi_f, P)