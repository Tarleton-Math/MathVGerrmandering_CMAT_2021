from src.mcmc import *
import multiprocessing, itertools as it
workers = 2

proposals_tbl = f'{data_bq}.TX_2020_proposals'
stats_tbl     = proposals_tbl + '_stats'
proposals_df = read_table(proposals_tbl).query('complete').reset_index(drop=True)
Proposals = proposals_df['proposal']
Levels = ['tabblock', 'bg', 'tract', 'cntyvtd']
# Contracts = ['0', '1', 'proposal']
# Contracts = ['0']
Contracts = ['1', 'proposal']
P = list(it.product(Levels, Proposals, Contracts))

idx_cols = ['level', 'proposal', 'contract']
try:
    stats_df = read_table(stats_tbl)
except:
    stats_df = pd.DataFrame(columns=idx_cols)
stats_df.set_index(idx_cols, inplace=True)

def get_proposal_stats(level='tabblock', proposal='plans2168', contract=0):
    M = MCMC(level=level, proposal=proposal, contract=contract, refresh_all=('nodes', 'districts', 'graph'))
    return M

def f(idx):
    print()
    if idx in stats_df.index:
        print(f'skipping {idx} - already in stats_df')
    else:
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

        df = stats_df[0:0].copy()
        for attr in ['pop_deviation', 'intersect_defect', 'whole_defect', 'defect']:
            rpt(f'{attr}={M[attr]}')
            df.loc[idx, attr] = M[attr]
        attr = 'disconnected_districts'
        df.loc[idx, attr] = ','.join(f'{i}' for i in sorted(M[attr]))
        df.loc[idx, attr+'_count'] = int(len(M[attr]))
        load_table(stats_tbl, df=df.reset_index(), overwrite=False)
        print(f'finished {idx}')

def multi_f(idx):
    time.sleep(5 * multiprocessing.current_process()._identity[0])
    return f(idx)

# for idx in P[:2]:
#     f(idx)
        
with multiprocessing.Pool() as pool:
    pool.map(multi_f, P)