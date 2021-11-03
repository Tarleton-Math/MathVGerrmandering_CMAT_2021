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
    'abbr'                 : 'TX',
    'level'                : 'cntyvtd',
    'proposal'             : 'plans2168',
    'contract'             : 'proposal',
    'max_steps'            : 100000,
    'report_period'        : 10,
    'save_period'          : 500,
    'yolo_length'          : 10,
    'election_filters'     : (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'"),
}
run_opts = {
    'seed_start'      : 1000,
    'seed_stop'       : 10000000000000,
    'jobs_per_worker' : 80,
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
    
# task = input('Using options above - do you want to (r)un MCMC, (p)ost-process each run, (c)onsolidate results, or any other to quit: ').lower()

task = 'p'

a = run_opts['seed_start']
b = min(a + run_opts['jobs_per_worker'] * run_opts['workers'], run_opts['seed_stop'])
random_seeds = np.arange(a, b)

if task in ['r', 'run', 'p']:
    def f(random_seed):
        print(f'starting seed {random_seed}')
        M = MCMC(random_seed=random_seed, **opts)
        if task == 'p':
            print('post-processing')
            try:
                M.post_process()
            except Exception as e:
                print(f'{random_seed} exception')
        else:
            M.run_chain()
        return M

    def multi_f(random_seed):
        time.sleep(multiprocessing.current_process()._identity[0] / 100)
        return f(random_seed)


    start_time = time.time()
    M = MCMC(**opts)#, refresh_all=('proposals'))

    if run_opts['workers'] == 1:
        for s in random_seeds:
            M = f(s)
    else:
        with multiprocessing.Pool(run_opts['workers']) as pool:
            M = pool.map(multi_f, random_seeds)
    print(f'total time elapsed = {time_formatter(time.time() - start_time)}')


elif task in ['c', 'consolidate']:
    M = MCMC(**opts)
    M.tbls['final']  = f'{M.dataset}.all'
    M.tbls['hashes'] = f'{M.tbls["final"]}_hashes'
#     src_tbls = dict()
#     for t in bqclient.list_tables(M.dataset):
#         full  = t.full_table_id.replace(':', '.')
#         short = t.table_id
#         w = short.split('_')
#         if len(w) >= 4 and w[0] == M.level and w[1] == M.contract and int(w[2]) in random_seeds:
#             # src_tbls.setdefault(w[2], {}).setdefault(w[3], full)
#             # src_tbls.setdefault(w[2], {}).update({w[3]:full})
#             src_tbls.setdefault(w[2], {})[w[3]] = full
#     src_tbls = {k: v for k, v in src_tbls.items() if len(v)>= 4}
    
#     query = ['\nunion all\n'.join([f"select A.random_seed, A.plan, A.hash as hash_plan from {tbls['summary']} as A" for seed, tbls in src_tbls.items()])]
#     query.append(f"""
# select
#     *,
#     row_number() over (partition by hash_plan order by random_seed asc, plan asc) as r
# from (
#     {subquery(query[-1])}
#     )
# """)
#     query.append(f"""
# select
#     * except (r),
# from (
#     {subquery(query[-1])}
#     )
# where
#     r = 1
# order by
#     random_seed asc, plan asc
# """)
    
#     load_table(tbl=M.tbls['hashes'], query=query[-1])
    
    N = run_query(f"select count(*) from {M.tbls['hashes']}").iloc[0,0]
    print(N)
#     chunks = 10000
#     for start in np.arange(0, N, chunks):
#         print(start)
#         query = list()
#         query.append(f"""
# select
#     *
# from
#     {M.tbls['hashes']}
# limit
#     {chunks}
# offset
#     {start}
# """)
#         load_table(tbl=M.tbls['final'], query=query[-1], overwrite=start==0)


    
#     print(df.head(3))
#     print(df.dtypes)
#     print(df['hash_plan'].value_counts().sort_values())
    
#         summary_cols  = ['hash'     , 'pop_deviation', 'polsby_popper', 'intersect_defect', 'whole_defect', 'defect']
#         district_cols = ['total_pop', 'pop_deviation', 'polsby_popper'                                              , 'aland']
#         county_cols   = [                                               'intersect_defect', 'whole_defect', 'defect']
    
#         query = f"""
# select
#     P.random_seed,
#     P.plan,
#     P.geoid,
#     P.district,
#     N.county,
#     {join_str(1).join([f'S.{c} as {c}_plan'     for c in summary_cols ])},
#     {join_str(1).join([f'D.{c} as {c}_district' for c in district_cols])},
#     {join_str(1).join([f'C.{c} as {c}_county'   for c in county_cols  ])},
#     --N.* except (geoid, district, county),
# from
#     {self.tbls['plan']} as P
# left join
#     {self.tbls['nodes']} as N
# on
#     N.geoid = P.geoid
# left join
#     {self.tbls['summary']} as S
# on
#     S.random_seed = P.random_seed and S.plan = P.plan
# left join
#     {self.tbls['district']} as D
# on
#     D.random_seed = P.random_seed and D.plan = P.plan and D.district = P.district
# left join
#     {self.tbls['county']} as C
# on
#     C.random_seed = P.random_seed and C.plan = P.plan and C.county = N.county
# """        

    
    
#     df = run_query(query[-1])
#     print(df['hash_plan'].value_counts().sort_values())
            
            # tbls.setdefault(w[3], []).append(full)
    # print(src_tbls)
            
        
        # if M.level in short and M.contract in short:
            # print(short)
        
#         print(full)
        # print(short)
    
    
    

################ Post-Processing & Analysis #################
# from src.analysis import *
# start = time.time()
# A = Analysis(nodes_tbl=G.nodes.tbl)#, batch_size=2, max_results=20)
# A.compute_results()
# print(f'analysis took {time_formatter(time.time() - start)}')

# print(f'total time elapsed = {time_formatter(time.time() - start_time)}')