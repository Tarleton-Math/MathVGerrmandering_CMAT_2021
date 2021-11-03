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
    # 'proposal'             : 'plans2168',
    'proposal'             : 'planh2316',
    'contract'             : 'proposal',
    'max_steps'            : 3000,
    'report_period'        : 100,
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
    'jobs_per_worker' : 1,
    'workers'         : 80,
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

task = input('Using options above - do you want to (r)un MCMC, (p)ost-process each run, (c)onsolidate results, or any other to quit: ').lower()

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
    start = time.time()
    
    src_tbls = dict()
    for t in bqclient.list_tables(M.dataset):
        full  = t.full_table_id.replace(':', '.')
        short = t.table_id
        w = short.split('_')
        try:
            assert len(w) >= 4 and w[0] == M.level and w[1] == M.contract and int(w[2]) in random_seeds
            src_tbls.setdefault(w[2], {})[w[3]] = full
        except:
            pass
    src_tbls = {k: v for k, v in src_tbls.items() if len(v)>= 4}

    stem = f'{M.dataset}.{M.level}_{M.contract}_0'
    final = {key : f'{stem}_final_{key}' for key in ['hashes', 'plan', 'district', 'county', 'summary', 'stats']}
    cols = get_cols(M.tbls['all'])
    a = cols.index('seats_cd')
    b = cols.index('polygon')
    data_cols = cols[a:b]
    data_sums = [f'sum({c}) as {c}' for c in data_cols]

    query = ['\nunion all\n'.join([f"select A.random_seed, A.plan, A.hash as hash_plan from {tbls['summary']} as A" for seed, tbls in src_tbls.items()])]
    query.append(f"""
select
    *,
    row_number() over (partition by hash_plan order by random_seed asc, plan asc) as r
from (
    {subquery(query[-1])}
    )
""")
    query.append(f"""
select
    * except (r),
from (
    {subquery(query[-1])}
    )
where
    r = 1
order by
    random_seed asc, plan asc
""")
    val = final['hashes']
    load_table(tbl=val, query=query[-1])
    # N = run_query(f"select count(*) from {M.tbls['final_hashes']}").iloc[0,0]
    
    k = 0
    overwrite = True
    for seed, tbls in src_tbls.items():
        rpt(f'starting {seed}')
        for key, val in final.items():
            if key not in ['hashes', 'stats']:
                rpt(key)
                query = f"""
select
    A.*
from
    {tbls[key]} as A
inner join
    {final['hashes']} as B
on
    A.random_seed = B.random_seed and A.plan = B.plan
"""
                load_table(tbl=val, query=query, overwrite=overwrite)
            
        key = 'stats'
        rpt(key)
        query = list()
        query.append(f"""
select
    A.*,
from
    {tbls['plan']} as A
inner join
    {final['hashes']} as B
on
    A.random_seed = B.random_seed and A.plan = B.plan
""")
        query.append(f"""
select
    A.random_seed,
    A.plan,
    A.district,
    {join_str().join(data_sums)},
from (
    {subquery(query[-1])}
    ) as A
inner join
    {M.tbls['all']} as B
on
    A.geoid = B.geoid
group by
    1, 2, 3
""")
        query.append(f"""
select
    B.* except (total_pop),
    A.* except (random_seed, plan,district),
from (
    {subquery(query[-1])}
    ) as A
inner join
    {tbls['district']} as B
on
    A.random_seed = B.random_seed and A.plan = B.plan and A.district = B.district
""")    
        load_table(tbl=final[key], query=query[-1], overwrite=overwrite)
        overwrite = False
        
        print(f'finshed {seed}')
        k += 1
        # if k >= 2:
        #     break

    print(f'analysis took {time_formatter(time.time() - start)}')