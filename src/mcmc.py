from . import *

@dataclasses.dataclass
class MCMC(Base):
    nodes_tbl             : str
    max_steps             : int = 0
    random_seed           : int = 0
    pop_diff_exp          : int = 2
    pop_deviation_target  : float = 1.0
    pop_deviation_stop    : bool = True
    defect_valid_activate : float = 1000.0
    anneal                : float = 0.0
    save                  : bool = True
    save_period           : int = 500
    report_period         : int = 500
    edge_attrs            : typing.Tuple = ('distance', 'shared_perim')
    node_attrs            : typing.Tuple = ('total_pop', 'aland', 'perim')
    max_results           : int = None
    batch_size            : int = 100
    pop_deviation_thresh  : float = 10.0


    def __post_init__(self):
        self.start_time = time.time()
        w = self.nodes_tbl.split('.')[-1].split('_')[1:]
        self.abbr, self.yr, self.level, self.district_type, self.contract_thresh = w
        self.stem = '_'.join(w)
        self.name = f'{self.stem}_{self.random_seed}'

        self.ds = f'{root_bq}.{self.stem}'
        self.bq = self.ds + f'.{self.name}'
        self.path = root_path / f'results/{self.stem}'
        self.pq = self.path / f'{self.name}.parquet'
        self.gpickle = self.pq.with_suffix('.gpickle')
        self.tbl = f'{self.ds}.{self.stem}_0000000_allresults'
    
        try:
            bqclient.create_dataset(self.ds)
        except:
            pass
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = int(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self.seats_col = f'seats_{self.district_type}'
        self.nodes_df = read_table(self.nodes_tbl, cols=['geoid', 'county', self.district_type, self.seats_col] + list(self.node_attrs)).set_index('geoid')
        self.nodes_df['random_seed'] = self.random_seed
        
        self.plan = 0
        self.get_districts()
        self.total_pop  = self.nodes_df['total_pop'].sum()
        self.target_pop = self.total_pop / Seats[self.district_type]
        self.get_splits()
        self.defect_init = self.defect

        
    def get_districts(self):
        grp = self.nodes_df.groupby(self.district_type)
        self.districts = {k:tuple(sorted(v)) for k,v in grp.groups.items()}
        self.partition = tuple(sorted(self.districts.values()))
        self.hash = self.partition.__hash__()
        
    def edges_to_graph(self, edges):
        return nx.from_pandas_edgelist(edges, source=f'geoid_x', target=f'geoid_y', edge_attr=self.edge_attrs)
    
    def get_components_district(self, district):
        H = self.graph.subgraph(self.nodes_df[self.nodes_df[self.district_type] == district].index)
        return self.get_components(H)
        
    def get_components(self, H):
        return sorted([sorted(tuple(x)) for x in nx.connected_components(H)], key=lambda x:len(x), reverse=True)

    def get_graph(self):
#         rpt(f'getting edges')
        query = f"""
select
    *
from (
    select
        x.geoid as geoid_x,
        y.geoid as geoid_y,        
        st_distance(x.point, y.point) / {meters_per_mile} as distance,
        st_perimeter(st_intersection(x.polygon, y.polygon)) / 2 / {meters_per_mile} as shared_perim
    from
        {self.nodes_tbl} as x,
        {self.nodes_tbl} as y
    where
        x.geoid < y.geoid
        and st_intersects(x.polygon, y.polygon)
    )
where
    shared_perim > 0.01
order by
    geoid_x, geoid_y
"""
        self.edges = run_query(query)
        self.graph = self.edges_to_graph(self.edges)
        
        print(f'connecting districts')
        for D in np.unique(self.nodes_df[self.district_type]):
            while True:
                comp = self.get_components_district(D)
                rpt(f"District {self.district_type} {str(D).rjust(3,' ')} component sizes = {[len(c) for c in comp]}")
                if len(comp) == 1:
                    print('connected')
                    break
                else:
                    print('regrouping to connect components')
                    self.comp = comp
                    for c in comp[1:]:
                        for x in c:
                            y = self.rng.choice(list(self.graph.neighbors(x)))
                            self.nodes_df.loc[x, self.district_type] = self.nodes_df.loc[y, self.district_type]

        new_districts = Seats[self.district_type] - self.nodes_df[self.district_type].nunique()
        M = int(self.nodes_df[self.district_type].max()) + 1
        N = self.nodes_df.nlargest(10 * new_districts, 'total_pop').index.tolist()
        while new_districts > 0:
            n = N.pop(0)
#             rpt(f'try to start district {M} at node {n}')
            D = self.nodes_df.loc[n, self.district_type]
            self.nodes_df.loc[n, self.district_type] = M
            comp = self.get_components_district(D)
            if len(comp) == 1:
#                 rpt('success')
                M += 1
                new_districts -= 1
            else:
#                 rpt('fail')
                self.nodes_df.loc[n, self.district_type] = D

        self.nodes_df['plan'] = self.plan
        nx.set_node_attributes(self.graph, self.nodes_df.to_dict('index'))

                

    def get_stats(self):
        self.get_districts()
        self.stats = pd.DataFrame()
        for d, N in self.districts.items():
            H = self.graph.subgraph(N)
            s = dict()
            s['total_pop'] =   sum(x for a,    x in H.nodes(data='total_pop')    if x is not None)
            internal_perim = 2*sum(x for a, b, x in H.edges(data='shared_perim') if x is not None)
            external_perim =   sum(x for a,    x in H.nodes(data='perim')        if x is not None) - internal_perim
            s['aland']     =   sum(x for a,    x in H.nodes(data='aland')        if x is not None)
            s['polsby_popper'] = 4 * np.pi * s['aland'] / (external_perim**2) * 100
            for k, v in s.items():
                self.stats.loc[d, k] = v
        self.stats['total_pop'] = self.stats['total_pop'].astype(int)
        self.stats['pop_deviation'] = (self.stats['total_pop'] - self.target_pop).abs() / self.target_pop
        self.stats['plan'] = int(self.plan)
        self.stats['random_seed'] = int(self.random_seed)
        self.pop_deviation = (abs(self.stats['total_pop'].max() - self.target_pop) + abs(self.stats['total_pop'].min() - self.target_pop)) / self.target_pop * 100
        return self.stats


    def get_summaries(self):
        self.get_splits()
        self.summaries = pd.DataFrame()
        self.summaries['random_seed'] = [self.random_seed]
        self.summaries['plan'] = [self.plan]
        self.summaries['hash'] = [self.hash]
        self.summaries['polsby_popper']  = [self.stats['polsby_popper'].mean()]
        self.summaries['pop_deviation'] = [self.pop_deviation]
        self.summaries['intersect_defect'] = [self.intersect_defect]
        self.summaries['whole_defect'] = [self.whole_defect]
        self.summaries['defect'] = [self.defect]
        return self.summaries


    def get_splits(self):
        self.splits = self.nodes_df[['county', self.district_type, self.seats_col]].drop_duplicates()
        self.splits['random_seed'] = self.random_seed
        self.splits['plan'] = self.plan

        self.splits = self.splits.groupby(['county', self.district_type])[self.seats_col].sum().reset_index()
        self.splits['whole'] = self.splits.groupby(self.district_type)['county'].transform('count') <= 1
        self.splits = self.splits.groupby('county').agg(whole=('whole', 'sum'), intersect=('whole', 'count'), target=(self.seats_col, 'sum'))
        self.splits['whole_defect'] = (np.floor(self.splits['target']) - self.splits['whole']).abs().astype(int)
        self.splits['intersect_defect'] = (np.ceil(self.splits['target']) - self.splits['intersect']).abs().astype(int)
        self.splits['defect'] = self.splits['whole_defect'] + self.splits['intersect_defect']
        self.intersect_defect = self.splits['intersect_defect'].sum()
        self.whole_defect = self.splits['whole_defect'].sum()
        self.defect = self.splits['defect'].sum()
        return self.splits


    def report(self):
        self.get_splits()
        self.get_stats()
        print(f'random_seed {self.random_seed}: step {self.plan} {time_formatter(time.time() - self.start_time)}, pop_deviation={self.pop_deviation:.1f}, intersect_defect={self.intersect_defect}, whole_defect={self.whole_defect}', flush=True)

        
    def run_chain(self):
        self.get_graph()
        self.plan = 0
        self.overwrite_tbl = True
        self.get_stats()
        self.plans_rec     = [self.nodes_df[['random_seed', 'plan', self.district_type]]]
        self.splits_rec    = [self.get_splits()]
        self.stats_rec     = [self.get_stats()]
        self.summaries_rec = [self.get_summaries()]
        self.hash_rec    = [self.hash]
        for k in range(1, self.max_steps+1):
            self.plan = k
            self.nodes_df['plan'] = self.plan
            msg = f"random_seed {self.random_seed} plan {self.plan} pop_deviation={self.pop_deviation:.1f}"

            if self.recomb():
                self.plans_rec    .append(self.nodes_df[['random_seed', 'plan', self.district_type]])
                self.splits_rec   .append(self.get_splits())
                self.stats_rec    .append(self.get_stats())
                self.summaries_rec.append(self.get_summaries())
                self.hash_rec     .append(self.hash)
                if self.plan % self.report_period == 0:
                    self.report()
                if self.plan % self.save_period == 0:
                    self.save_results()
                if self.pop_deviation_stop:
                    if self.pop_deviation < self.pop_deviation_target:
                        break
            else:
                rpt(msg)
                break
        self.save_results()
        self.report()
        print(f'random_seed {self.random_seed} done')


    def save_results(self):
        nx.write_gpickle(self.graph, self.gpickle)
        to_gcs(self.graph_file)
        
        def reorder(df):
            idx = [c for c in ['random_seed', 'plan'] if c in df.columns]
            return df[idx + [c for c in df.columns if c not in idx]]

        tbls = {f'{nm}_rec': f'{self.bq}_{nm}' for nm in ['plans', 'splits', 'stats', 'summaries']}
        if len(self.plans_rec) > 0:
            self.plans_rec     = pd.concat(self.plans_rec    , axis=0).rename_axis('geoid').reset_index()
            self.splits_rec    = pd.concat(self.splits_rec   , axis=0).rename_axis('geoid').reset_index()
            self.stats_rec     = pd.concat(self.stats_rec    , axis=0).rename_axis(self.district_type).reset_index()
            self.summaries_rec = pd.concat(self.summaries_rec, axis=0)

            for nm, tbl in tbls.items():
                saved = False
                for i in range(1, 60):
                    try:
                        load_table(tbl=tbl, df=reorder(self[nm]), overwrite=self.overwrite_tbl)
                        self[nm] = list()
                        saved = True
                        break
                    except:
                        time.sleep(1)
                assert saved, f'I tried to write the result of random_seed {self.random_seed} {i} times without success - giving up'
            self.overwrite_tbl = False


    def edges_tuple(self, G=None):
        if G is None:
            G = self.graph
        return tuple(sorted(tuple((min(u,v), max(u,v)) for u, v in G.edges)))
        

    def recomb(self):
        def gen(pop_diff):
            while len(pop_diff) > 0:
                pop_diff /= pop_diff.sum()
                a = self.rng.choice(pop_diff.index, p=pop_diff)
                pop_diff.pop(a)
                yield a
        L = self.stats['total_pop']
        pop_diff = pd.DataFrame([(x, y, abs(p-q)) for x, p in L.iteritems() for y, q in L.iteritems() if x < y]).set_index([0,1]).squeeze()
        pop_diff = (pop_diff / pop_diff.sum()) ** self.pop_diff_exp
        pairs = gen(pop_diff)
        while True:
            try:
                d0, d1 = next(pairs)
            except StopIteration:
                rpt(f'exhausted all district pairs - I think I am stuck')
                return False
            except Exception as e:
                raise Exception(f'unknown error {e}')
            m = list(self.districts[d0]+self.districts[d1])  # nodes in d0 or d1
            H = self.graph.subgraph(m).copy()  # subgraph on those nodes
            if not nx.is_connected(H):  # if H is not connect, go to next district pair
#                     rpt(f'{d0},{d1} not connected')
                continue
#                 else:
#                     rpt(f'{d0},{d1} connected')
            P = self.stats['total_pop'].copy()
            p0 = P.pop(d0)
            p1 = P.pop(d1)
            q = p0 + p1
            # q is population of d0 & d1
            # P lists all OTHER district populations
            P_min, P_max = P.min(), P.max()

            trees = []  # track which spanning trees we've tried so we don't repeat failures
            for i in range(100):  # max number of spanning trees to try
                for e in self.edges_tuple(H):
                    H.edges[e]['weight'] = self.rng.uniform()
                T = nx.minimum_spanning_tree(H)  # find minimum spanning tree - we assiged random weights so this is really a random spanning tress
                h = self.edges_tuple(T).__hash__()  # hash tree for comparion
                if h not in trees:  # prevents retrying a previously failed treee
                    trees.append(h)
                    # try to make search more efficient by searching for a suitable cut edge among edges with high betweenness-centrality
                    # Since cutting an edge near the perimeter of the tree is veru unlikely to produce population balance,
                    # we focus on edges near the center.  Betweenness-centrality is a good metric for this.
                    B = nx.edge_betweenness_centrality(T)
                    B = sorted(B.items(), key=lambda x:x[1], reverse=True)  # sort edges on betweenness-centrality (largest first)
                    for e, cent in B:
                        if cent < 0.01:
                            break
                        T.remove_edge(*e)
                        comp = nx.connected_components(T)  # T nows has 2 components
                        next(comp)  # second one tends to be smaller → faster to sum over → skip over the first component
                        s = sum(H.nodes[n]['total_pop'] for n in next(comp))  # sum population in component 2
                        t = q - s  # pop of component 0 (recall q is the combined pop of d0&d1)
                        if s > t:  # ensure s < t
                            s, t = t, s
                        imb_new = (abs(max(t, P_max) - self.target_pop) + abs(min(s, P_min) - self.target_pop)) / self.target_pop * 100  # compute new pop deviation
                        I = self.pop_deviation - imb_new
                        if I < 0:
                            if self.anneal < 1e-7:
                                if I < -0.01:
                                    T.add_edge(*e)  #  if pop_balance not achieved, re-insert e
                                    continue
                            elif self.rng.uniform() > np.exp(I / self.anneal):
                                T.add_edge(*e)  #  if pop_balance not achieved, re-insert e
                                continue
                                
                        comp = self.get_components(T)
                        self.nodes_df['old'] = self.nodes_df[self.district_type].copy()
                        self.nodes_df.loc[comp[0], self.district_type] = d0
                        self.nodes_df.loc[comp[1], self.district_type] = d1
                        
                        if self.pop_deviation < self.defect_valid_activate:
                            self.get_splits()
                            I = self.defect_init - self.defect
                            if I < 0:
                                T.add_edge(*e)
                                self.nodes_df[self.district_type] = self.nodes_df['old']
                                self.get_splits()
                                continue
                            

                        # We found a good cut edge & made 2 new districts.  They will be label with the values of d0 & d1.
                        # But which one should get d0?  This is surprisingly important so colors "look right" in animations.
                        # Else, colors can get quite "jumpy" and give an impression of chaos and instability
                        # To achieve this, add aland of nodes that have the same od & new district label
                        # and subtract aland of nodes that change district label.  If negative, swap d0 & d1.
                        
                        x = H.nodes(data=True)
                        s = (sum(x[n]['aland'] for n in comp[0] if x[n][self.district_type]==d0) -
                             sum(x[n]['aland'] for n in comp[0] if x[n][self.district_type]!=d0) +
                             sum(x[n]['aland'] for n in comp[1] if x[n][self.district_type]==d1) -
                             sum(x[n]['aland'] for n in comp[1] if x[n][self.district_type]!=d1))
                        if s < 0:
                            d0, d1 = d1, d0
                                
                        # Update district labels
                        for n in comp[0]:
                            self.graph.nodes[n][self.district_type] = d0
                        for n in comp[1]:
                            self.graph.nodes[n][self.district_type] = d1
                            
                            
                        # update stats
                        self.get_stats()
                        assert abs(self.pop_deviation - imb_new) < 1e-2, f'disagreement betwen pop_deviation calculations {self.pop_deviation} v {imb}'
                        if self.hash in self.hash_rec: # if we've already seen that plan before, reject and keep trying for a new one
#                             rpt(f'duplicate plan {self.hash}')
                            T.add_edge(*e)
                            # Restore old district labels
                            for n in H.nodes:
                                self.graph.nodes[n][self.district_type] = H.nodes[n][self.district_type]
                            self.get_stats()
                        else:  # if this is a never-before-seen plan, keep it and return happy
#                             rpt(f'recombed {self.district_type} {d0} & {d1} got pop_deviation={self.pop_deviation:.2f}%')
                            return True



    def post_process(self):
        u = '\nunion all\n'
        self.tbls = dict()
        for src_tbl in bqclient.list_tables(self.ds, max_results=self.max_results):
            full  = src_tbl.full_table_id.replace(':', '.')
            short = src_tbl.table_id
            random_seed = short.split('_')[-2]
            key  = short.split('_')[-1]
            if random_seed.isnumeric():
                try:
                    self.tbls[random_seed][key] = full
                except:
                    self.tbls[random_seed] = {key : full}

        self.tbls = {random_seed : tbls for random_seed, tbls in self.tbls.items() if len(tbls)>=3}
        self.cols = [c for c in get_cols(self.nodes_tbl) if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]
        
        def run_batches(query_list, batch_size=self.batch_size, tbl=self.tbl, run=True):
            temp_tbls = list()
            k = 0
            while query_list:
                query = query_list.pop()
                try:
                    query_stack = query_stack + u + query
                except:
                    query_stack = query
                    
                if len(query_list) % batch_size == 0:
                    temp_tbls.append(f'{tbl}_{k}')
                    if run:
                        load_table(tbl=temp_tbls[-1], query=query_stack)
                    print(f'{len(query_list)} remain')
                    del query_stack
                    k += 1
            return temp_tbls


        
        print('stacking hashes into batches')
        self.hash_query_list = [f"""
select
    cast(random_seed as int) as random_seed,
    cast(plan as int) as plan,
    cast(A.hash as int) as hash_plan
from
    {tbls["summaries"]} as A""" for random_seed, tbls in self.tbls.items()]
        self.hash_tbl = f'{self.tbl}_hash'
        self.hash_temp_tbls = run_batches(self.hash_query_list, tbl=self.hash_tbl, run=False)


        print('stacking hash batches')
        self.hash_batch_stack = u.join([f'select * from {tbl}' for tbl in self.hash_temp_tbls])
        self.hash_batch_stack = f"""
select
    *
from (
    select
        *,
        row_number() over (partition by hash_plan order by plan asc, random_seed asc) as r
    from (
        {subquery(self.hash_batch_stack, indents=1)}
        )
    )
where
    r = 1
"""
#         load_table(tbl=self.hash_tbl, query=self.hash_batch_stack)



        print('joining tables in batches')
        self.join_query_list = [f"""
select
    A.random_seed,
    A.plan,
    A.{self.district_type},
    A.geoid,
    C.hash_plan,
    C.whole_defect_plan,
    C.intersect_defect_plan,
    C.defect_plan,
    --C.pop_imbalance_plan,
    C.pop_deviation_plan,
    B.pop_deviation_district,
    C.polsby_popper_plan,
    B.polsby_popper_district,
    B.aland,
    B.total_pop,
from (
    select
        cast(random_seed as int) as random_seed,
        cast(plan as int) as plan,
        cast({self.district_type} as int) as {self.district_type},
        geoid,
    from
        {tbls['plans']}
    ) as A
inner join (
    select
        cast(random_seed as int) as random_seed,
        cast(plan as int) as plan,
        cast({self.district_type} as int) as {self.district_type},
        aland,
        --pop_deviation as pop_deviation_district,
        polsby_popper as polsby_popper_district,
        total_pop
    from
        {tbls['stats']}
    ) as B
on
    A.random_seed = B.random_seed and A.plan = B.plan and A.{self.district_type} = B.{self.district_type}
inner join (
    select
        X.*,
        Y.hash_plan
    from (
        select
            cast(random_seed as int) as random_seed,
            cast(plan as int) as plan,
            pop_deviation as pop_deviation_plan,
            --pop_imbalance as pop_imbalance_plan,
            whole_defect as whole_defect_plan,
            intersect_defect as intersect_defect_plan,
            defect as defect_plan,
            polsby_popper as polsby_popper_plan
        from
            {tbls['summaries']}
        where
            pop_deviation < {self.pop_deviation_thresh}
            --pop_imbalance < {self.pop_deviation_thresh}
        ) as X
    inner join
        {self.hash_tbl} as Y
    on
        X.random_seed = Y.random_seed and X.plan = Y.plan
    ) as C
on
    B.random_seed = C.random_seed and B.plan = C.plan
""" for random_seed, tbls in self.tbls.items()]
        self.join_tbl = f'{self.tbl}_join'
        self.join_temp_tbls = run_batches(self.join_query_list, tbl=self.join_tbl, run=False)



        print('stacking joined table batches')
        self.join_batch_stack = u.join([f'select * from {tbl}' for tbl in self.join_temp_tbls])
        self.stack_tbl = f'{self.tbl}_stack'
#         load_table(tbl=self.stack_tbl, query=self.join_batch_stack)



        print('joining and aggregating data')
        self.final_query = f"""
select
    A.random_seed,
    A.plan,
    A.{self.district_type},
    max(A.hash_plan) as hash_plan,
    max(A.pop_deviation_plan) as pop_deviation_plan,
    --max(A.pop_imbalance_plan) as pop_imbalance_plan,
    max(nodes_plan) as nodes_plan,
    count(*) as nodes_district,
    max(A.polsby_popper_plan) as polsby_popper_plan,
    max(A.polsby_popper_district) as polsby_popper_district,
    max(A.aland) as aland,
    max(A.total_pop) as total_pop,
    case when max(A.aland) > 0 then max(A.total_pop) / max(A.aland) else 0 end as density,
    {join_str(1).join([f'sum(B.{c}) as {c}' for c in self.cols])}
from (
    select
        *,
        count(*) over (partition by random_seed, plan) as nodes_plan
    from
        {self.stack_tbl}
    ) as A
inner join
    {self.nodes_tbl} as B
on
    A.geoid = B.geoid
group by
    random_seed, plan, {self.district_type}
"""
        load_table(tbl=self.tbl, query=self.final_query)
