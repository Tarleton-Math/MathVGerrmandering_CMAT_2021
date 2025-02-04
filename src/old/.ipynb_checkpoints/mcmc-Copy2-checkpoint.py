from . import *

@dataclasses.dataclass
class MCMC(Base):
    nodes_tbl             : str
    max_steps             : int = 0
    random_seed           : int = 0
    pop_diff_exp          : int = 2
    pop_deviation_target  : float = 10.0
    pop_deviation_stop    : bool = False
    defect_multiplier     : float = 1.0
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
        self.node_attrs = listify(self.node_attrs) + [self.district_type]
        self.nodes_df = read_table(self.nodes_tbl, cols=['geoid', 'county', self.seats_col] + list(self.node_attrs)).set_index('geoid')
        self.nodes_df['random_seed'] = self.random_seed
        
        self.step = 0
        self.total_pop  = self.nodes_df['total_pop'].sum()
        self.target_pop = self.total_pop / Seats[self.district_type]
        
        grp = self.nodes_df.groupby('county')
        self.counties = {k:{'nodes':tuple(sorted(v.index)),
                            'total_pop'       :              v['total_pop'].sum(),
                            'seats_share'     :              v[self.seats_col].sum(),
                            'whole_target'    : int(np.floor(v[self.seats_col].sum())),
                            'intersect_target': int(np.ceil (v[self.seats_col].sum())),
                            } for k,v in grp}
        
        self.get_graph()
        self.get_stats()
        self.get_plan()

        self.get_defect()
        self.defect_init = self.defect
        self.defect_cap = int(self.defect_multiplier * self.defect_init)
#         print(f'defect_init = {self.defect_init}, setting ceiling for mcmc of {self.defect_cap}')

        
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
        self.graph = nx.from_pandas_edgelist(self.edges, source=f'geoid_x', target=f'geoid_y', edge_attr=self.edge_attrs)
        nx.set_node_attributes(self.graph, self.nodes_df.to_dict('index'))
        grp = self.nodes_df.groupby(self.district_type)
        self.districts = {k:{'nodes':tuple(sorted(v))} for k,v in grp.groups.items()}
        
        new_districts = Seats[self.district_type] - len(self.districts)
        new_district_starts = self.nodes_df.nlargest(10 * new_districts, 'total_pop').index.tolist()
        del self.nodes_df

        self.get_districts()        
#         print(f'connecting districts')
        connected = False
        while not connected:
            connected = True
            for d, data in self.districts.items():
                comp = self.get_components_district(d)
#                 rpt(f"District {self.district_type} {str(D).rjust(3,' ')} component sizes = {[len(c) for c in comp]}")
                if len(comp) > 1:
#                     print('regrouping to connect components')
                    connected = False
                    for c in comp[1:]:
                        for x in c:
                            y = self.rng.choice(list(self.graph.neighbors(x)))
                            self.graph.nodes[x][self.district_type] = self.graph.nodes[y][self.district_type]
        self.get_districts()

        d_new = int(max(self.districts.keys())) + 1 
        while new_districts > 0:
            n = new_district_starts.pop(0)
#             rpt(f'try to start district {M} at node {n}')
            d_old = self.graph.nodes[n][self.district_type]
            self.graph.nodes[n][self.district_type] = d_new
            comp = self.get_components_district(d_old)
            if len(comp) == 1:
#                 rpt('success')
                self.districts[d_new] = dict()            
                d_new += 1
                new_districts -= 1
            else:
#                 rpt('fail')
                self.graph.nodes[n][self.district_type] = d_old
        self.get_districts()
        
        
    def get_components_district(self, district):
        H = nx.subgraph_view(self.graph, lambda n: self.graph.nodes[n][self.district_type] == district)
        return self.get_components(H)


    def get_components(self, H):
        return sorted([sorted(tuple(x)) for x in nx.connected_components(H)], key=lambda x:len(x), reverse=True)


    def get_districts(self):
        for d, data in self.districts.items():
            data['nodes'] = list()
        for n, d in self.graph.nodes(data=self.district_type):
            self.districts[d]['nodes'].append(n)
        for d, data in self.districts.items():
            data['nodes'] = tuple(sorted(tuple(set(data['nodes']))))
        self.partition = tuple(sorted((data['nodes'] for d, data in self.districts.items())))
        self.hash = self.partition.__hash__()


    def get_stats(self, return_df=False):
        self.get_districts()
        attrs = ['total_pop', 'aland', 'perim']
        for d, data in self.districts.items():
            for a in attrs:
                data[a] = 0
            data['external_perim'] = 0

        for n, data in self.graph.nodes(data=True):
            d = data[self.district_type]
            for a in attrs:
                self.districts[d][a] += data[a]

        dev_max = 0
        dev_min = 10000
        self.polsby_popper = 0
        for d, data in self.districts.items():
            H = self.graph.subgraph(data['nodes'])
            data['internal_perim'] = sum(x for a, b, x in H.edges(data='shared_perim'))
            data['external_perim'] = data['perim'] - 2 * data['internal_perim']
            data['polsby_popper'] = 4 * np.pi * data['aland'] / (data['external_perim']**2) * 100
            data['pop_deviation'] = (data['total_pop'] - self.target_pop) / self.target_pop * 100
            self.polsby_popper += data['polsby_popper']
            dev_max = max(dev_max, data['pop_deviation'])
            dev_min = min(dev_min, data['pop_deviation'])
        self.pop_deviation = abs(dev_max) + abs(dev_min)
        if return_df:
            df = pd.DataFrame.from_dict(self.districts, orient='index').drop(columns=['nodes', 'county_intersect'])
            df.insert(0, 'step', int(self.step))
            df.insert(0, 'random_seed', int(self.random_seed))
            return df
            

    def get_defect(self, return_df=False):
        self.get_districts()
        for d, data in self.districts.items():
            data['county_intersect'] = list()
        
        for c, data_county in self.counties.items():
            data_county['district_intersect'] = list()
            for n in data_county['nodes']:
                d = self.graph.nodes[n][self.district_type]
                data_county['district_intersect'].append(d)
                self.districts[d]['county_intersect'].append(c)
            data_county['district_intersect'] = tuple(set(data_county['district_intersect']))
            
        for d, data in self.districts.items():
            data['county_intersect'] = set(data['county_intersect'])
            data['whole'] = len(data['county_intersect']) == 1
        
        self.intersect_defect = 0
        self.whole_defect = 0
        self.defect = 0
        for c, data in self.counties.items():
            data['intersect'] = 0
            data['whole'] = 0
            for d in data['district_intersect']:
                data['intersect'] += 1
                data['whole'] += self.districts[d]['whole']
            data['intersect_defect'] = abs(data['intersect'] - data['intersect_target'])
            data['whole_defect']     = abs(data['whole']     - data['whole_target'])
            data['defect'] = data['intersect_defect'] + data['whole_defect']
            self.intersect_defect += data['intersect_defect']
            self.whole_defect += data['whole_defect']
            self.defect += data['defect']
        if return_df:
            df = pd.DataFrame.from_dict(self.counties, orient='index').drop(columns=['nodes', 'district_intersect']).reset_index()
            df.insert(0, 'step', int(self.step))
            df.insert(0, 'random_seed', int(self.random_seed))
            return df


    def get_plan(self):
        P = {n:{'random_seed':self.random_seed, 'step':self.step, self.district_type:d} for n, d in self.graph.nodes(data=self.district_type)}
        df = pd.DataFrame.from_dict(P, orient='index')
        return df


    def get_summaries(self):
        self.get_stats()
        self.get_defect()
        self.summaries = pd.DataFrame()
        self.summaries['random_seed'] = [self.random_seed]
        self.summaries['step'] = [self.step]
        self.summaries['hash'] = [self.hash]
        self.summaries['polsby_popper']  = [self.polsby_popper]
        self.summaries['pop_deviation'] = [self.pop_deviation]
        self.summaries['intersect_defect'] = [self.intersect_defect]
        self.summaries['whole_defect'] = [self.whole_defect]
        self.summaries['defect'] = [self.defect]
        return self.summaries


    def report(self):
        self.get_defect()
        self.get_stats()
        print(f'random_seed {self.random_seed}: step {self.step} {time_formatter(time.time() - self.start_time)}, pop_deviation={self.pop_deviation:.1f}, intersect_defect={self.intersect_defect}, whole_defect={self.whole_defect}', flush=True)

        
    def run_chain(self):
        self.step = 0
        self.overwrite_tbl = True
        self.get_stats()
        self.plans_rec     = [self.get_plan()]
        self.defect_rec    = [self.get_defect(return_df=True)]
        self.stats_rec     = [self.get_stats(return_df=True)]
        self.summaries_rec = [self.get_summaries()]
        self.hash_rec    = [self.hash]
        for k in range(1, self.max_steps+1):
            self.step = k
            msg = f"random_seed {self.random_seed} step {self.step} pop_deviation={self.pop_deviation:.1f}"

            if self.recomb():
                self.plans_rec    .append(self.get_plan())
                self.defect_rec   .append(self.get_defect(return_df=True))
                self.stats_rec    .append(self.get_stats(return_df=True))
                self.summaries_rec.append(self.get_summaries())
                self.hash_rec     .append(self.hash)
                if self.step % self.report_period == 0:
                    self.report()
                if self.step % self.save_period == 0:
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
        to_gcs(self.gpickle)
        
        def reorder(df):
            idx = [c for c in ['random_seed', 'step'] if c in df.columns]
            return df[idx + [c for c in df.columns if c not in idx]]

        tbls = {f'{nm}_rec': f'{self.bq}_{nm}' for nm in ['plans', 'defect', 'stats', 'summaries', 'params']}
        if len(self.plans_rec) > 0:
            self.plans_rec     = pd.concat(self.plans_rec    , axis=0).rename_axis('geoid').reset_index()
            self.defect_rec    = pd.concat(self.defect_rec   , axis=0).rename_axis('geoid').reset_index()
            self.stats_rec     = pd.concat(self.stats_rec    , axis=0).rename_axis(self.district_type).reset_index()
            self.summaries_rec = pd.concat(self.summaries_rec, axis=0)
            self.params_rec    = pd.DataFrame()
            for p in ['random_seed', 'max_steps', 'pop_diff_exp', 'defect_multiplier', 'anneal', 'pop_deviation_target', 'pop_deviation_stop', 'report_period', 'save_period']:
                self.params_rec[p] = [self[p]]

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
        self.graph_old = self.graph.copy()
        def gen(pop_diff):
            while len(pop_diff) > 0:
                pop_diff /= pop_diff.sum()
                a = self.rng.choice(pop_diff.index, p=pop_diff)
                pop_diff.pop(a)
                yield a
        df = self.get_stats(return_df=True)
        L = df['total_pop']
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
            m = list(self.districts[d0]['nodes'] + self.districts[d1]['nodes'])  # nodes in d0 or d1
            H = self.graph.subgraph(m).copy()  # subgraph on those nodes
            if not nx.is_connected(H):  # if H is not connect, go to next district pair
#                     rpt(f'{d0},{d1} not connected')
                continue
#                 else:
#                     rpt(f'{d0},{d1} connected')
            P = df['total_pop'].copy()
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
                        pop_deviation_new = (abs(max(t, P_max) - self.target_pop) + abs(min(s, P_min) - self.target_pop)) / self.target_pop * 100  # compute new pop deviation
                        
                        def accept(comp):
                            for n in comp[0]:
                                self.graph.nodes[n][self.district_type] = d0
                            for n in comp[1]:
                                self.graph.nodes[n][self.district_type] = d1
                            self.get_districts()
                            novel = self.hash not in self.hash_rec
                            if not novel: # if we've already seen that plan before, reject and keep trying for a new one
#                             rpt(f'duplicate plan {self.hash}')
                                reject(comp)
                                self.get_districts()
                            return novel
                            
                        def reject(comp):
                            T.add_edge(*e)
                            for n in comp[0] + comp[1]:
                                self.graph.nodes[n][self.district_type] = self.graph_old.nodes[n]
                        
                        comp = self.get_components(T)
                        if self.pop_deviation > self.pop_deviation_target:
                            if pop_deviation_new > self.pop_deviation:
                                T.add_edge(*e)
                                continue
                            else:
                                if accept(comp):
                                    self.get_defect()
                                    if self.defect > self.defect_cap:
                                        reject(comp)
                                        self.get_defect()
                                        continue
                        else:
                            if pop_deviation_new > self.pop_deviation_target:
                                T.add_edge(*e)
                                continue
                            else:
                                defect_old = self.defect
                                if accept(comp):
                                    self.get_defect()
                                    if self.defect > defect_old:
                                        reject(comp)
                                        self.get_defect()
                                        continue
                        self.get_stats()
                        assert abs(self.pop_deviation - pop_deviation_new) < 1e-2, f'disagreement betwen pop_deviation calculations {self.pop_deviation} v {pop_deviation_new}'

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
                            comp[0], comp[1] = comp[1], comp[0]
                            accept(comp)
                        self.get_defect()
                        self.get_stats()
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
    random_seed,
    step,
    A.hash as hash_plan
from
    {tbls["summaries"]} as A""" for random_seed, tbls in self.tbls.items()]
        self.hash_tbl = f'{self.tbl}_hash'
        self.hash_temp_tbls = run_batches(self.hash_query_list, tbl=self.hash_tbl, run=True)


        print('stacking hash batches')
        self.hash_batch_stack = u.join([f'select * from {tbl}' for tbl in self.hash_temp_tbls])
        self.hash_batch_stack = f"""
select
    * except (r)
from (
    select
        *,
        row_number() over (partition by hash_plan order by step asc, random_seed asc) as r
    from (
        {subquery(self.hash_batch_stack, indents=1)}
        )
    )
where
    r = 1
"""
        load_table(tbl=self.hash_tbl, query=self.hash_batch_stack)


        print('joining tables in batches')
        self.join_query_list = [f"""
select
    H.random_seed,
    H.step,
    H.{self.district_type},
    I.hash as hash_plan,
    J.* except (random_seed),
    I.whole_defect as whole_defect_plan,
    I.intersect_defect as intersect_defect_plan,
    I.defect as defect_plan,
    I.pop_deviation as pop_deviation_plan,
    H.pop_deviation as pop_deviation_district,
    I.polsby_popper as polsby_popper_plan,
    H.polsby_popper as polsby_popper_district,
    H.aland,
    H.total_pop,
    case when H.aland > 0 then H.total_pop / H.aland else 0 end as density,
    G.* except (random_seed, step, {self.district_type})
from (
    select
        E.* except (geoid),
        {join_str(2).join([f'sum(F.{c}) as {c}' for c in self.cols])}
    from (
        select
            D.*
        from (
            select
                A.random_seed,
                A.step
            from
                {self.hash_tbl} as A
            inner join
                {tbls['summaries']} as B
            on
                A.random_seed = B.random_seed and A.step = B.step
            where
                B.pop_deviation < {self.pop_deviation_thresh}
            ) as C
        inner join
            {tbls['plans']} as D
        on
            C.random_seed = D.random_seed and C.step = D.step
        ) as E
    inner join
        {self.nodes_tbl} as F
    on
        E.geoid = F.geoid
    group by
        random_seed, step, {self.district_type}
    ) as G
inner join
    {tbls['stats']} as H
on
    G.random_seed = H.random_seed and G.step = H.step and G.{self.district_type} = H.{self.district_type}
inner join
    {tbls['summaries']} as I
on
    H.random_seed = I.random_seed and H.step = I.step
inner join
    {tbls['params']} as J
on
    I.random_seed = J.random_seed
""" for random_seed, tbls in self.tbls.items()]
        self.join_tbl = f'{self.tbl}_join'
        self.join_temp_tbls = run_batches(self.join_query_list, tbl=self.join_tbl, run=True)

        print('stacking joined table batches')
        self.join_batch_stack = u.join([f'select * from {tbl}' for tbl in self.join_temp_tbls])
        self.stack_tbl = f'{self.tbl}_stack'
        load_table(tbl=self.tbl, query=self.join_batch_stack)
        for tbl in self.hash_temp_tbls:
            delete_table(tbl)
        for tbl in self.join_temp_tbls:
            delete_table(tbl)


# select
#     H.random_seed,
#     H.plan,
#     H.{self.district_type},
#     I.hash as hash_plan,
#     I.whole_defect as whole_defect_plan,
#     I.intersect_defect as intersect_defect_plan,
#     I.defect as defect_plan,
#     I.pop_deviation as pop_deviation_plan,
#     H.pop_deviation as pop_deviation_district,
#     I.polsby_popper as polsby_popper_plan,
#     H.polsby_popper as polsby_popper_district,
#     H.aland,
#     H.total_pop,
#     case when H.aland > 0 then H.total_pop / H.aland else 0 end as density
#     G.* except (random_seed, plan, {self.district_type})
# from (
#     select
#         E.*,
#         {join_str(2).join([f'sum(F.{c}) as {c}' for c in self.cols])}
#     from (
#         select
#             D.*
#         from (
#             select
#                 A.random_seed,
#                 A.plan
#             from
#                 self.hash_tbl as A
#             inner join
#                 tbl['summaries'] as B
#             on
#                 A.random_seed = B.random_seed and A.plan = B.plan
#             where
#                 B.pop_deviation < {self.pop_deviation_thresh}
#             ) as C
#         inner join
#             tbl['plan'] as D
#         on
#             B.random_seed = C.random_seed and B.plan = C.plan
#         ) as E
#     inner join
#         self.nodes_tbl as F
#     on
#         E.geoid = F.geoid
#     group by
#         random_seed, plan, {self.district_type}
#     ) as G
# inner join
#     {tbls['stats']} as H
# on
#     G.random_seed = H.random_seed and G.plan = H.plan and G.{self.district_type} = H.{self.district_type}
# inner join
#     {tbls['summaries']} as I
# on
#     H.random_seed = I.random_seed and H.plan = I.plan


        
        
        

# query
#     *
# from
#     tbl['plan'] as A
# inner join
#     self.hash_tbl as B
# on
#     A.random_seed = B.random_seed and A.plan = B.plan
# inner join
#     tbl['summaries'] as C
# on
#     B.random_seed = C.random_seed and B.plan = C.plan



        
        
        

#         print('joining and aggregating data')
#         self.final_query = f"""
# select
#     D.*,
#     C.* except (random_seed, plan, {self.district_type})
# from (
#     select
#         A.random_seed,
#         A.plan,
#         A.{self.district_type},
#         {join_str(1).join([f'sum(B.{c}) as {c}' for c in self.cols])}
#     from
#         {self.stack_tbl} as A
#     inner join
#         {self.nodes_tbl} as B
#     on
#         A.geoid = B.geoid
#     group by
#         random_seed, plan, {self.district_type}
#     ) as C
# inner join
#     {self.stack_tbl} as D
# on
#     C.random_seed = D.random_seed and C.plan = D.plan and C.{self.district_type} = D.{self.district_type}
# """
#         load_table(tbl=self.tbl, query=self.final_query)
        
#         for tbl in self.hash_temp_tbls:
#             delete_table(tbl)
#         for tbl in self.join_temp_tbls:
#             delete_table(tbl)
#         delete_table(self.stack_tbl)
    
    
#         self.final_query = f"""
# select
#     A.random_seed,
#     A.plan,
#     A.{self.district_type},
    
#     max(A.hash_plan) as hash_plan,
#     max(A.pop_deviation_plan) as pop_deviation_plan,
#     max(A.polsby_popper_plan) as polsby_popper_plan,
#     max(A.polsby_popper_district) as polsby_popper_district,
#     max(A.aland) as aland,
#     max(A.total_pop) as total_pop,
#     case when max(A.aland) > 0 then max(A.total_pop) / max(A.aland) else 0 end as density,
#     {join_str(1).join([f'sum(B.{c}) as {c}' for c in self.cols])}
# from
#     {self.stack_tbl} as A
# inner join
#     {self.nodes_tbl} as B
# on
#     A.geoid = B.geoid
# group by
#     random_seed, plan, {self.district_type}
# """
#         load_table(tbl=self.tbl, query=self.final_query)

        
        
        
        
        
        
#     def post_process(self):
#         u = '\nunion all\n'
#         self.tbls = dict()
#         for src_tbl in bqclient.list_tables(self.ds, max_results=self.max_results):
#             full  = src_tbl.full_table_id.replace(':', '.')
#             short = src_tbl.table_id
#             random_seed = short.split('_')[-2]
#             key  = short.split('_')[-1]
#             if random_seed.isnumeric():
#                 try:
#                     self.tbls[random_seed][key] = full
#                 except:
#                     self.tbls[random_seed] = {key : full}

#         self.tbls = {random_seed : tbls for random_seed, tbls in self.tbls.items() if len(tbls)>=3}
#         self.cols = [c for c in get_cols(self.nodes_tbl) if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]
        
#         def run_batches(query_list, batch_size=self.batch_size, tbl=self.tbl, run=True):
#             temp_tbls = list()
#             k = 0
#             while query_list:
#                 query = query_list.pop()
#                 try:
#                     query_stack = query_stack + u + query
#                 except:
#                     query_stack = query
                    
#                 if len(query_list) % batch_size == 0:
#                     temp_tbls.append(f'{tbl}_{k}')
#                     if run:
#                         load_table(tbl=temp_tbls[-1], query=query_stack)
#                     print(f'{len(query_list)} remain')
#                     del query_stack
#                     k += 1
#             return temp_tbls


        
#         print('stacking hashes into batches')
#         self.hash_query_list = [f"""
# select
#     random_seed,
#     plan,
#     A.hash as hash_plan
# from
#     {tbls["summaries"]} as A""" for random_seed, tbls in self.tbls.items()]
#         self.hash_tbl = f'{self.tbl}_hash'
#         self.hash_temp_tbls = run_batches(self.hash_query_list, tbl=self.hash_tbl, run=False)


#         print('stacking hash batches')
#         self.hash_batch_stack = u.join([f'select * from {tbl}' for tbl in self.hash_temp_tbls])
#         self.hash_batch_stack = f"""
# select
#     *
# from (
#     select
#         *,
#         row_number() over (partition by hash_plan order by plan asc, random_seed asc) as r
#     from (
#         {subquery(self.hash_batch_stack, indents=1)}
#         )
#     )
# where
#     r = 1
# """
# #         load_table(tbl=self.hash_tbl, query=self.hash_batch_stack)



#         print('joining tables in batches')
    
    
    
    
#     query = f"""
# select
#     *
# from (
#     select
#         *
#     from (
#         select
#             A.random_seed,
#             A.plan,
#             A.{self.district_type},
#             {join_str(1).join([f'sum(B.{c}) as {c}' for c in self.cols])}
#         from
#             {tbls['plans']} as A
#         inner join
        
        
        
#         {self.nodes_tbl} as B
#     on
#         A.geoid = B.geoid
#     group by
#         random_seed, plan, {self.district_type}
#     ) as C
# inner join
    
# """
    
    
    
    
    
#         self.join_query_list = [f"""
# select
#     A.*,
#     D.* except (random_seed),
#     C.hash_plan,
#     C.whole_defect as whole_defect_plan,
#     C.intersect_defect as intersect_defect_plan,
#     C.defect as defect_plan,
#     C.pop_deviation as pop_deviation_plan,
#     B.pop_deviation as pop_deviation_district,
#     C.polsby_popper as polsby_popper_plan,
#     B.polsby_popper as polsby_popper_district,
#     B.aland,
#     B.total_pop,
# from (
#     select
#         *
#     from
#         {tbls['plans']}
#     ) as A
# inner join (
#     select
#         *
#     from
#         {tbls['stats']}
#     ) as B
# on
#     A.random_seed = B.random_seed and A.plan = B.plan and A.{self.district_type} = B.{self.district_type}
# inner join (
#     select
#         X.*,
#         Y.hash_plan
#     from (
#         select
#             *
#         from
#             {tbls['summaries']}
#         where
#             pop_deviation < {self.pop_deviation_thresh}
#         ) as X
#     inner join
#         {self.hash_tbl} as Y
#     on
#         X.random_seed = Y.random_seed and X.plan = Y.plan
#     ) as C
# on
#     B.random_seed = C.random_seed and B.plan = C.plan
# inner join (
#     {tbls['params']} as D
# on
#     C.random_seed = D.random_seed
# """ for random_seed, tbls in self.tbls.items()]
#         self.join_tbl = f'{self.tbl}_join'
#         self.join_temp_tbls = run_batches(self.join_query_list, tbl=self.join_tbl, run=False)



#         print('stacking joined table batches')
#         self.join_batch_stack = u.join([f'select * from {tbl}' for tbl in self.join_temp_tbls])
#         self.stack_tbl = f'{self.tbl}_stack'
# #         load_table(tbl=self.stack_tbl, query=self.join_batch_stack)



#         print('joining and aggregating data')
#         self.final_query = f"""
# select
#     A.random_seed,
#     A.plan,
#     A.{self.district_type},
#     max(A.hash_plan) as hash_plan,
#     max(A.pop_deviation_plan) as pop_deviation_plan,
#     max(nodes_plan) as nodes_plan,
#     count(*) as nodes_district,
#     max(A.polsby_popper_plan) as polsby_popper_plan,
#     max(A.polsby_popper_district) as polsby_popper_district,
#     max(A.aland) as aland,
#     max(A.total_pop) as total_pop,
#     case when max(A.aland) > 0 then max(A.total_pop) / max(A.aland) else 0 end as density,
#     {join_str(1).join([f'sum(B.{c}) as {c}' for c in self.cols])}
# from (
#     select
#         *,
#         count(*) over (partition by random_seed, plan) as nodes_plan
#     from
#         {self.stack_tbl}
#     ) as A
# inner join
#     {self.nodes_tbl} as B
# on
#     A.geoid = B.geoid
# group by
#     random_seed, plan, {self.district_type}
# """
#         load_table(tbl=self.tbl, query=self.final_query)

        
        
        
        
        
        
        
        
        
        
        
#     def post_process(self):
#         u = '\nunion all\n'
#         self.tbls = dict()
#         for src_tbl in bqclient.list_tables(self.ds, max_results=self.max_results):
#             full  = src_tbl.full_table_id.replace(':', '.')
#             short = src_tbl.table_id
#             random_seed = short.split('_')[-2]
#             key  = short.split('_')[-1]
#             if random_seed.isnumeric():
#                 try:
#                     self.tbls[random_seed][key] = full
#                 except:
#                     self.tbls[random_seed] = {key : full}

#         self.tbls = {random_seed : tbls for random_seed, tbls in self.tbls.items() if len(tbls)>=3}
#         self.cols = [c for c in get_cols(self.nodes_tbl) if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]
        
#         def run_batches(query_list, batch_size=self.batch_size, tbl=self.tbl, run=True):
#             temp_tbls = list()
#             k = 0
#             while query_list:
#                 query = query_list.pop()
#                 try:
#                     query_stack = query_stack + u + query
#                 except:
#                     query_stack = query
                    
#                 if len(query_list) % batch_size == 0:
#                     temp_tbls.append(f'{tbl}_{k}')
#                     if run:
#                         load_table(tbl=temp_tbls[-1], query=query_stack)
#                     print(f'{len(query_list)} remain')
#                     del query_stack
#                     k += 1
#             return temp_tbls


        
#         print('stacking hashes into batches')
#         self.hash_query_list = [f"""
# select
#     cast(random_seed as int) as random_seed,
#     cast(plan as int) as plan,
#     cast(A.hash as int) as hash_plan
# from
#     {tbls["summaries"]} as A""" for random_seed, tbls in self.tbls.items()]
#         self.hash_tbl = f'{self.tbl}_hash'
#         self.hash_temp_tbls = run_batches(self.hash_query_list, tbl=self.hash_tbl, run=False)


#         print('stacking hash batches')
#         self.hash_batch_stack = u.join([f'select * from {tbl}' for tbl in self.hash_temp_tbls])
#         self.hash_batch_stack = f"""
# select
#     *
# from (
#     select
#         *,
#         row_number() over (partition by hash_plan order by plan asc, random_seed asc) as r
#     from (
#         {subquery(self.hash_batch_stack, indents=1)}
#         )
#     )
# where
#     r = 1
# """
# #         load_table(tbl=self.hash_tbl, query=self.hash_batch_stack)



#         print('joining tables in batches')
#         self.join_query_list = [f"""
# select
#     A.random_seed,
#     A.plan,
#     A.{self.district_type},
#     A.geoid,
#     C.hash_plan,
#     D.* except (random_seed)
#     C.whole_defect_plan,
#     C.intersect_defect_plan,
#     C.defect_plan,
#     C.pop_deviation_plan,
#     B.pop_deviation_district,
#     C.polsby_popper_plan,
#     B.polsby_popper_district,
#     B.aland,
#     B.total_pop,
# from (
#     select
#         cast(random_seed as int) as random_seed,
#         cast(plan as int) as plan,
#         cast({self.district_type} as int) as {self.district_type},
#         geoid,
#     from
#         {tbls['plans']}
#     ) as A
# inner join (
#     select
#         cast(random_seed as int) as random_seed,
#         cast(plan as int) as plan,
#         cast({self.district_type} as int) as {self.district_type},
#         aland,
#         pop_deviation as pop_deviation_district,
#         polsby_popper as polsby_popper_district,
#         total_pop
#     from
#         {tbls['stats']}
#     ) as B
# on
#     A.random_seed = B.random_seed and A.plan = B.plan and A.{self.district_type} = B.{self.district_type}
# inner join (
#     select
#         X.*,
#         Y.hash_plan
#     from (
#         select
#             cast(random_seed as int) as random_seed,
#             cast(plan as int) as plan,
#             pop_deviation as pop_deviation_plan,
#             --pop_imbalance as pop_imbalance_plan,
#             whole_defect as whole_defect_plan,
#             intersect_defect as intersect_defect_plan,
#             defect as defect_plan,
#             polsby_popper as polsby_popper_plan
#         from
#             {tbls['summaries']}
#         where
#             pop_deviation < {self.pop_deviation_thresh}
#             --pop_imbalance < {self.pop_deviation_thresh}
#         ) as X
#     inner join
#         {self.hash_tbl} as Y
#     on
#         X.random_seed = Y.random_seed and X.plan = Y.plan
#     ) as C
# on
#     B.random_seed = C.random_seed and B.plan = C.plan
# inner join (
#     {tbls['params']} as D
# on
#     C.random_seed = D.random_seed
# """ for random_seed, tbls in self.tbls.items()]
#         self.join_tbl = f'{self.tbl}_join'
#         self.join_temp_tbls = run_batches(self.join_query_list, tbl=self.join_tbl, run=False)



#         print('stacking joined table batches')
#         self.join_batch_stack = u.join([f'select * from {tbl}' for tbl in self.join_temp_tbls])
#         self.stack_tbl = f'{self.tbl}_stack'
# #         load_table(tbl=self.stack_tbl, query=self.join_batch_stack)



#         print('joining and aggregating data')
#         self.final_query = f"""
# select
#     A.random_seed,
#     A.plan,
#     A.{self.district_type},
#     max(A.hash_plan) as hash_plan,
#     max(A.pop_deviation_plan) as pop_deviation_plan,
#     max(nodes_plan) as nodes_plan,
#     count(*) as nodes_district,
#     max(A.polsby_popper_plan) as polsby_popper_plan,
#     max(A.polsby_popper_district) as polsby_popper_district,
#     max(A.aland) as aland,
#     max(A.total_pop) as total_pop,
#     case when max(A.aland) > 0 then max(A.total_pop) / max(A.aland) else 0 end as density,
#     {join_str(1).join([f'sum(B.{c}) as {c}' for c in self.cols])}
# from (
#     select
#         *,
#         count(*) over (partition by random_seed, plan) as nodes_plan
#     from
#         {self.stack_tbl}
#     ) as A
# inner join
#     {self.nodes_tbl} as B
# on
#     A.geoid = B.geoid
# group by
#     random_seed, plan, {self.district_type}
# """
#         load_table(tbl=self.tbl, query=self.final_query)
