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
    postprocess_thresh    : float = 10.0


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
        
        grp = self.nodes_df.groupby(self.district_type)
        self.districts = {k:set(v) for k,v in grp.groups.items()}
        grp = self.nodes_df.groupby('county')
        self.counties  = {k:set(v) for k,v in grp.groups.items()}

        self.total_pop  = self.nodes_df['total_pop'].sum()
        self.target_pop = self.total_pop / Seats[self.district_type]
        self.step = 0
        self.polsby_popper = 0
        
        self.get_graph()
        self.get_defect()
        self.get_stats()

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

#         print(f'connecting districts')
        connected = False
        while not connected:
            connected = True
            for d in self.districts.keys():
                comp = self.get_components_district(d)
#                 rpt(f"District {self.district_type} {str(D).rjust(3,' ')} component sizes = {[len(c) for c in comp]}")
                if len(comp) > 1:
#                     print('regrouping to connect components')
                    connected = False
                    for c in comp[1:]:
                        for x in c:
                            y = self.rng.choice(list(self.graph.neighbors(x)))
                            self.graph.nodes[x][self.district_type] = self.graph.nodes[y][self.district_type]

        new_districts = Seats[self.district_type] - len(self.districts)
        new_district_starts = self.nodes_df.nlargest(10 * new_districts, 'total_pop').index.tolist()
        del self.nodes_df

        d_new = max(self.districts.keys()) + 1
        while new_districts > 0:
            n = new_district_starts.pop(0)
#             rpt(f'try to start district {M} at node {n}')
            d_old = self.graph.nodes[n][self.district_type]
            self.graph.nodes[n][self.district_type] = d_new
            comp = self.get_components_district(d_old)
            if len(comp) == 1:
#                 rpt('success')
                self.districts[d_new] = set([n])
                self.districts[d_old].remove(n)
                d_new += 1
                new_districts -= 1
            else:
#                 rpt('fail')
                self.graph.nodes[n][self.district_type] = d_old
    
        self.adj = nx.Graph()
        for n, data in self.graph.nodes(data=True):
            d = data[self.district_type]
            self.adj.add_node(d)
            self.adj.nodes[d]['polsby_popper'] = 0
            for k in ['total_pop', 'aland', 'perim']:
                try:
                    self.adj.nodes[d][k] += data[k]
                except:
                    self.adj.nodes[d][k] = data[k]
            
            c = data['county']
            self.adj.add_node(c)
            for k in ['total_pop', self.seats_col]:
                try:
                    self.adj.nodes[c][k] += data[k]
                except:
                    self.adj.nodes[c][k] = data[k]
            
            self.adj.add_edge(c, d)
        
        for c in self.counties:
            self.adj.nodes[c]['whole_target']     = int(np.floor(self.adj.nodes[c][self.seats_col]))
            self.adj.nodes[c]['intersect_target'] = int(np.ceil (self.adj.nodes[c][self.seats_col]))


    def graph_to_df(self, G=None):
        if G is None:
            G = self.adj
        df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
        df.insert(0, 'step', int(self.step))
        df.insert(0, 'random_seed', int(self.random_seed))
        return df

    def get_components(self, H):
        return sorted([tuple(x) for x in nx.connected_components(H)], key=lambda x:len(x), reverse=True)

    def district_view(self, district):
        return nx.subgraph_view(self.graph, lambda n: self.graph.nodes[n][self.district_type] == district)

    def get_components_district(self, district):
        return self.get_components(self.district_view(district))

    def get_hash(self):
        self.partition = tuple(sorted(tuple(sorted(v)) for v in self.districts.values()))
        self.hash = self.partition.__hash__()
        return self.hash

    def get_stats(self, return_df=False, districts=None):
        if districts is None:
            districts = self.districts.keys()
        D = self.districts.keys() - districts
        if len(D) > 0:
            dev_max = max(x for n, x in self.adj.nodes(data='pop_deviation') if n in D)
            dev_min = min(x for n, x in self.adj.nodes(data='pop_deviation') if n in D)
        else:
            dev_max = 0
            dev_min = 10000
            
        attrs = ['total_pop', 'aland', 'perim']
        for d in districts:
            for a in attrs:
                self.adj.nodes[d][a] = 0
            self.adj.nodes[d]['internal_perim'] = 0
        
        for n, data_node in self.graph.nodes(data=True):
            d = data_node[self.district_type]
            if d in districts:
                for a in attrs:
                    self.adj.nodes[d][a] += data_node[a]

        for u, v, data_edge in self.graph.edges(data=True):
            d = self.graph.nodes[u][self.district_type]
            if d in districts:
                if self.graph.nodes[v][self.district_type] == d:
                    self.adj.nodes[d]['internal_perim'] += 2 * data_edge['shared_perim']

        for d in districts:
            stats = self.adj.nodes[d]
            self.polsby_popper -= stats['polsby_popper']
            stats['external_perim'] = stats['perim'] - stats['internal_perim']
            stats['polsby_popper'] = 4 * np.pi * stats['aland'] / (stats['external_perim']**2) * 100
            self.polsby_popper += stats['polsby_popper']
            stats['pop_deviation'] = (stats['total_pop'] - self.target_pop) / self.target_pop * 100
            dev_max = max(dev_max, stats['pop_deviation'])
            dev_min = min(dev_min, stats['pop_deviation'])
        self.pop_deviation = abs(dev_max) + abs(dev_min)


    def get_stats_df(self):
        self.get_stats()
        H = self.adj.subgraph(self.districts.keys())
        self.stats_df = self.graph_to_df(H)
        return self.stats_df


    def get_defect(self):
        self.intersect_defect = 0
        self.whole_defect = 0
        self.defect = 0
        for c in self.counties:
            w = sum(self.adj.degree[d] == 1 for d in self.adj[c])
            dw = abs(self.adj.nodes[c]['whole_target'] - w)
            i = self.adj.degree[c]
            di = abs(self.adj.nodes[c]['intersect_target'] - i)
            self.whole_defect += dw
            self.intersect_defect += di
            self.defect += (dw + di)
            
            self.adj.nodes[c]['whole'] = w
            self.adj.nodes[c]['intersect'] = i            
            self.adj.nodes[c]['whole_defect'] = dw
            self.adj.nodes[c]['intersect_defect'] = di
            self.adj.nodes[c]['defect'] = (dw + di)


    def get_defect_df(self):
        self.get_defect()
        H = self.adj.subgraph(self.counties)
        self.defect_df = self.graph_to_df(H)
        return self.defect_df
    

    def get_plan_df(self):
        P = {n:{'random_seed':self.random_seed, 'step':self.step, self.district_type:d} for n, d in self.graph.nodes(data=self.district_type)}
        df = pd.DataFrame.from_dict(P, orient='index')
        return df


    def get_summaries_df(self):
        self.get_stats()
        self.get_defect()
        self.summaries = pd.DataFrame()
        self.summaries['random_seed'] = [self.random_seed]
        self.summaries['step'] = [self.step]
        self.summaries['hash'] = [self.get_hash()]
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
        self.plans_rec     = [self.get_plan_df()]
        self.defect_rec    = [self.get_defect_df()]
        self.stats_rec     = [self.get_stats_df()]
        self.summaries_rec = [self.get_summaries_df()]
        self.hash_rec      = [self.get_hash()]
        for k in range(1, self.max_steps+1):
            self.step = k
            msg = f"random_seed {self.random_seed} step {self.step} pop_deviation={self.pop_deviation:.1f}"

            if self.recomb():
                self.plans_rec    .append(self.get_plan_df())
                self.defect_rec   .append(self.get_defect_df())
                self.stats_rec    .append(self.get_stats_df())
                self.summaries_rec.append(self.get_summaries_df())
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
            return df[idx + [c for c in df.columns if c not in idx]].rename(columns={'step':'plan'})

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
        self.graph_backup = self.graph.copy()
        self.adj_backup   = self.adj.copy()
        self.districts_backup = self.districts.copy()
        def gen(pop_diff):
            while len(pop_diff) > 0:
                pop_diff /= pop_diff.sum()
                a = self.rng.choice(pop_diff.index, p=pop_diff)
                pop_diff.pop(a)
                yield a
                
        Q = self.get_stats_df()['total_pop']
        pop_diff = pd.DataFrame([(x, y, abs(p-q)) for x, p in Q.iteritems() for y, q in Q.iteritems() if x < y]).set_index([0,1]).squeeze()
        pop_diff = pop_diff ** self.pop_diff_exp
        pairs = gen(pop_diff)
        while True:
            try:
                d0, d1 = next(pairs)
            except StopIteration:
                rpt(f'exhausted all district pairs - I think I am stuck')
                return False
            except Exception as e:
                raise Exception(f'unknown error {e}')
            H = nx.subgraph_view(self.graph, lambda n: self.graph.nodes[n][self.district_type] in [d0, d1])
            
            if not nx.is_connected(H):  # if H is not connect, go to next district pair
#                     rpt(f'{d0},{d1} not connected')
                continue
#                 else:
#                     rpt(f'{d0},{d1} connected')
            P  = Q.copy()
            p0 = P.pop(d0)
            p1 = P.pop(d1)
            q  = p0 + p1
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
                            for d, c in zip([d0,d1], comp):
                                self.districts[d] = set(c).copy()
                                self.adj.remove_edges_from([(d, n) for n in self.adj[d]])

                            for n in comp[0]:
                                self.graph.nodes[n][self.district_type] = d0
                                self.adj.add_edge(self.graph.nodes[n]['county'], self.graph.nodes[n][self.district_type])
                            for n in comp[1]:
                                self.graph.nodes[n][self.district_type] = d1
                                self.adj.add_edge(self.graph.nodes[n]['county'], self.graph.nodes[n][self.district_type])

#                             novel = self.get_hash() not in self.hash_rec
#                             if not novel: # if we've already seen that plan before, reject and keep trying for a new one
# #                             rpt(f'duplicate plan {self.hash}')
#                                 reject(comp)
                            return True
                            
                        def reject(comp):
                            T.add_edge(*e)
                            self.districts[d0] = self.districts_backup[d0].copy()
                            self.districts[d1] = self.districts_backup[d1].copy()
                            self.adj = self.adj_backup.copy()
                            for n in comp[0] + comp[1]:
                                self.graph.nodes[n][self.district_type] = self.graph_backup.nodes[n][self.district_type]
                        
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

                    
    def run_batches(self, query_list, batch_size, tbl):
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
                load_table(tbl=temp_tbls[-1], query=query_stack)
                rpt(f'{len(query_list)} remain')
                del query_stack
                k += 1
        return temp_tbls


    def post_process1(self):    
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
    
    
    def post_process2(self):
        u = '\nunion all\n'    
        rpt('stacking hashes into batches')
        self.hash_query_list = [f"""
select
    random_seed,
    plan,
    A.hash as hash_plan,
    pop_deviation as pop_deviation_plan,
    intersect_defect as intersect_defect_plan,
    whole_defect as whole_defect_plan,
    defect as defect_plan,
from
    {tbls["summaries"]} as A
""" for random_seed, tbls in self.tbls.items()]
        self.hash_tbl = f'{self.tbl}_hash'
        self.hash_temp_tbls = self.run_batches(self.hash_query_list, self.batch_size, self.hash_tbl)

        rpt('stacking hash batches')
        self.hash_batch_stack = u.join([f'select * from {tbl}' for tbl in self.hash_temp_tbls])
        self.hash_batch_stack = f"""
select
    * except (r)
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
        load_table(tbl=self.hash_tbl, query=self.hash_batch_stack)
        for tbl in self.hash_temp_tbls:
            delete_table(tbl)
                    
        
    def post_process3(self):
        u = '\nunion all\n'
        self.cols = [c for c in get_cols(self.nodes_tbl) if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]
        for random_seed, tbls in self.tbls.items():
            rpt(f'combining tables for random_seed {random_seed}')
            tbls['combined'] = f"{tbls['stats'][:-5]}combined"
            query = f"""
select
    H.random_seed,
    H.plan,
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
    G.* except (random_seed, plan, {self.district_type})
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
                A.plan
            from
                {self.hash_tbl} as A
            inner join
                {tbls['summaries']} as B
            on
                A.random_seed = B.random_seed and A.plan = B.plan
            where
                B.pop_deviation < {self.postprocess_thresh}
            ) as C
        inner join
            {tbls['plans']} as D
        on
            C.random_seed = D.random_seed and C.plan = D.plan
        ) as E
    inner join
        {self.nodes_tbl} as F
    on
        E.geoid = F.geoid
    group by
        random_seed, plan, {self.district_type}
    ) as G
inner join
    {tbls['stats']} as H
on
    G.random_seed = H.random_seed and G.plan = H.plan and G.{self.district_type} = H.{self.district_type}
inner join
    {tbls['summaries']} as I
on
    H.random_seed = I.random_seed and H.plan = I.plan
inner join
    {tbls['params']} as J
on
    I.random_seed = J.random_seed
"""
            tbls['combined'] = f"{tbls['stats'][:-5]}combined"
#             if not check_table(tbls['combined']):
            load_table(tbl=tbls['combined'], query=query)
            
        rpt('stacking combined tables')
        query = u.join([f'select * from {tbls[combined]} where pop_deviation_plan < {self.postprocess_thresh}' for random_seed, tbls in self.tbls.items()])
        load_table(tbl=self.tbl, query=query)





            
            
            
            
            
            
#             query = f"""
# select
#     A.random_seed,
#     A.plan,
#     A.{self.district_type},
#     A.hash_plan,
#     PARAMS.* except (random_seed),
#     A.whole_defect_plan,
#     A.intersect_defect_plan,
#     A.defect_plan,
#     A.pop_deviation_plan,
#     STATS.pop_deviation as pop_deviation_district,
#     A.polsby_popper_plan,
#     STATS.polsby_popper as polsby_popper_district,
#     STATS.aland,
#     STATS.total_pop,
#     case when STATS.aland > 0 then STATS.total_pop / STATS.aland else 0 end as density,
#     A.* except (random_seed, plan, {self.district_type}, hash_plan, whole_defect_plan, intersect_defect_plan, defect_plan, pop_deviation_plan)
# from (
#     select
#         E.random_seed,
#         E.plan,
#         E.{self.district_type},
        
#         E.* except (geoid),
#         {join_str(2).join([f'sum(F.{c}) as {c}' for c in self.cols])}
#     from (
#         select
#             C.*,
#             D.* except (random_seed, plan)
#         from (
#             select
#                 *
#             from
#                 {self.hash_tbl}
#             where
#                 pop_deviation_plan < {self.postprocess_thresh}
#             ) as C
#         inner join
#             {tbls['plans']} as D
#         on
#             C.random_seed = D.random_seed and C.plan = D.plan
#         ) as E
#     inner join
#         {self.nodes_tbl} as F
#     on
#         E.geoid = F.geoid
#     group by
#         random_seed, plan, {self.district_type}
#     ) as A
# inner join
#     {tbls['stats']} as STATS
# on
#     A.random_seed = STATS.random_seed and A.plan = STATS.plan and A.{self.district_type} = STATS.{self.district_type}
# inner join
#     {tbls['params']} as PARAMS
# on
#     A.random_seed = PARAMS.random_seed
# """
            
            
            
            

            
            
            
#             query = f"""
# select
#     H.random_seed,
#     H.plan,
#     H.{self.district_type},
#     I.hash as hash_plan,
#     J.* except (random_seed),
#     I.whole_defect as whole_defect_plan,
#     I.intersect_defect as intersect_defect_plan,
#     I.defect as defect_plan,
#     I.pop_deviation as pop_deviation_plan,
#     H.pop_deviation as pop_deviation_district,
#     I.polsby_popper as polsby_popper_plan,
#     H.polsby_popper as polsby_popper_district,
#     H.aland,
#     H.total_pop,
#     case when H.aland > 0 then H.total_pop / H.aland else 0 end as density,
#     G.* except (random_seed, plan, {self.district_type})
# from (
#     select
#         E.* except (geoid),
#         {join_str(2).join([f'sum(F.{c}) as {c}' for c in self.cols])}
#     from (
#         select
#             D.*
#         from (
#             select
#                 *
#             from
#                 {self.hash_tbl} as A
#             inner join
#                 {tbls['summaries']} as B
#             on
#                 A.random_seed = B.random_seed and A.plan = B.plan
#             where
#                 B.pop_deviation < {self.postprocess_thresh}
#             ) as C
#         inner join
#             {tbls['plans']} as D
#         on
#             C.random_seed = D.random_seed and C.plan = D.plan
#         ) as E
#     inner join
#         {self.nodes_tbl} as F
#     on
#         E.geoid = F.geoid
#     group by
#         random_seed, plan, {self.district_type}
#     ) as G
# inner join
#     {tbls['stats']} as H
# on
#     G.random_seed = H.random_seed and G.plan = H.plan and G.{self.district_type} = H.{self.district_type}
# --inner join
# --    {tbls['summaries']} as I
# --on
# --    H.random_seed = I.random_seed and H.plan = I.plan
# inner join
#     {tbls['params']} as J
# on
#     I.random_seed = J.random_seed
# """
            
            
            
            
            
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
#         self.hash_temp_tbls = run_batches(self.hash_query_list, tbl=self.hash_tbl, run=True)


#         print('stacking hash batches')
#         self.hash_batch_stack = u.join([f'select * from {tbl}' for tbl in self.hash_temp_tbls])
#         self.hash_batch_stack = f"""
# select
#     * except (r)
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
#         load_table(tbl=self.hash_tbl, query=self.hash_batch_stack)


#         print('joining tables in batches')
#         self.join_query_list = [f"""
# select
#     H.random_seed,
#     H.plan,
#     H.{self.district_type},
#     I.hash as hash_plan,
#     J.* except (random_seed),
#     I.whole_defect as whole_defect_plan,
#     I.intersect_defect as intersect_defect_plan,
#     I.defect as defect_plan,
#     I.pop_deviation as pop_deviation_plan,
#     H.pop_deviation as pop_deviation_district,
#     I.polsby_popper as polsby_popper_plan,
#     H.polsby_popper as polsby_popper_district,
#     H.aland,
#     H.total_pop,
#     case when H.aland > 0 then H.total_pop / H.aland else 0 end as density,
#     G.* except (random_seed, plan, {self.district_type})
# from (
#     select
#         E.* except (geoid),
#         {join_str(2).join([f'sum(F.{c}) as {c}' for c in self.cols])}
#     from (
#         select
#             D.*
#         from (
#             select
#                 A.random_seed,
#                 A.plan
#             from
#                 {self.hash_tbl} as A
#             inner join
#                 {tbls['summaries']} as B
#             on
#                 A.random_seed = B.random_seed and A.plan = B.plan
#             where
#                 B.pop_deviation < {self.pop_deviation_thresh}
#             ) as C
#         inner join
#             {tbls['plans']} as D
#         on
#             C.random_seed = D.random_seed and C.plan = D.plan
#         ) as E
#     inner join
#         {self.nodes_tbl} as F
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
# inner join
#     {tbls['params']} as J
# on
#     I.random_seed = J.random_seed
# """ for random_seed, tbls in self.tbls.items()]
#         self.join_tbl = f'{self.tbl}_join'
#         self.join_temp_tbls = run_batches(self.join_query_list, tbl=self.join_tbl, run=True)

#         print('stacking joined table batches')
#         self.join_batch_stack = u.join([f'select * from {tbl}' for tbl in self.join_temp_tbls])
#         self.stack_tbl = f'{self.tbl}_stack'
#         load_table(tbl=self.tbl, query=self.join_batch_stack)
#         for tbl in self.hash_temp_tbls:
#             delete_table(tbl)
#         for tbl in self.join_temp_tbls:
#             delete_table(tbl)