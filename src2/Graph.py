@dataclasses.dataclass
class Graph(Variable):
    name: str = 'graph'

    def __post_init__(self):
        self.yr = self.g.census_yr
        self.level = self.g.level
        self.nodes = self.g.nodes.df
        super().__post_init__()


    def get(self):
        print(f"Get {self.name} {self.state.abbr} {self.yr} {self.level} {self.district_type}".ljust(33, ' '), end=concat_str)
        try:
            self.graph
            print(f'graph exists', end=concat_str)
        except:
            try:
                self.graph = nx.read_gpickle(self.gpickle)
                print(f'gpickle exists', end=concat_str)
            except:
                print(f'creating graph', end=concat_str)
                self.process()
                self.gpickle.parent.mkdir(parents=True, exist_ok=True)
                nx.write_gpickle(self.graph, self.gpickle)
        
        return self
    

    def edges_to_graph(self, edges, edge_attrs=None):
        return nx.from_pandas_edgelist(edges, source=f'geoid_x', target=f'geoid_y', edge_attr=edge_attrs)

        
    def process(self):
        try:
            self.edges = self.g.edges.df
        except:
            print(f'get edges first')
            self.g.edges = Edges(g=self.g)
            self.edges = self.g.edges.df
            print(f'returning to graph', end=concat_str)
        self.graph = self.edges_to_graph(self.edges, edge_attrs=('distance', 'shared_perim'))
        nx.set_node_attributes(self.graph, self.nodes[list(self.g.node_attrs)].to_dict('index'))
        
        print(f'connecting districts', end=concat_str+'\n')
        for D, N in self.nodes.groupby(self.district_type):
            while True:
                H = self.graph.subgraph(N.index)
                comp = self.g.get_components(H)
                print(f"District {self.district_type} {str(D).rjust(3,' ')} component sizes = {[len(c) for c in comp]}", end=concat_str)

                if len(comp) == 1:
                    print('connected')
                    break
                else:
                    print('adding edges', end=concat_str)
                    C = ["', '".join(c) for c in comp[:2]]
                    query = f"""
select
    geoid_x,
    geoid_y,
    distance,
    0.0 as shared_perim
from (
    select
        *,
        min(distance) over () as min_distance
    from (
        select
            x.geoid as geoid_x,
            y.geoid as geoid_y,
            st_distance(x.point, y.point) as distance
        from
            {self.g.combined.tbl} as x,
            {self.g.combined.tbl} as y
        where
            x.geoid < y.geoid
            and x.geoid in ('{C[0]}')
            and y.geoid in ('{C[1]}')
        )
    )
--where distance < 1.05 * min_distance
"""
                    new_edges = run_query(query)
                    self.graph.update(self.edges_to_graph(new_edges))
                print('done', flush=True)
                
                
    def recomb(self):
        districts = self.g.districts.update()
        tol = max(districts.pop_imbalance, districts.pop_imbalance_tol)
        print(f'Current pop imbalance = {districts.pop_imbalance:.2f}%{concat_str}setting tol = {tol:.2f}%', end=concat_str)
        
        best_imbalance = 100
        recom_found = False
#         R = rng.permutation(np.array([(d0, d1, n0, n1) for d0, n0 in districts.tuple for d1, n1 in districts.tuple if d0 < d1], dtype=object))
#         for d0, d1, n0, n1 in R:
        q = districts.pops.sort_values().index
        for d0 in q:
            for d1 in reversed(q):
                m = list(districts.dict[d0]+districts.dict[d1])
                N = self.nodes.loc[m]
                H = self.graph.subgraph(m)
                if not nx.is_connected(H):
    #                 print(f'{d0, d1} not connected')
                    continue
    #             else:
    #                 print(f'{d0, d1} connected')
                P = districts.pops.copy()
                p0 = P.pop(d0)
                p1 = P.pop(d1)
                q = p0 + p1
                P_min, P_max = P.min(), P.max()
                trees = []
                for i in range(100):
                    w = {e: rng.uniform() for e in H.edges}
                    nx.set_edge_attributes(H, w, "weight")
                    T = nx.minimum_spanning_tree(H)
                    h = hash(tuple(sorted(T.edges)))
                    if h not in trees:
                        trees.append(h)
                        B = nx.edge_betweenness_centrality(T)
                        B = sorted(B.items(), key=lambda x:x[1], reverse=True)
                        max_tries = max(100, int(0.02 * len(B)))
                        for e, cent in B[:max_tries]:
                            T.remove_edge(*e)
    #                         comp = nx.connected_components(T)
    #                         print(len(list(comp)))
                            comp = nx.connected_components(T)                
                            next(comp)
                            s = sum(T.nodes[n]['total_pop'] for n in next(comp))
                            t = q - s
                            if t < s:
                                s, t = t, s
                            pop_imbalance = (max(t, P_max) - min(s, P_min)) / districts.pop_ideal * 100
                            best_imbalance = min(best_imbalance, pop_imbalance)
                            if pop_imbalance < tol:
                                print(f'found recomb with {self.g.district_type} {d0} and {d1} split with pop_imbalance={pop_imbalance:.2f}%', end=concat_str)
                                comp = self.g.get_components(T)
                                N.loc[comp[0], 'new'] = d0
                                N.loc[comp[1], 'new'] = d1
                                i = N.groupby(['new', districts.name])['aland'].sum().idxmax()
                                if i[0] != i[1]:
                                    d0, d1 = d1, d0
                                self.nodes.loc[comp[0], districts.name] = d0
                                self.nodes.loc[comp[1], districts.name] = d1

                                self.g.districts.update()
                                if self.g.districts.hash not in self.g.hashes:
                                    recom_found = True
                                    break
                                else:
                                    print(f'Found a duplicate plan {self.g.districts.hash}{concat_str}continuing', end=concat_str)
                            T.add_edge(*e)
                    if recom_found:
                        break
                if recom_found:
                    break
            if recom_found:
                break
        return recom_found