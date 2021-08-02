@dataclasses.dataclass
class Graph(Variable):
    name: str = 'graph'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        self.district = self.g.district
        self.nodes = self.g.nodes.df
        super().__post_init__()


    def get(self):
        print(f"Get {self.name} {self.state.abbr} {self.yr} {self.level} {self.district}".ljust(32, ' '), end=concat_str)
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

    
    def edges_to_graph(self, edges):
        edge_attr = ['distance', 'shared_perim']
        return nx.from_pandas_edgelist(edges, source=f'geoid_x', target=f'geoid_y', edge_attr=edge_attr)


    def get_components(self, G=None):
        if G is None:
            G = self.graph
        return sorted([tuple(c) for c in nx.connected_components(G)], key=lambda x:len(x), reverse=True)

        
    def process(self):
        try:
            self.edges = self.g.edges.df
        except:
            print(f'get edges first')
            self.g.edges = Edges(g=self.g)
        finally:
            self.edges = self.g.edges.df
            print(f'returning to graph', end=concat_str)
        self.graph = self.edges_to_graph(self.edges)
        nx.set_node_attributes(self.graph, self.nodes[['pop']].to_dict('index'))
        
        print(f'connecting districts', end=concat_str+'\n')
        for D, N in self.nodes.groupby(self.district):
            while True:
                H = self.graph.subgraph(N.index)
                comp = self.get_components(H)
                print(f"District {str(D).rjust(3,' ')} component sizes = {[len(c) for c in comp]}", end=concat_str)
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
where distance < 1.05 * min_distance
"""
                    new_edges = run_query(query)
                    self.graph.update(self.edges_to_graph(new_edges))
                print('done', flush=True)
                
                
    def recomb(self):
        district_pops = self.g.get_district_pops()
        pop_imbalance_current = (district_pops.max() - district_pops.min()) / self.g.pop_ideal * 100
        tol = max(self.g.pop_imbalance_tol, pop_imbalance_current)
        print(f'Current population imbalance = {pop_imbalance_current:.2f}% ... setting population imbalance tolerance = {tol:.2f}%')
        
        best_imbalance = 100
        recom_found = False
        D = district_pops.index
        R = rng.permutation([(a,b) for a in D for b in D if a < b]).tolist()
        for pair in R:
            N = self.nodes.query(f'{self.district} in {pair}').copy()
            H = self.graph.subgraph(N.index)
            if not nx.is_connected(H):
#                 print(f'{district_pair} not connected')
                continue
#             else:
#                 print(f'{pair} connected')
            P = district_pops.copy()
            p0 = P.pop(pair[0])
            p1 = P.pop(pair[1])
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
                        comp = nx.connected_components(T)
                        next(comp)
                        s = sum(T.nodes[n]['pop'] for n in next(comp))
                        t = q - s
                        if t < s:
                            s, t = t, s
                        pop_imbalance = (max(t, P_max) - min(s, P_min)) / self.g.pop_ideal * 100
                        best_imbalance = min(best_imbalance, pop_imbalance)
                        if pop_imbalance > tol:
                            T.add_edge(*e)
                        else:
#                             print(f'found split with pop_imbalance={pop_imbalance.round(1)}')
                            recom_found = True
                            comp_new = self.get_components(T)
                            for n, d in zip(comp_new, pair):
                                N.loc[n, 'new'] = d
                            i = N.groupby([self.district, 'new'])['aland'].sum().idxmax()
                            if i[0] != i[1]:
                                comp_new[0], comp_new[1] = comp_new[1], comp_new[0]
                            for n, d in zip(comp_new, pair):
                                self.nodes.loc[n, self.district] = d
                            break
                if recom_found:
                    break
            if recom_found:
                break
#         assert recom_found, "No suitable recomb step found"
        return recom_found