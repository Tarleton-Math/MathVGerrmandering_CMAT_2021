from . import *
@dataclasses.dataclass
class Graph(Variable):
    name: str = 'graph'

    def __post_init__(self):
        self.yr = self.g.census_yr
        self.level = self.g.level
        self.nodes = self.g.nodes.df
        super().__post_init__()


    def get(self):
        self.tbl += f'_{self.district_type}'
        exists = super().get()

        try:
            self.graph
            rpt(f'graph exists')
        except:
            try:
                self.graph = nx.read_gpickle(self.gpickle)
                rpt(f'gpickle exists')
            except:
                rpt(f'creating graph')
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
            rpt(f'get edges first')
            self.g.edges = Edges(g=self.g)
            self.edges = self.g.edges.df
            rpt(f'returning to graph')
        self.graph = self.edges_to_graph(self.edges, edge_attrs=('distance', 'shared_perim'))
        nx.set_node_attributes(self.graph, self.nodes[list(self.g.node_attrs)].to_dict('index'))
        
        rpt(f'connecting districts')
        for D, N in self.nodes.groupby(self.district_type):
            while True:
                H = self.graph.subgraph(N.index)
                comp = self.g.get_components(H)
                rpt(f"District {self.district_type} {str(D).rjust(3,' ')} component sizes = {[len(c) for c in comp]}")

                if len(comp) == 1:
                    print('connected')
                    break
                else:
                    rpt('adding edges')
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
                print('done')
                
                
    def recomb(self):
        # get current districts and their populations
        districts = self.g.districts.update()
        
        
        L = districts.pops.copy().sort_values().index
        if districts.pop_imbalance < districts.pop_imbalance_tol:
            tol = districts.pop_imbalance_tol
            pairs = rng.permutation([(a, b) for a in L for b in L if a<b])
        else:
            rpt(f'pushing')
            tol = districts.pop_imbalance + 0.01
            n = int(len(L) / 2)
            pairs = [(d0, d1) for d0 in L[:n] for d1 in L[n:][::-1]]
        rpt(f'current pop_imbalance={districts.pop_imbalance:.2f}{concat_str}setting tol={tol:.2f}%')
        
        recom_found = False
        for d0, d1 in pairs:
            m = list(districts.dict[d0]+districts.dict[d1])  # nodes in d0 or d1
            N = self.nodes.loc[m]  # make local dataframe with data for those nodes only
            H = self.graph.subgraph(m)  # subgraph on those nodes
            if not nx.is_connected(H):  # if H is not connect, go to next district pair
#                     rpt(f'{d0},{d1} not connected')
                continue
#                 else:
#                     rpt(f'{d0},{d1} connected')
            P = districts.pops.copy()
            p0 = P.pop(d0)
            p1 = P.pop(d1)
            q = p0 + p1
            # q is population of d0 & d1
            # P lists all OTHER district populations
            P_min, P_max = P.min(), P.max()

            trees = []  # track which spanning trees we've tried so we don't repeat failures
            for i in range(100):  # max number of spanning trees to try
                w = {e: rng.uniform() for e in H.edges}  # assign random weight to edges
                nx.set_edge_attributes(H, w, "weight")
                T = nx.minimum_spanning_tree(H)  # find minimum spanning tree - we assiged random weights so this is really a random spanning tress
                h = hash(tuple(sorted(T.edges)))  # hash tree for comparion
                if h not in trees:  # prevents retrying a previously failed treee
                    trees.append(h)
                    # try to make search more efficient by searching for a suitable cut edge among edges with high betweenness-centrality
                    # Since cutting an edge near the perimeter of the tree is veru unlikely to produce population balance,
                    # we focus on edges near the center.  Betweenness-centrality is a good metric for this.
                    B = nx.edge_betweenness_centrality(T)
                    B = sorted(B.items(), key=lambda x:x[1], reverse=True)  # sort edges on betweenness-centrality (largest first)
                    max_tries = int(min(100, 0.15*len(B)))  # number of edge cuts to attempt before giving up on this tree
                    k = 0
                    for e, cent in B[:max_tries]:
                        T.remove_edge(*e)
                        comp = nx.connected_components(T)  # T nows has 2 components
                        next(comp)  # second one tends to be smaller → faster to sum over → skip over the first component
                        s = sum(T.nodes[n]['total_pop'] for n in next(comp))  # sum population in component 2
                        t = q - s  # pop of component 1 (recall q is the combined pop of d0&d1)
                        if s > t:  # ensure s < t
                            s, t = t, s
                        pop_imbalance = (max(t, P_max) - min(s, P_min)) / districts.pop_ideal * 100  # compute new pop imbalance
                        if pop_imbalance > tol:
                            T.add_edge(*e)  #  if pop_balance not achieved, re-insert e
                        else:
                            # We found a good cut edge & made 2 new districts.  They will be label with the values of d0 & d1.
                            # But which one should get d0?  This is surprisingly important so colors "look right" in animations.
                            # Else, colors can get quite "jumpy" and give an impression of chaos and instability
                            # So, we look for the largest area of intersection between a new district and an old district.
                            # The new district in that pair receives that old district's label
                            comp = self.g.get_components(T)
                            N.loc[comp[0], 'new'] = d0  # create column called "new" and fill component 0 rows with d0
                            N.loc[comp[1], 'new'] = d1  # dito for d1
                            i = N.groupby(['new', districts.name])['aland'].sum().idxmax()  # find land area of all 4 intersections between new & old districts.  Then take max.
                            if i[0] != i[1]:  # We want that row to have same old and new label for color continuity - if not, flip
                                d0, d1 = d1, d0

                            # Update district labels in global dataframe self.nodes
                            backup = self.nodes[districts.name].copy()
                            self.nodes.loc[comp[0], districts.name] = d0
                            self.nodes.loc[comp[1], districts.name] = d1

                            # update districts
                            self.g.districts.update()
                            if self.g.districts.hash in self.g.hashes: # if we've already seen that plan before, reject and keep trying for a new one
#                                 rpt(e, h)
                                rpt(f'duplicate plan {self.g.districts.hash}')
                                T.add_edge(*e)
                                self.nodes[districts.name] = backup.copy()
                            else:  # if this is a never-before-seen plan, keep it and return happy
                                rpt(f'recombed {self.g.district_type} {d0} and {d1} with pop_imbalance={pop_imbalance:.2f}%')
                                recom_found = True
                                break
                    if recom_found:
                        break
                if recom_found:
                    break
            if recom_found:
                break
        return recom_found