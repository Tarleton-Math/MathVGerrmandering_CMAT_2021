from . import *

@dataclasses.dataclass
class MCMC(Base):
    gpickle            : str
    district_type      : str
    max_steps          : int
    user_name          : str
    random_seed        : int = 1
    num_colors         : int = 10
    pop_imbalance_tol  : float = 10.0
    pop_imbalance_stop : bool = False
    new_districts      : int = 0

    def __post_init__(self):
        self.random_seed = int(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        
        self.gpickle = pathlib.Path(self.gpickle)
        a = self.gpickle.stem.split('_')
        b = '_'.join(a[1:])
#         label = str(pd.Timestamp.now().round("s")).replace(' ','_').replace('-','_').replace(':','_')
        label = 'seed_' + str(self.random_seed).rjust(4, "0")
        self.tbl = f'{proj_id}.redistricting_results_{self.user_name}.{b}_{label}'
        self.graph = nx.read_gpickle(self.gpickle)
        self.gpickle_out = f'{str(self.gpickle)[:-8]}_{label}.gpickle'
        
        if self.new_districts > 0:
            M = int(self.nodes_df()[self.district_type].max())
            for n in self.nodes_df().nlargest(self.new_districts, 'total_pop').index:
                M += 1
                self.graph.nodes[n][self.district_type] = str(M)
        self.plan = 0
        self.get_districts()
        self.num_districts = len(self.districts)
        self.pop_total = self.sum_nodes(self.graph, 'total_pop')
        self.pop_ideal = self.pop_total / self.num_districts

    def nodes_df(self, G=None):
        if G is None:
            G = self.graph
        return pd.DataFrame.from_dict(G.nodes, orient='index')
        
    def edges_tuple(self, G=None):
        if G is None:
            G = self.graph
        return tuple(sorted(tuple((min(u,v), max(u,v)) for u, v in G.edges)))
    
    def get_districts(self):
        grp = self.nodes_df().groupby('cd')
        self.districts = {k:tuple(sorted(v)) for k,v in grp.groups.items()}
        self.partition = tuple(sorted(self.districts.values())).__hash__()

    def sum_nodes(self, G, attr='total_pop'):
        return sum(x for n, x in G.nodes(data=attr))
    
    def get_stats(self):
        self.get_districts()
        self.stat = pd.DataFrame()
        for d, N in self.districts.items():
            H = self.graph.subgraph(N)
            s = dict()
            s['aland'] = self.sum_nodes(H, 'aland')
            shared_perim = 2*sum(x for a, b, x in H.edges(data='shared_perim') if x is not None)
#             if shared_perim is None:
#                 shared_perim = 0
            s['perim'] = self.sum_nodes(H, 'perim') - shared_perim
            s['polsby_popper'] = 4 * np.pi * s['aland'] / (s['perim']**2) * 100
            s['total_pop'] = self.sum_nodes(H, 'total_pop')
            s['density'] = s['total_pop'] / s['aland']
            for k, v in s.items():
                self.stat.loc[d, k] = v
        self.stat.insert(0, 'plan', self.plan)
        self.stat['total_pop'] = self.stat['total_pop'].astype(int)
        
        self.pop_imbalance = (self.stat['total_pop'].max() - self.stat['total_pop'].min()) / self.pop_ideal * 100
        self.summary = pd.DataFrame()
        self.summary['plan'] = [self.plan]
        self.summary['pop_imbalance'] = [self.pop_imbalance]
        self.summary['polsy_popper']  = [self.stat['polsby_popper'].mean()]


    def run_chain(self):
        nx.set_node_attributes(self.graph, self.plan, 'plan')
        self.get_stats()
        self.plans      = [self.nodes_df()[['plan', self.district_type]]]
        self.stats      = [self.stat.copy()]
        self.summaries  = [self.summary.copy()]
        self.partitions = [self.partition]
        for k in range(1, self.max_steps+1):
#             rpt(f"MCMC {k}")
            self.plan += 1
            nx.set_node_attributes(self.graph, self.plan, 'plan')
            while True:
                if self.recomb():
                    self.plans.append(self.nodes_df()[['plan', self.district_type]])
                    self.stats.append(self.stat.copy())
                    self.summaries.append(self.summary.copy())
                    self.partitions.append(self.partition)
#                     print('success')
                    break
                else:
#                     print(f"No suitable recomb found at {col} - trying again")
                    continue
            if self.pop_imbalance_stop and self.pop_imbalance < self.pop_imbalance_tol:
#                 rpt(f'pop_imbalance_tol {self.pop_imbalance_tol} satisfied - stopping')
                break
#         print('MCMC done')

        self.plans = pd.concat(self.plans, axis=0).rename_axis('geoid')
        self.stats = pd.concat(self.stats, axis=0).rename_axis(self.district_type)
        self.summaries = pd.concat(self.summaries, axis=0)
        
        load_table(tbl=self.tbl+'_plans'  , df=self.plans.reset_index()  , preview_rows=0)
        load_table(tbl=self.tbl+'_stats'  , df=self.stats.reset_index()  , preview_rows=0)
        load_table(tbl=self.tbl+'_summary', df=self.summaries, preview_rows=0)
        nx.write_gpickle(self.graph, self.gpickle_out)
        
        
    def recomb(self):
        self.get_stats()
        L = self.stat['total_pop'].sort_values().index.copy()
        if self.pop_imbalance < self.pop_imbalance_tol:
            tol = self.pop_imbalance_tol
            pairs = self.rng.permutation([(a, b) for a in L for b in L if a<b])
        else:
#             print(f'pushing', end=concat_str)
            tol = self.pop_imbalance + 0.01
            k = int(len(L) / 2)
            pairs = [(d0, d1) for d0 in L[:k] for d1 in L[k:][::-1]]
#         print(f'pop_imbalance={self.pop_imbalance:.2f}{concat_str}setting tol={tol:.2f}%', end=concat_str)
        
        recom_found = False
        for d0, d1 in pairs:
            m = list(self.districts[d0]+self.districts[d1])  # nodes in d0 or d1
            H = self.graph.subgraph(m).copy()  # subgraph on those nodes
            if not nx.is_connected(H):  # if H is not connect, go to next district pair
#                     print(f'{d0},{d1} not connected', end=concat_str)
                continue
#                 else:
#                     print(f'{d0},{d1} connected', end=concat_str)
            P = self.stat['total_pop'].copy()
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
                    max_tries = int(min(300, 0.2*len(B)))  # number of edge cuts to attempt before giving up on this tree
                    k = 0
                    for e, cent in B[:max_tries]:
                        T.remove_edge(*e)
                        comp = nx.connected_components(T)  # T nows has 2 components
                        next(comp)  # second one tends to be smaller → faster to sum over → skip over the first component
                        s = sum(H.nodes[n]['total_pop'] for n in next(comp))  # sum population in component 2
                        t = q - s  # pop of component 0 (recall q is the combined pop of d0&d1)
                        if s > t:  # ensure s < t
                            s, t = t, s
                        imb = (max(t, P_max) - min(s, P_min)) / self.pop_ideal * 100  # compute new pop imbalance
                        if imb > tol:
                            T.add_edge(*e)  #  if pop_balance not achieved, re-insert e
                        else:
                            # We found a good cut edge & made 2 new districts.  They will be label with the values of d0 & d1.
                            # But which one should get d0?  This is surprisingly important so colors "look right" in animations.
                            # Else, colors can get quite "jumpy" and give an impression of chaos and instability
                            # To achieve this, add aland of nodes that have the same od & new district label
                            # and subtract aland of nodes that change district label.  If negative, swap d0 & d1.
                            comp = get_components(T)
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
                            assert abs(self.pop_imbalance - imb) < 1e-2, f'disagreement betwen pop_imbalance calculations {self.pop_imbalance} v {imb}'
                            if self.partition in self.partitions: # if we've already seen that plan before, reject and keep trying for a new one
#                                 print(f'duplicate plan {self.hash}', end=concat_str)
                                T.add_edge(*e)
                                # Restore old district labels
                                for n in H.nodes:
                                    self.graph.nodes[n][self.district_type] = H.nodes[n][self.district_type]
                                self.get_stats()
                            else:  # if this is a never-before-seen plan, keep it and return happy
#                                 print(f'recombed {self.district_type} {d0} & {d1} got pop_imbalance={self.pop_imbalance:.2f}%', end=concat_str)
                                recom_found = True
                                break
                    if recom_found:
                        break
                if recom_found:
                    break
            if recom_found:
                break
        return recom_found