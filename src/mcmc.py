from . import *

default_user_name = 'cook'
default_random_seed = 1
# u = input(f'user_name (default={default_user_name})')
# r = input(f'random_seed (default={default_random_seed})')

try:
    assert len(u) > 0
    user_name = u
except:
    user_name = default_user_name

try:
    random_seed = int(r)
    rng = np.random.default_rng(random_seed)
except:
    random_seed = default_random_seed
    rng = np.random.default_rng(random_seed)
    
@dataclasses.dataclass
class MCMC(Base):
    gpickle           : str
    district_type     : str
    num_steps         : int = 2
    num_colors        : int = 10
    pop_imbalance_tol : float = 10.0

    def __post_init__(self):
        a = self.gpickle.stem.split('_')
        a[0] = user_name
        b = '_'.join(a)
        self.tbl = f'{bq_dataset}.{b}'
        self.graph = nx.read_gpickle(self.gpickle)
        
        P = pd.Series(dict(self.graph.nodes(data='total_pop'))).nlargest(2).index
        self.graph.nodes[P[0]][self.district_type] = '37'
        self.graph.nodes[P[1]][self.district_type] = '38'

        self.plan = 0
        self.get_colors()
        self.num_districts = len(self.districts)
        self.pop_total = self.sum_nodes(self.graph, 'total_pop')
        self.pop_ideal = self.pop_total / self.num_districts
        
#         self.steps = [[k, self.col_name(k)] for k in range(self.num_steps+1)]
        
    def get_districts(self):
        D = {}
        for n, d in self.graph.nodes(data=self.district_type):
            D.setdefault(d, set()).add(n)
        self.districts = {d: tuple(sorted(D[d])) for d in sorted(D.keys())}
        tup = tuple(sorted(self.districts.values()))
        self.hash  = hash(tup)
    
    def get_colors(self):
        self.get_districts()
        H = nx.Graph()
        for i, N in self.districts.items():
            for j, M in self.districts.items():
                if i < j:
                    if len(nx.node_boundary(self.graph, N, M)) > 0:
                        H.add_edge(i, j)
        k = max([d for n, d in H.degree]) + 1
        self.colors = nx.equitable_color(H, k)
#         return pd.Series(nx.equitable_color(H, k)) + 1

    def sum_nodes(self, G, attr='total_pop'):
        return sum(x for n, x in G.nodes(data=attr))
    
    def get_stats(self):
        self.get_districts()
        self.stat = pd.DataFrame()
        for d, N in self.districts.items():
            H = self.graph.subgraph(N)
            s = dict()
            s['total_pop'] = self.sum_nodes(H, 'total_pop')
            s['aland'] = self.sum_nodes(H, 'aland')
            s['perim'] = self.sum_nodes(H, 'perim') - 2*sum(x for a, b, x in H.edges(data='shared_perim'))
            s['polsby_popper'] = 4 * np.pi * s['aland'] / (s['perim']**2) * 100
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
        self.plans     = [pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient='index')[['plan', 'cd']]]
        self.stats     = [self.stat.copy()]
        self.summaries = [self.summary.copy()]
        self.hashes    = [self.hash]
        for k in range(1, self.num_steps+1):
            rpt(f"MCMC {k}")
            self.plan += 1
            nx.set_node_attributes(self.graph, self.plan, 'plan')
            while True:
                if self.recomb():
                    self.get_stats()
                    self.plans.append(pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient='index')[['plan', 'cd']])
                    self.stats.append(self.stat.copy())
                    self.summaries.append(self.summary.copy())
                    self.hashes.append(self.hash)
                    print('success')
                    break
                else:
                    print(f"No suitable recomb found at {col} - trying again")
        print('MCMC done')

        self.plans = pd.concat(self.plans, axis=0).rename_axis('geoid')
        self.stats = pd.concat(self.stats, axis=0).rename_axis('geoid')
        self.summaries = pd.concat(self.summaries, axis=0)
        
        load_table(tbl=self.tbl+'_plans'  , df=self.plans.reset_index()  , preview_rows=0)
        load_table(tbl=self.tbl+'_stats'  , df=self.stats.reset_index()  , preview_rows=0)
        load_table(tbl=self.tbl+'_summary', df=self.summaries, preview_rows=0)
        
    def recomb(self):
        P = self.stat['total_pop'].copy().sort_values()
        L = P.index
        if self.pop_imbalance < self.pop_imbalance_tol:
            tol = districts.pop_imbalance_tol
            pairs = rng.permutation([(a, b) for a in L for b in L if a<b])
        else:
            print(f'pushing', end=concat_str)
            tol = self.pop_imbalance + 0.01
            k = int(len(L) / 2)
            pairs = [(d0, d1) for d0 in L[:k] for d1 in L[k:][::-1]]
        print(f'pop_imbalance={self.pop_imbalance:.2f}{concat_str}setting tol={tol:.2f}%', end=concat_str)
        
        recom_found = False
        for d0, d1 in pairs:
            m = list(self.districts[d0]+self.districts[d1])  # nodes in d0 or d1
            H = self.graph.subgraph(m).copy()  # subgraph on those nodes
            if not nx.is_connected(H):  # if H is not connect, go to next district pair
#                     print(f'{d0},{d1} not connected', end=concat_str)
                continue
#                 else:
#                     print(f'{d0},{d1} connected', end=concat_str)
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
                        pop_imbalance = (max(t, P_max) - min(s, P_min)) / self.pop_ideal * 100  # compute new pop imbalance
                        if pop_imbalance > tol:
                            T.add_edge(*e)  #  if pop_balance not achieved, re-insert e
                        else:
                            # We found a good cut edge & made 2 new districts.  They will be label with the values of d0 & d1.
                            # But which one should get d0?  This is surprisingly important so colors "look right" in animations.
                            # Else, colors can get quite "jumpy" and give an impression of chaos and instability
                            # To achieve this, add aland of nodes that have the same od & new district label
                            # and subtract aland of nodes that change district label.  If negative, swap d0 & d1.
                            
                            comp = get_components(T)
                            x = H.nodes(data=True)
                            s = (sum((-1)**(x[n][self.district_type]==d0) * x[n]['aland'] for n in comp[0]) +
                                 sum((-1)**(x[n][self.district_type]==d1) * x[n]['aland'] for n in comp[1]))
                            if s < 0:
                                d0, d1 = d1, d0
                                
                            # Update district labels
                            for n in comp[0]:
                                self.graph.nodes[n][self.district_type] = d0
                            for n in comp[1]:
                                self.graph.nodes[n][self.district_type] = d1
                            
                            # update districts
                            self.get_districts()
                            if self.hash in self.hashes: # if we've already seen that plan before, reject and keep trying for a new one
                                print(f'duplicate plan {self.hash}', end=concat_str)
                                T.add_edge(*e)
                                # Restore old district labels
                                for n in H.nodes:
                                    self.graph.nodes[n][self.district_type] = H.nodes[n][self.district_type]
                            else:  # if this is a never-before-seen plan, keep it and return happy
                                print(f'recombed {self.district_type} {d0} & {d1} got pop_imbalance={pop_imbalance:.2f}%', end=concat_str)
                                recom_found = True
                                break
                    if recom_found:
                        break
                if recom_found:
                    break
            if recom_found:
                break
        return recom_found