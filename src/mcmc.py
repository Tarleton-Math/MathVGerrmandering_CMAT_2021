from . import *

@dataclasses.dataclass
class MCMC(Base):
    max_steps             : int
    gpickle               : str
    save                  : bool = True
    seed                  : int = 0
    new_districts         : int = 0
    anneal                : float = 0.0
    pop_diff_exp          : int = 0
    pop_imbalance_target  : float = 1.0
    pop_imbalance_stop    : bool = True
    report_period         : int = 500
    save_period           : int = 500
    

    def __post_init__(self):
        self.results_stem = self.gpickle.stem[6:]
        self.abbr, self.yr, self.level, self.district_type = self.results_stem.split('_')
        ds = f'{root_bq}.{self.results_stem}'
        try:
            bqclient.create_dataset(ds)
        except:
            pass
        self.results_bq = ds + f'.{self.results_stem}_{self.seed}'
        self.results_path = root_path / f'results/{self.results_stem}/{self.results_stem}_{self.seed}/'
        
        self.rng = np.random.default_rng(int(self.seed))
        self.graph = nx.read_gpickle(self.gpickle)
        nx.set_node_attributes(self.graph, self.seed, 'seed')
        
        if self.new_districts > 0:
            M = int(self.nodes_df()[self.district_type].max())
            for n in self.nodes_df().nlargest(self.new_districts, 'total_pop').index:
                M += 1
                self.graph.nodes[n][self.district_type] = str(M)
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
        grp = self.nodes_df().groupby(self.district_type)
        self.districts = {k:tuple(sorted(v)) for k,v in grp.groups.items()}
        self.partition = tuple(sorted(self.districts.values()))
        self.hash = self.partition.__hash__()

    def sum_nodes(self, G, attr='total_pop'):
        return sum(x for n, x in G.nodes(data=attr))
    
    def get_stats(self):
        self.get_districts()
        self.stat = pd.DataFrame()
        for d, N in self.districts.items():
            H = self.graph.subgraph(N)
            s = dict()
            internal_perim = 2*sum(x for a, b, x in H.edges(data='shared_perim') if x is not None)
            external_perim = self.sum_nodes(H, 'perim') - internal_perim
            s['aland'] = self.sum_nodes(H, 'aland')
            s['polsby_popper'] = 4 * np.pi * s['aland'] / (external_perim**2) * 100
            s['total_pop'] = self.sum_nodes(H, 'total_pop')
            for k, v in s.items():
                self.stat.loc[d, k] = v
        self.stat['total_pop'] = self.stat['total_pop'].astype(int)
        self.stat['plan'] = self.step
        self.stat['seed'] = self.seed
        
        
        self.pop_imbalance = (self.stat['total_pop'].max() - self.stat['total_pop'].min()) / self.pop_ideal * 100
        self.summary = pd.DataFrame()
        self.summary['seed'] = [self.seed]
        self.summary['plan'] = [self.step]
        self.summary['hash'] = [self.hash]
        self.summary['pop_imbalance'] = [self.pop_imbalance]
        self.summary['polsby_popper']  = [self.stat['polsby_popper'].mean()]


    def run_chain(self):
        self.step = 0
        self.overite_tbl = True
        nx.set_node_attributes(self.graph, self.step, 'plan')
        self.get_stats()
        self.plans      = [self.nodes_df()[['seed', 'plan', self.district_type]]]
        self.stats      = [self.stat.copy()]
        self.summaries  = [self.summary.copy()]
        self.hashes     = [self.hash]
        for k in range(1, self.max_steps+1):
            self.step = k
            nx.set_node_attributes(self.graph, self.step, 'plan')
            msg = f"seed {self.seed} step {self.step} pop_imbalance={self.pop_imbalance:.1f}"

            if self.recomb():
                self.plans.append(self.nodes_df()[['seed', 'plan', self.district_type]])
                self.stats.append(self.stat.copy())
                self.summaries.append(self.summary.copy())
                self.hashes.append(self.hash)
#                 print('success')
                if self.step % self.report_period == 0:
                    print(msg)
                if self.step % self.save_period == 0:
                    self.save_results()
                if self.pop_imbalance_stop:
                    if self.pop_imbalance < self.pop_imbalance_target:
#                         rpt(f'pop_imbalance_target {self.pop_imbalance_target} satisfied - stopping')
                        break
            else:
                rpt(msg)
                break
        self.save_results()
#         print('MCMC done')


    def save_results(self):
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.results_path / f'graph.gpickle'
        nx.write_gpickle(self.graph, self.graph_file)
        to_gcs(self.graph_file)
        
        def reorder(df):
            idx = [c for c in ['seed', 'plan'] if c in df.columns]
            return df[idx + [c for c in df.columns if c not in idx]]

        tbls = {nm: self.results_bq+f'_{nm}' for nm in ['plans', 'stats', 'summaries']}
        if len(self.plans) > 0:
            self.plans     = pd.concat(self.plans    , axis=0).rename_axis('geoid').reset_index()
            self.stats     = pd.concat(self.stats    , axis=0).rename_axis(self.district_type).reset_index()
            self.summaries = pd.concat(self.summaries, axis=0)

            for nm, tbl in tbls.items():
                saved = False
                for i in range(1, 60):
                    try:
                        load_table(tbl=tbl, df=reorder(self[nm]), overwrite=self.overite_tbl)
                        self[nm] = list()
                        saved = True
                        break
                    except:
                        time.sleep(1)
                assert saved, f'I tried to write the result of seed {self.seed} {i} times without success - giving up'
            self.overite_tbl = False


    def recomb(self):
        def gen(pop_diff):
            while len(pop_diff) > 0:
                pop_diff /= pop_diff.sum()
                a = self.rng.choice(pop_diff.index, p=pop_diff)
                pop_diff.pop(a)
                yield a
        L = self.stat['total_pop']
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
                        I = self.pop_imbalance - imb
                        if I < 0:
                            if self.anneal < 1e-7:
                                if I < -0.01:
                                    T.add_edge(*e)  #  if pop_balance not achieved, re-insert e
                                    continue
                            elif self.rng.uniform() > np.exp(I / self.anneal):
                                T.add_edge(*e)  #  if pop_balance not achieved, re-insert e
                                continue
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
                        if self.hash in self.hashes: # if we've already seen that plan before, reject and keep trying for a new one
#                             rpt(f'duplicate plan {self.hash}')
                            T.add_edge(*e)
                            # Restore old district labels
                            for n in H.nodes:
                                self.graph.nodes[n][self.district_type] = H.nodes[n][self.district_type]
                            self.get_stats()
                        else:  # if this is a never-before-seen plan, keep it and return happy
#                             rpt(f'recombed {self.district_type} {d0} & {d1} got pop_imbalance={self.pop_imbalance:.2f}%')
                            return True