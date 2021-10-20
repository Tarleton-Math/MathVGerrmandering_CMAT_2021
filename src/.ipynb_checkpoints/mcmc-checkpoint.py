from . import *

@dataclasses.dataclass
class MCMC(Base):
    gpickle              : str = ''
    random_seed          : int = 0
    max_steps            : int = 5
    report_period        : int = 1
    save_period          : int = 2
    pop_deviation_target : float = 10.0
    yolo_length          : int = 10
    defect_cap           : int = 0
        
    
    def __post_init__(self):
        self.Sources = ('nodes', 'plan', 'county', 'district', 'summary')
        super().__post_init__()
        self.random_seed = int(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)

        self.gpickle = pathlib.Path(self.gpickle)
        s = '_'
        w = self.gpickle.stem.split(s)
        stem = f'{root_bq}.{s.join(w[0:3])}.{s.join(w[3:5])}'
        self.tbls = {f'{src}_rec': f'{stem}_{self.random_seed}_{src}' for src in self.Sources}#, 'params']}
        
        self.graph = nx.read_gpickle(self.gpickle)
        self.districts  = sorted({d for x, d in self.graph.nodes(data='district')})
        self.counties   = sorted({d for x, d in self.graph.nodes(data='county')})
        self.total_pop  = sum(d for x, d in self.graph.nodes(data='total_pop'))
        self.target_pop = self.total_pop / len(self.districts)
        self.get_adj()
        self.plan = 0
        self.update()
        if self.defect_cap == 0:
            self.defect_cap = self.defect
            
            
    def post_process(self):
        # WORKING HERE
        cols = get_cols(self.tbls['nodes'])
        a = cols.index('seats')
        b = cols.index('polygon')
        self.data_cols = cols[a:b]

        
        query = list()
        query.append(f"""
select
    A.random_seed,
    A.plan,
    A.district,
    B.county,
    B.cntyvtd,
    B.* except (geoid, district)
from
    {self.tbls['plan']} as A
inner join
    {self.tbls['nodes']} as B
on
    A.geoid = B.geoid
""")
                     
        
        
        
    def save_results(self):
        self.report()
        print('saving')
        load_table(tbl=tbl, df=pd.concat(self[src], axis=0), overwrite=self.overwrite_tbl)
        for src, tbl in self.tbls.items():
            saved = False
            for i in range(1, 60):
                try:
                    load_table(tbl=tbl, df=pd.concat(self[src], axis=0), overwrite=self.overwrite_tbl)
                    self[src] = list()
                    saved = True
                    break
                except:
                    time.sleep(1)
            assert saved, f'I tried to write the result of random_seed {self.random_seed} {i} times without success - giving up'
        self.overwrite_tbl = False

    
    def run_chain(self):
        self.plan = 0
        self.update()
        self.overwrite_tbl = True
        self.record()
        self.start_time = time.time()
        while self.plan < self.max_steps:
            self.plan += 1
            msg = f"random_seed {self.random_seed} step {self.plan} pop_deviation={self.pop_deviation:.1f}"
            if self.recomb():
                self.record()
                if self.plan % self.report_period == 0:
                    self.report()
                if self.plan % self.save_period == 0:
                    self.save_results()
            else:
                rpt(msg)
                break
        self.save_results()
        self.report()
        print(f'random_seed {self.random_seed} done')


    def update(self):
        for G in [self.graph, self.adj]:
            for a in ['random_seed', 'plan']:
                nx.set_node_attributes(G, self[a], a)
        self.hash = get_hash(self.graph)
        self.get_county_stats()
        self.get_district_stats()
        self.plan_df = graph_to_df(self.graph, 'geoid', attr=('random_seed', 'plan', 'district'))
        H = self.adj.subgraph(self.counties)
        self.county_df = graph_to_df(H, 'county', attr=('random_seed', 'plan', 'whole_defect', 'intersect_defect', 'defect'))
        H = self.adj.subgraph(self.districts)
        self.district_df =  graph_to_df(H, 'district', attr=('random_seed', 'plan', 'total_pop', 'pop_deviation', 'polsby_popper', 'aland'))
        attr = ['random_seed', 'plan', 'hash', 'polsby_popper', 'pop_deviation', 'intersect_defect', 'whole_defect', 'defect']
        self.summary_df = pd.DataFrame([{a:self[a] for a in attr}])

    
    def record(self):
        self.update()
        for a in ['plan', 'county', 'district', 'summary', 'hash']:
            r = f'{a}_rec'
            if a == 'hash':
                X = self.hash
            else:
                X = self[f'{a}_df'].copy()
            try:
                self[r].append(X)
            except:
                self[r] = [X]


    def report(self):
        print(f'random_seed {self.random_seed}: step {self.plan} {time_formatter(time.time() - self.start_time)}, pop_deviation={self.pop_deviation:.1f}, intersect_defect={self.intersect_defect}, whole_defect={self.whole_defect}, defect={self.defect}', flush=True)


    def recomb(self):
        # Make backups - used to undo rejected steps
        self.update()
        self.graph_backup = self.graph.copy()
        self.adj_backup   = self.adj.copy()

        def accept():
            for dist, comp in zip(districts, components):
                cuts = tuple((dist, cty) for cty in self.adj[dist])
                self.adj.remove_edges_from(cuts)  # cut all edges of self.adj touching d0 or d1
                for n in comp:
                    self.graph.nodes[n]['district'] = dist  # relabel nodes
                    self.adj.add_edge(dist, self.graph.nodes[n]['county'])

        def reject():
            T.add_edge(*e)  # restore e
            self.adj = self.adj_backup.copy()  # restore self.adj
            for n in H.nodes:
                self.graph.nodes[n]['district'] = self.graph_backup.nodes[n]['district']

        # Make generator to yield district pairs in random order weighted by population difference
        # yields pairs with large pop difference first to encourage convergence to population balance.
        # To disable weighting (purely random sample), set pop_diff_exp=0
        # Make dataframe of district pairs with pop_difference raised to pop_diff_exp
        def gen(R):
            while len(R) > 0:
                p = R / R.sum()
                r = self.rng.choice(R.index, p=p)  # yield from remaining pairs with prefence for larger population difference
                R.pop(r)
                yield r
        push_deviation = self.pop_deviation > self.pop_deviation_target
        pop_diff_exp = 2 * push_deviation
        P = self.district_df[['district', 'total_pop']].values
        Q = pd.DataFrame([(x, y, abs(p-q)) for x, p in P for y, q in P if x < y]).set_index([0,1]).squeeze()
        R = (Q / Q.sum()) ** pop_diff_exp
        pairs = gen(R)
        
        while True:
            try:
                districts = next(pairs)
            except StopIteration:
                rpt(f'exhausted all district pairs - I think I am stuck')
                return False

            H = district_view(self.graph, districts)
            if not nx.is_connected(H):  # if H not connected, go to next district pair
                continue

            P = self.district_df.set_index('district')['total_pop']
            q = P.pop(districts[0]) + P.pop(districts[1])
            p_min, p_max = P.min(), P.max()
            # q is population of d0 & d1
            # P lists all OTHER district populations
            # So P_min & P_max are the min & max population of all districts except d0 & d1

            trees = []  # track which spanning trees we've tried so we don't repeat failures
            Edges = get_edges(H)
            for i in range(100):  # max number of spanning trees to try before going to next district pair
                # We want a random spanning tree, but networkx can only deterministically find a MINIMUM spanning tree (Kruskal's algorithm).
                # So, we first assign random weights to the edges then find MINIMUM spanning tree based on those random weights
                # thus producing a random spanning tree.
                for e in Edges:
                    H.edges[e]['weight'] = self.rng.uniform()
                T = nx.minimum_spanning_tree(H)
                h = get_edges(T).__hash__()   # store T's hash so we avoid trying it again later if it fails
                if h not in trees:  # prevents retrying a previously failed treee
                    trees.append(h)
                    # Make search for good edge cut with population balance more efficient by looking at edges with high betweenness-centrality.
                    # Since cutting an edge near the leaves of the tree is very unlikely to produce population balance,
                    # we focus on edges near the center.  Betweenness-centrality is a good metric for this.
                    B = nx.edge_betweenness_centrality(T)
                    B = sorted(B.items(), key=lambda x:x[1], reverse=True)  # sort edges on betweenness-centrality (largest first)
                    for e, bw_centrality in B:
                        if bw_centrality < 0.1: # We exhausted all good edges - move on to next tree
                            break
                        T.remove_edge(*e)
                        components = get_components(T, sort=False)  # T nows has 2 components
                        next(components)  # second component tends to be smaller → faster to sum over
                        s = sum(H.nodes[n]['total_pop'] for n in next(components))  # sum population in second component 
                        t = q - s  # population in first component (recall q is the combined population of d0 & d1)
                        if s > t:  # ensure s < t
                            s, t = t, s
                        pop_deviation_min = self.target_pop - min(s, p_min)
                        pop_deviation_max = max(t, p_max) - self.target_pop
                        pop_deviation_new = (pop_deviation_min + pop_deviation_max) / self.target_pop * 100  # new pop deviation
                        
                        # Phase 1: If pop_deviation too high, reject steps that increase it
                        if push_deviation:
                            if pop_deviation_new > self.pop_deviation:
                                T.add_edge(*e)
                                continue
                        # Phase 2: If pop_deviation within target range, reject steps that would leave target range
                        else:
                            if pop_deviation_new > self.pop_deviation_target:
                                T.add_edge(*e)
                                continue

                        components = get_components(T)
                        accept()
                         # if we've seen that plan recently, reject and try again
                        h = get_hash(self.graph)
                        if h in self.hash_rec[-self.yolo_length:]:
                            reject()
                            continue

#                         if defect exceeds cap, reject and try again
                        old_defect = self.defect
                        self.get_county_stats()
                        if self.defect > self.defect_cap and self.defect > old_defect:
                            reject()
                            self.get_county_stats()
                            continue

                        self.update()
                        assert abs(self.pop_deviation - pop_deviation_new) < 1e-2, f'disagreement betwen pop_deviation calculations {self.pop_deviation} v {pop_deviation_new}'

                        # We found a good cut edge & made 2 new districts.  They will be label with the values of d0 & d1.
                        # But which one should get d0?  This is surprisingly important so colors "look right" in animations.
                        # Else, colors can get quite "jumpy" and give an impression of chaos and instability
                        # To achieve this, add aland of nodes that have the same od & new district label
                        # and subtract aland of nodes that change district label.  If negative, swap d0 & d1.
                        s = 0
                        for dist, comp in zip(districts, components):
                            for n in comp:
                                ds = self.graph.nodes[n]['aland']
                                if self.graph.nodes[n]['district'] == dist:
                                    s += ds
                                else:
                                    s -= ds
                        if s < 0:
                            components[0], components[1] = components[1], components[0]
                            accept()
                            self.update()
                        return True

        
    def get_county_stats(self):
        # The "county-line" rule prefers minimal county & district splitting. We implement as follows:
        # seats_share = county population / distrinct ideal population
        # Ideally, county should wholly contain floor(seats_share) and intersect ceiling(seats_share) districts
        # Ex: County seats_share=2.4, so it should ideally wholly contain 2 districts and intersect a 3rd.
        # whole_defect = |actual wholly contained - floor(seats_share)|
        # intersect_defect = |actual intersected - ceil(seats_share)|
        # defect = whole_defect + intersect_defect
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
            
            
    def get_district_stats(self):
        # compute district stats & store in self.adj
        # initialize to 0
        attrs = ['total_pop', 'aland', 'perim']
        for d in self.districts:
            for a in attrs:
                self.adj.nodes[d][a] = 0
            self.adj.nodes[d]['internal_perim'] = 0
        
        # iterate over nodes in self.graph and increment corresponding district node in self.adj
        for n, data_node in self.graph.nodes(data=True):
            d = data_node['district']
            for a in attrs:
                self.adj.nodes[d][a] += data_node[a]

        # iterate over edges in self.graph and increment corresponding district node in self.adj
        for u, v, data_edge in self.graph.edges(data=True):
            d = self.graph.nodes[u]['district']
            if self.graph.nodes[v]['district'] == d: # if u & v in same district, (u, v) is an internal edge
                self.adj.nodes[d]['internal_perim'] += 2 * data_edge['shared_perim']  # must double because this boundary piece counts in perim for BOTH u & v

        dev_min =  10000
        dev_max = -10000
        self.polsby_popper = 0
        for d in self.districts:
            stats = self.adj.nodes[d]
            # computer external_perim & polsby-popper (aland, perim, & internal perim computed in prior loops)
            stats['external_perim'] = stats['perim'] - stats['internal_perim']
            stats['polsby_popper'] = 4 * np.pi * stats['aland'] / (stats['external_perim']**2) * 100
            self.polsby_popper += stats['polsby_popper']
            
            # compute pop_deivations and update dev_min & dev_max
            stats['pop_deviation'] = (stats['total_pop'] - self.target_pop) / self.target_pop * 100
            dev_min = min(dev_min, stats['pop_deviation'])
            dev_max = max(dev_max, stats['pop_deviation'])
        self.pop_deviation = abs(dev_max) + abs(dev_min)
        self.polsby_popper /= len(self.districts)
        
        
    def get_adj(self):
        src = 'adj'
        # Create the county-district bi-partite adjacency graph.
        # This graph has 1 node for each county and district &
        # an edge for all (county, district) that intersect (share land).
        # It is an efficient tool to track map defect and other properties.
        self.adj = nx.Graph()
        for n, data in self.graph.nodes(data=True):
            D = data['district']
            self.adj.add_node(D)  # adds district node if not already present
            self.adj.nodes[D]['polsby_popper'] = 0
            for k in ['total_pop', 'aland', 'perim']:
                try:
                    self.adj.nodes[D][k] += data[k]  # add to attribute if exists
                except:
                    self.adj.nodes[D][k] = data[k]  # else create attribute

            C = data['county']
            self.adj.add_node(C)  # adds county node if not already present
            for k in ['total_pop', 'seats']:
                try:
                    self.adj.nodes[C][k] += data[k]  # add to attribute if exists
                except:
                    self.adj.nodes[C][k] = data[k]  # else create attribute

            self.adj.add_edge(C, D)  # create edge

        # get defect targets
        for C in self.counties:
            self.adj.nodes[C]['whole_target']     = int(np.floor(self.adj.nodes[C]['seats']))
            self.adj.nodes[C]['intersect_target'] = int(np.ceil (self.adj.nodes[C]['seats']))
