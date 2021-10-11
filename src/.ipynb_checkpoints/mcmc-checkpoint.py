from . import *

@dataclasses.dataclass
class MCMC(Base):
    gpickle    : str = ''
    random_seed: int = 0
    
    def __post_init__(self):
        self.Sources = ('adjs')
        super().__post_init__()
        self.gpickle = pathlib.Path(self.gpickle)
        s = '_'
        w = self.gpickle.stem.split(s)
        self.stem = f'{root_bq}.{s.join(w[0:3])}.{s.join(w[3:5])}'
        
        self.graph = nx.read_gpickle(self.gpickle)
        self.districts  = sorted({d for x, d in self.graph.nodes(data='district')})
        self.counties   = sorted({d for x, d in self.graph.nodes(data='county')})
        self.total_pop  = sum(d for x, d in self.graph.nodes(data='total_pop'))
        self.target_pop = self.total_pop / len(self.districts)
        self.get_adjs()
        self.get_defect()
        self.get_stats()
        
        
        
    def get_adjs(self):
        src = 'adjs'
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
        
        
        
        
        
        
    def get_defect(self):
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
            
            
            
            
    def get_stats(self):
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