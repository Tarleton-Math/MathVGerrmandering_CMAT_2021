@dataclasses.dataclass
class Graph(Variable):
    name: str = 'graph'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        self.district = self.g.district
        self.nodes = self.g.nodes.df
        self.edges = self.g.edges.df
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
        self.graph = self.edges_to_graph(self.edges)
        nx.set_node_attributes(self.graph, self.nodes.to_dict('index'))
        
        print(f'connecting districts', end=concat_str+'\n')
        for D, N in self.nodes.groupby(self.district):
            while True:
                H = self.graph.subgraph(N.index)
                comp = get_components(H)
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