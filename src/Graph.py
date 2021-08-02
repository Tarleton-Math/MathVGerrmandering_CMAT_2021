@dataclasses.dataclass
class Graph(Variable):
    name: str = 'graph'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        self.district = self.g.district
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

                
    def process(self):
        self.graph = self.edges_to_graph(self.g.edges.df)
        nx.set_node_attributes(self.graph, self.g.nodes.df.to_dict('index'))