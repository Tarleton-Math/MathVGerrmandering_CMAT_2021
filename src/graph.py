
@dataclasses.dataclass
class Graph(Variable):
    
    
    
    
    
    # These are default values that can be overridden when you create the object
    name              : str = 'graph'
    g                 : typing.Any = None

    def __post_init__(self):
        self.yr = self.census_yr
        self.g = self
        
        super().__post_init__()
        
        
    def get(self):

        self.nodes         = Nodes(g=self)

        self.tbl = self.nodes.tbl.replace('nodes', 'graph')
        self.gpickle = self.tbl_to_file().with_suffix('.gpickle')
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
                to_gcs(self.gpickle)
        return self
    
    
    def edges_to_graph(self, edges, edge_attrs=None):
        return nx.from_pandas_edgelist(edges, source=f'geoid_x', target=f'geoid_y', edge_attr=edge_attrs)


    def process(self):
        rpt(f'getting edges')
        query = f"""
select
    *
from (
    select
        x.geoid as geoid_x,
        y.geoid as geoid_y,        
        st_distance(x.point, y.point) / {meters_per_mile} as distance,
        st_length(st_intersection(x.polygon, y.polygon)) / {meters_per_mile} as shared_perim
    from
        {self.nodes.tbl} as x,
        {self.nodes.tbl} as y
    where
        x.geoid < y.geoid
        and st_intersects(x.polygon, y.polygon)
    )
where
    shared_perim > 0.01
order by
    geoid_x, geoid_y
"""
        self.edges = run_query(query)
        self.graph = self.edges_to_graph(self.edges, edge_attrs=('distance', 'shared_perim'))
        self.nodes.df = read_table(self.nodes.tbl, cols=list(self.node_attrs) + [self.district_type, 'geoid']).set_index('geoid')
        nx.set_node_attributes(self.graph, self.nodes.df.to_dict('index'))

        print(f'connecting districts')
        rng = np.random.default_rng(0)
        for D in np.unique(self.nodes.df[self.district_type]):
            while True:
                H = nx.subgraph_view(self.graph, filter_node=lambda n: self.graph.nodes[n][self.district_type] == D)
                comp = get_components(H)
                rpt(f"District {self.district_type} {str(D).rjust(3,' ')} component sizes = {[len(c) for c in comp]}")
                if len(comp) == 1:
                    print('connected')
                    break
                else:
                    rpt('adding edges')
                    for c in comp[1:]:
                        for x in c:
                            y = rng.choice(self.graph[x])
                            d = self.graph.nodes[y][self.district_type]
                            self.graph.nodes[x][self.district_type] = d
                            self.nodes.df.loc[x, self.district_type] = d        
        
#         for D, N in self.nodes.df.groupby(self.district_type):
#             while True:
#                 H = self.graph.subgraph(N.index)
#                 comp = get_components(H)
#                 rpt(f"District {self.district_type} {str(D).rjust(3,' ')} component sizes = {[len(c) for c in comp]}")

#                 if len(comp) == 1:
#                     print('connected')
#                     break
#                 else:
#                     rpt('adding edges')
#                     for c in comp[2:]:
#                         for x in c:
#                             y = rng.choice(self.graph[x])
#                             print(self.graph.nodes[x][self.district_type])
#                             self.node
# #                             self.graph.nodes[x][self.district_type] = self.graph.nodes[y][self.district_type]
#                             print(self.graph.nodes[x][self.district_type])
#                             assert 1==2
                    
                    
                    
#                     C = ["', '".join(c) for c in comp[:2]]
#                     query = f"""
# select
#     geoid_x,
#     geoid_y,
#     distance,
#     0.0 as shared_perim
# from (
#     select
#         *,
#         min(distance) over () as min_distance
#     from (
#         select
#             x.geoid as geoid_x,
#             y.geoid as geoid_y,
#             st_distance(x.point, y.point) / {meters_per_mile} as distance
#         from
#             {self.nodes.tbl} as x,
#             {self.nodes.tbl} as y
#         where
#             x.geoid < y.geoid
#             and x.geoid in ('{C[0]}')
#             and y.geoid in ('{C[1]}')
#         )
#     )
# where distance < 1.05 * min_distance
# """
#                     new_edges = run_query(query)
#                     self.graph.update(self.edges_to_graph(new_edges))
                print('done')