from . import *
from .crosswalks import Crosswalks
from .assignments import Assignments
from .shapes import Shapes
from .census import Census
from .elections import Elections
from .nodes import Nodes

@dataclasses.dataclass
class Graph(Variable):
    # These are default values that can be overridden when you create the object
    name              : str = 'graph'
    abbr              : str = 'TX'
    shapes_yr         : int = 2020
    census_yr         : int = 2020
    level             : str = 'tract'
    district_type     : str = 'cd'
    county_line       : bool = True
    node_attrs        : typing.Tuple = ('county', 'total_pop', 'density', 'aland', 'perim', 'polsby_popper')
    refresh_tbl       : typing.Tuple = ()
    refresh_all       : typing.Tuple = ()
    election_filters  : typing.Tuple = (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'")
    g                 : typing.Any = None

    def __post_init__(self):
        check_level(self.level)
        check_district_type(self.district_type)
        check_year(self.census_yr)
        check_year(self.shapes_yr)
        
        self.state = states[states['abbr']==self.abbr].iloc[0]
        self.__dict__.update(self.state)
        self.yr = self.census_yr
        self.g = self
        
        self.refresh_all = set(self.refresh_all)
        self.refresh_tbl = set(self.refresh_tbl).union(self.refresh_all)
        if self.name in self.refresh_tbl:
            self.refresh_all.add(self.name)
        super().__post_init__()
        
    def get(self):
        s = set(self.refresh_tbl).union(self.refresh_all).difference(('nodes', 'graph'))
        if len(s) > 0:
            self.refresh_all = listify(self.refresh_all) + ['nodes', 'graph']
        self.crosswalks    = Crosswalks(g=self)
        self.assignments   = Assignments(g=self)
        self.shapes        = Shapes(g=self)
        self.census        = Census(g=self)
        self.elections     = Elections(g=self)
        self.nodes         = Nodes(g=self)

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
        for D, N in self.nodes.df.groupby(self.district_type):
            while True:
                H = self.graph.subgraph(N.index)
                comp = get_components(H)
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
            st_distance(x.point, y.point) / {meters_per_mile} as distance
        from
            {self.nodes.tbl} as x,
            {self.nodes.tbl} as y
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
                print('done')