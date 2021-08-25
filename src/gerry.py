from . import *
from .crosswalks import Crosswalks
from .assignments import Assignments
from .shapes import Shapes
from .census import Census
from .elections import Elections
from .votes import Votes
from .combined import Combined
from .nodes import Nodes
from .edges import Edges
from .census import Census
from .graph import Graph
from .districts import Districts

@dataclasses.dataclass
class Gerry(Base):
    # These are default values that can be overridden when you create the object
    abbr              : str = 'TX'
    level             : str = 'tract'
    county_line       : bool = True
    num_steps         : int = 10
    shapes_yr         : int = 2020
    census_yr         : int = 2020
    district_type     : str = 'cd'
    simplification    : int = 600
    num_colors        : int = 8
    refresh_tbl       : typing.Tuple = ()
    refresh_all       : typing.Tuple = ()
    election_filters  : typing.Tuple = (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'")
    pop_imbalance_tol : float = 10.0
    node_attrs        : typing.Tuple = ('total_pop', 'aland', 'perim', 'polsby_popper')
    
    def __post_init__(self):
        check_level(self.level)
        check_district_type(self.district_type)
        check_year(self.census_yr)
        check_year(self.shapes_yr)
        self.state = states[states['abbr']==self.abbr].iloc[0]
        self.__dict__.update(self.state)
        self.tbl = f'{bq_dataset}.{user_name}_plans_{self.state.abbr}_{self.census_yr}_{self.level}_{self.district_type}'
        self.get_data()

    def get_data(self):
        self.crosswalks    = Crosswalks(g=self)
        self.assignments   = Assignments(g=self)
        self.shapes        = Shapes(g=self)
        self.census        = Census(g=self)
        self.elections     = Elections(g=self)
        self.votes_all     = Votes(g=self, group='all')
        self.votes_hl      = Votes(g=self, group='hl')
        self.combined      = Combined(g=self, simplification=0)
        self.nodes         = Nodes(g=self)
        self.edges         = Edges(g=self)
        self.graph         = Graph(g=self)
        self.districts     = Districts(g=self)
        self.steps    = [[k, self.col_name(k), self.tbl+'_'+self.col_name(k)] for k in range(self.num_steps+1)]        

    def col_name(self, k):
        d = len(str(self.num_steps))
        return f"plan_{str(k).rjust(d, '0')}"
        
    def get_components(self, G=None):
        if G is None:
            G = self.graph.graph
        return sorted([tuple(c) for c in nx.connected_components(G)], key=lambda x:len(x), reverse=True)

    def get_colors(self, G=None):
        if G is None:
            G = self.graph.graph
        k = max([d for n, d in G.degree]) + 1
        return pd.Series(nx.equitable_color(G, k)) + 1
        
    def MCMC(self):
        g = lambda k: self.nodes.df[self.districts.name].copy().astype(str).rename(self.col_name(k))
        self.districts.update()
        self.plans   = [g(0)]
        self.hashes  = [self.districts.hash]
        self.stats   = [self.districts.stats.copy()]
        self.summary = [self.districts.summary.copy()]
        print(self.districts.pop_imbalance, self.pop_imbalance_tol)
        for k, col, tbl in self.steps:
            if k == 0:
                continue
            if self.districts.pop_imbalance < self.pop_imbalance_tol:
                break
            rpt(f"MCMC {col}")
            while True:
                if self.graph.recomb():
                    self.districts.update()
                    self.districts.stats  ['plan'] = k
                    self.districts.summary['plan'] = k
                    rpt(self.districts.hash)
                    self.plans.append(g(k))
                    self.hashes.append(self.districts.hash)
                    self.stats.append(self.districts.stats.copy())
                    self.summary.append(self.districts.summary.copy())
                    print('success\n')
                    break
                else:
                    print(f"No suitable recomb found at {col} - trying again")

        print('MCMC done')
        self.plans = pd.concat(self.plans, axis=1)
        load_table(tbl=self.tbl, df=self.plans.reset_index(), preview_rows=0)

        self.stats = pd.concat(self.stats, axis=0).reset_index()
        cols = self.stats.columns.to_list()
        cols[0], cols[1] = cols[1], cols[0]
        load_table(tbl=self.tbl+'_stats', df=self.stats[cols], preview_rows=0)

        self.summary = pd.DataFrame.from_dict(self.summary).set_index('plan')
        load_table(tbl=self.tbl+'_summary', df=self.summary.reset_index(), preview_rows=0)


    def agg_plans(self, start=0, stop=999999, agg_polygon_steps=True):
        if agg_polygon_steps is True:
            agg_polygon_steps = [k for k, col, tbl in self.steps]
        elif agg_polygon_steps is False:
            agg_polygon_steps = []
        agg_polygon_steps = listify(agg_polygon_steps)
        
        for k, col, tbl in self.steps:
            if start <= k and k <= stop:
                rpt(f"Post-processing {col} to make {tbl}")
                self.combined.agg(agg_tbl=self.tbl, agg_col=col, out_tbl=tbl, agg_district=False, agg_polygon=k in agg_polygon_steps, agg_point=False, clr_tbl=self.districts.tbl, simplification=self.simplification)
                print('done')


    def stack_plans(self):
        L = [f"select {k} as plan, * from {tbl}" for k, col, tbl in self.steps]
        j = "\nunion all\n"
        query = j.join(L) + "\norder by plan, geoid"
        load_table(tbl=self.tbl+'_stacked', query=query, preview_rows=0)
        for k, col, tbl in self.steps:
            delete_table(tbl)