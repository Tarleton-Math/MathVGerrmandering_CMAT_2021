@dataclasses.dataclass
class Gerry(Base):
    # These are default values that can be overridden when you create the object
    abbr              : str = 'TX'
    level             : str = 'tract'
    shapes_yr         : int = 2020
    census_yr         : int = 2010
    district          : str = 'cd'
    refresh_tbl       : typing.Tuple = ()
    refresh_all       : typing.Tuple = ()
    election_filters  : typing.Tuple = (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'")
    pop_imbalance_tol : float = 10.0
    node_attrs        : typing.Tuple = ('geoid', 'total', 'aland', 'perim', 'polsby_popper')
    
    def __post_init__(self):
        check_level(self.level)
        check_district(self.district)
        check_year(self.census_yr)
        check_year(self.shapes_yr)
        self.state = states[states['abbr']==self.abbr].iloc[0]
        self.__dict__.update(self.state)

    def get_data(self):
        self.tbl = f'{bq_dataset}.plans_{self.state.abbr}_{self.census_yr}_{self.level}_{self.district}'
        self.crosswalks  = Crosswalks(g=self)
        self.assignments = Assignments(g=self)
        self.shapes      = Shapes(g=self)
        self.census      = Census(g=self)
        self.elections   = Elections(g=self)
        self.votes_all   = Votes(g=self, group='all')
        self.votes_hl    = Votes(g=self, group='hl')
        self.combined    = Combined(g=self)
#         self.edges       = Edges(g=self)
        self.nodes       = Nodes(g=self)
        self.graph       = Graph(g=self)
        
        
    def get_district_pops(self):
        return self.nodes.df.groupby(self.district)['pop'].sum()

    def MCMC(self, steps=10):
        self.get_data()
        P = self.get_district_pops()
        self.pop_ideal = P.mean()
        pop_imbalance_current = (P.max() - P.min()) / self.pop_ideal * 100
        self.pop_imbalance_tol = max(self.pop_imbalance_tol, pop_imbalance_current)
        print(f'Current population imbalance = {pop_imbalance_current:.2f}% ... setting population imbalance tolerance = {self.pop_imbalance_tol:.2f}%')

        d = len(str(steps))
        f = lambda k: f"plan_{str(k).ljust(d, '0')}"
        self.plans = [self.nodes.df[self.district].copy().rename(f(0))]
        for step in range(1,steps+1):
            if self.graph.recomb():
                self.plans.append(self.nodes.df[self.district].copy().rename(f(step)))
        self.plans = pd.concat(self.plans, axis=1)
        load_table(tbl=self.tbl, df=self.plans.reset_index(), preview_rows=0)
        
        for col in self.plans.columns:
            out_tbl = self.tbl + f"_{col.split('_')[-1]}"
            print(f'Aggregating {self.combined.tbl} by {col} on table {self.tbl}')
            self.combined.process(agg_tbl=self.tbl, agg_col=col, out_tbl=out_tbl)
            print('done')