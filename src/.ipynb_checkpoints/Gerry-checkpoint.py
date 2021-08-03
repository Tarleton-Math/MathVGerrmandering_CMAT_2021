@dataclasses.dataclass
class Gerry(Base):
    # These are default values that can be overridden when you create the object
    abbr              : str = 'TX'
    level             : str = 'tract'
    shapes_yr         : int = 2020
    census_yr         : int = 2010
    district_type     : str = 'cd'
    agg_shapes        : bool = True
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
        check_district_type(self.district_type)
        check_year(self.census_yr)
        check_year(self.shapes_yr)
        self.state = states[states['abbr']==self.abbr].iloc[0]
        self.__dict__.update(self.state)

    def get_data(self):
        self.tbl = f'{bq_dataset}.{user_name}_plans_{self.state.abbr}_{self.census_yr}_{self.level}_{self.district_type}'
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
        self.districts   = Districts(g=self)
        self.graph       = Graph(g=self)

    def MCMC(self, steps=10):
        self.get_data()

        d = len(str(steps))
        f = lambda k: f"plan_{str(k).ljust(d, '0')}"
        g = lambda k: self.nodes.df[self.districts.name].copy().astype(str).rename(f(k))
        self.plans = [g(0)]
        self.hashes = [self.districts.hash]
        for step in range(1,steps+1):
            print(f'MCMC {f(step)}', end=concat_str)
            while True:
                if self.graph.recomb():
                    self.districts.update()
                    if self.districts.hash not in self.hashes:
                        self.hashes.append(self.districts.hash)
                        self.plans.append(g(step))
                        print('success')
                        break
                    else:
                        print(f'Found a duplicate plan at {f(step)} - discarding and trying again; hash = {self.districts.hash}')
                else:
                    print(f'No suitable recomb found at {f(step)} - trying again')
                
        print('MCMC done')
        self.plans = pd.concat(self.plans, axis=1)
        load_table(tbl=self.tbl, df=self.plans.reset_index(), preview_rows=0)

        for col in self.plans.columns:
            out_tbl = self.tbl + f"_{col.split('_')[-1]}"
            print(f'Aggregating {self.combined.raw} by {col} on table {self.tbl}')
            self.combined.agg(agg_tbl=self.tbl, agg_col=col, out_tbl=out_tbl, agg_shapes=self.agg_shapes)
            print('done')