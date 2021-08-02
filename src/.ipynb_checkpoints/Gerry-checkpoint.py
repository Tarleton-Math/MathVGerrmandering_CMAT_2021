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

        self.plans = [self.nodes.df[self.district].copy().rename(f'plan_0')]
        for step in range(1,steps+1):
            if self.graph.recomb():
                self.plans.append(self.nodes[self.district].copy().rename(f'plan_{step}'))
        self.plans = pd.concat(self.plans, axis=1)
        load_table(tbl=self.tbl, df=self.plans, preview_rows=0)
        
        
        
    def write_results(self):
        variable, yr, level = 'plans', self.level, self.shapes_yr
        tbl = self.table_id(variable, level, f'{yr}_{self.district}')
        tbl_temp = tbl + '_temp'
        print(f'loading {tbl_temp}', end=concat_str)
        load_table(tbl_temp, df=self.plans.reset_index(), overwrite=True)
        print(f'temp table written{concat_str}joining shapes', end=concat_str)
        query = f"""
select
    A.*,
    B.geography
from
    {tbl_temp} as A
inner join
    {self.table_id('shapes', yr, level)} as B
on
    A.geoid = B.geoid_{yr}
"""
        load_table(tbl, query=query)
        self.write_viz_tables()


    def write_viz_tables(self):
        variable, yr, level = 'plans', self.level, self.shapes_yr
        tbl = self.table_id(variable, level, f'{yr}_{self.district}')
        for step in range(self.steps):
            print(f'Creating viz table for step {step}')
            query = f"""

select
    plan_{step},
    pop_total,
    R,
    D,
    G,
    L,
    votes_total,
    case when perim > 0 then round(4 * acos(-1) * aland / (perim * perim) * 100) else 0 end as polsby_popper,
    aland,
    perim,
    geography
from (
    select
        *,
        st_perimeter(geography) as perim
    from (
        select
            plan_{step},
            sum(aland) as aland,
            round(sum(pop_total)) as pop_total,
            round(sum(R)) as R,
            round(sum(D)) as D,
            round(sum(G)) as G,
            round(sum(L)) as L,
            round(sum(votes_total)) as votes_total,
            st_union_agg(geography) as geography,
        from
            {tbl}
        group by
            1
        )
    )
"""
            load_table(f'{tbl}_{step}', query=query, overwrite=True)