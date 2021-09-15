from . import *
from .crosswalks import Crosswalks
from .assignments import Assignments
from .shapes import Shapes
from .census import Census
from .elections import Elections

@dataclasses.dataclass
class Nodes(Variable):
    name              : str = 'nodes'
    abbr              : str = 'TX'
    shapes_yr         : int = 2020
    census_yr         : int = 2020
    level             : str = 'tract'
    district_type     : str = 'cd'
    contract_thresh   : int = 3
    refresh_tbl       : typing.Tuple = ()
    refresh_all       : typing.Tuple = ()
    election_filters  : typing.Tuple = (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'")
    n                 : typing.Any = None


    def __post_init__(self):
        check_level(self.level)
        check_district_type(self.district_type)
        check_year(self.census_yr)
        check_year(self.shapes_yr)

        self.state = states[states['abbr']==self.abbr].iloc[0]
        self.__dict__.update(self.state)
        self.yr = self.census_yr
        
        self.refresh_all = set(self.refresh_all)
        self.refresh_tbl = set(self.refresh_tbl).union(self.refresh_all)
        if self.name in self.refresh_tbl:
            self.refresh_all.add(self.name)
        self.n = self
        super().__post_init__()


    def get(self):
        s = set(self.refresh_tbl).union(self.refresh_all).difference(('nodes', 'graph'))
        if len(s) > 0:
            self.refresh_all = listify(self.refresh_all) + ['nodes', 'graph']
        self.crosswalks    = Crosswalks(n=self)
        self.assignments   = Assignments(n=self)
        self.shapes        = Shapes(n=self)
        self.census        = Census(n=self)
        self.elections     = Elections(n=self)
        
        self.total_pop     = read_table(self.census.tbl, cols=['total_pop']).sum()[0]
        self.target_pop    = self.total_pop / Seats[self.district_type]
        
        self.tbl += f'_{self.district_type}_contract{self.contract_thresh}'
        self.pq = self.tbl_to_file().with_suffix('.parquet')
        self.seats_col = f'seats_{self.g.district_type}'
        self.cols = {'assignments': Levels + District_types,
                     'shapes'     : ['aland', 'polygon'],
                     'census'     : ['total_pop_prop', 'seats_cd', 'seats_sldu', 'seats_sldl'] + Census_columns['data'],
                     'elections'  : [c for c in get_cols(self.g.elections.tbl) if c not in ['geoid', 'county']]
                    }
        exists = super().get()
        if not exists['tbl']:
            if not exists['raw']:
                rpt(f'creating raw table')
                self.process_raw()
            rpt(f'creating table')
            self.process()
#             self.save_tbl()
        return self


    def process_raw(self):
        A_sels = [f'A.{c}'                     for c in self.cols['assignments']]
        S_sels = [f'S.{c}'                     for c in self.cols['shapes']]
        C_sels = [f'coalesce(C.{c}, 0) as {c}' for c in self.cols['census']]
        E_sels = [f'coalesce(E.{c}, 0) as {c}' for c in self.cols['elections']]
        sels = A_sels + C_sels + E_sels + S_sels 
        query = f"""
select
    A.geoid,
    max(E.county) over (partition by cnty) as county,
    {join_str(1).join(sels)},
from
    {self.g.assignments.tbl} as A
left join
    {self.g.shapes.tbl} as S
on
    A.geoid = S.geoid
left join
    {self.g.census.tbl} as C
on
    A.geoid = C.geoid
left join
    {self.g.elections.tbl} as E
on
    A.geoid = E.geoid
"""
        load_table(self.raw, query=query, preview_rows=0)


    def process(self):
        if self.level in ['tabblock', 'bg', 'tract', 'cnty']:
            query_temp = f"select geoid, cnty, {self.seats_col}, substring({self.g.level}, 3) as level from {self.raw}"
        else:
            query_temp = f"select geoid, cnty, {self.seats_col},           {self.g.level}     as level from {self.raw}"
        
        if self.g.contract_thresh < 1:
            query_temp = f"""
select
    geoid,
    level as geoid_new,
from
    ({query_temp})
"""
    
        elif self.g.contract_thresh == 2010:
            query_temp = f"""
select
    geoid,
    case when ct = 1 then substring(cnty, 3) else level end as geoid_new
from (
    select
        geoid,
        level,
        cnty,
        count(distinct {self.g.district_type}) over (partition by cnty) as ct,
    from
        ({query_temp})
    )
"""
        elif self.g.contract_thresh >= 1:
            query_temp = f"""
select
    geoid,
    case when 10 * seats < {self.g.contract_thresh} then substring(cnty, 3) else level end as geoid_new
from (
    select
        geoid,
        level,
        cnty,
        sum({self.seats_col}) over (partition by cnty) as seats,
    from
        ({query_temp})
    )
"""


            
        sels = [f'cast(round(sum({c})) as int) as {c}' for c in self.cols['census'] + self.cols['elections']]
        query = f"""
select
    *,
    case when perim > 0 then round(4 * {np.pi} * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    case when aland > 0 then total_pop / aland else 0 end as density,
    st_centroid(polygon) as point
from (
    select
        *,
        st_perimeter(polygon) / {meters_per_mile} as perim 
    from (
        select
            geoid_new as geoid,
            max(county_new) as county,
            cast(max(district_new) as int) as {self.g.district_type},
            {join_str(3).join(sels)},
            st_union_agg(polygon) as polygon,
            sum(aland) / {meters_per_mile**2} as aland
        from (
            select
                *,
                case when N_county   = (max(N_county)   over (partition by geoid_new)) then county               else NULL end as county_new,
                case when N_district = (max(N_district) over (partition by geoid_new)) then {self.g.district_type} else NULL end as district_new,
                --case when N_sldu = (max(N_sldu) over (partition by geoid_new)) then sldu else NULL end as sldu,
                --case when N_sldl = (max(N_sldl) over (partition by geoid_new)) then sldl else NULL end as sldl
            from (
                select
                    A.geoid_new,
                    B.*,
                    count(1) over (partition by geoid_new, county)                 as N_county,
                    count(1) over (partition by geoid_new, {self.g.district_type}) as N_district,
--                    count(1) over (partition by geoid_new, sldu)   as N_sldu,
  --                  count(1) over (partition by geoid_new, sldl)   as N_sldl
                from (
                    {subquery(query_temp, 5)}
                    ) as A
                left join
                    {self.raw} as B
                on
                    A.geoid = B.geoid
                )
            )
        group by
            geoid
        )
    )
"""
        load_table(self.tbl, query=query, preview_rows=0)