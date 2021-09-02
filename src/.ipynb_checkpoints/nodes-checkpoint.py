from . import *
@dataclasses.dataclass
class Nodes(Variable):
    name: str = 'nodes'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        super().__post_init__()


    def get(self):
        self.tbl += f'_{self.g.district_type}'
        self.cols = {'assignments': Levels + District_types,
                     'shapes'     : ['aland', 'polygon'],
                     'census'     : Census_columns['data'],
                     'elections'  : [c for c in get_cols(self.g.elections.tbl) if c not in ['geoid', 'county']]
                    }
        exists = super().get()
        if not exists['tbl']:
            if not exists['raw']:
                rpt(f'creating raw table')
                self.process_raw()
            rpt(f'creating table')
            self.process()
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
            query_temp = f"select *, substring({self.g.level}, 3) as level_temp from {self.g.assignments.tbl}"
        else:
            query_temp = f"select *, {self.g.level} as level_temp from {self.g.assignments.tbl}"
        
        if not self.g.county_line:
            query_temp = f"""
select
    geoid,
    level_temp as geoid_new,
from
    ({query_temp})
"""
    
        else:
            query_temp = f"""
select
    geoid,
    case when ct > 1 then level_temp else substring(cnty, 3) end as geoid_new,
from (
    select
        geoid,
        level_temp,
        cnty,
        count(distinct {self.g.district_type}) over (partition by cnty) as ct,
    from
        ({query_temp})
    )
"""
            
        sels = [f'sum({c}) as {c}' for c in self.cols['census'] + self.cols['elections']]
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
            max(county)   as county,
            max(district) as {self.g.district_type},
            {join_str(3).join(sels)},
            st_union_agg(polygon) as polygon,
            sum(aland) / {meters_per_mile**2} as aland
        from (
            select
                *,
                case when N = (max(N) over (partition by geoid_new)) then {self.g.district_type} else NULL end as district
            from (
                select
                    A.geoid_new,
                    B.*,
                    count(1) over (partition by geoid_new, {self.g.district_type}) as N
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