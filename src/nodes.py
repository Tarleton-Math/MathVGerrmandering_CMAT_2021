from . import *
@dataclasses.dataclass
class Nodes(Variable):
    name: str = 'nodes'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        super().__post_init__()


    def get(self):
        self.tbl += f'_{self.g.district_type}_countyline{self.g.countyline_rule}'
        self.pq = self.tbl_to_file().with_suffix('.parquet')
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
#         cols = ['geoid', self.g.level, 'cnty', 'total_pop']
        if self.level in ['tabblock', 'bg', 'tract', 'cnty']:
            query_temp = f"select geoid, cnty, total_pop, {self.g.district_type}, substring({self.g.level}, 3) as level_temp from {self.raw}"
        else:
            query_temp = f"select geoid, cnty, total_pop, {self.g.district_type},           {self.g.level}     as level_temp from {self.raw}"
        
        if self.g.countyline_rule == 1:
            query_temp = f"""
select
    geoid,
    level_temp as geoid_new,
from
    ({query_temp})
"""
    
        elif self.g.countyline_rule == 2:
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
        elif self.g.countyline_rule == 3:
            query_temp = f"""
select
    geoid,
    case when target_districts > 1 then level_temp else substring(cnty, 3) end as geoid_new,
from (
    select
        geoid,
        level_temp,
        cnty,
        sum(total_pop) over (partition by cnty) / {self.g.target_pop} as target_districts
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
            max(county) as county,
            --max(district) as {self.g.district_type},
            cast(max(district) as int) as {self.g.district_type},
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