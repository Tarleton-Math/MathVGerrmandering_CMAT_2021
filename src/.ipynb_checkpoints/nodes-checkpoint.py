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
                     'elections'  : get_cols(self.g.elections.tbl)
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
        if not self.g.county_line:
            query_temp = f"""
select
    geoid,
    {self.g.level} as geoid_new
from
    {self.g.assignments.tbl}
"""
    
        else:
            query_temp = f"""
select
    geoid,
    case when ct=1 then cnty else {self.g.level} end as geoid_new
from (
    select
        geoid,
        cnty,
        cntyvtd,
        count(distinct {self.g.district_type}) over (partition by cnty) as ct,
    from
        {self.g.assignments.tbl}
    )
"""
            
        sels = [f'sum({c}) as {c}' for c in self.cols['census'] + self.cols['elections']]
        query = f"""
select
    *,
    case when perim > 0 then round(4 * acos(-1) * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    st_centroid(polygon) as point
from (
    select
        *,
        st_perimeter(polygon) as perim
    from (
        select
            geoid_new as geoid,
            max(district) as {self.g.district_type},
            {join_str(3).join(sels)},
            st_union_agg(polygon) as polygon,
            sum(aland) as aland
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