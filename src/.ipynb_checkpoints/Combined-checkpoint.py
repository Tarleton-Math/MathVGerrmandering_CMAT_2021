@dataclasses.dataclass
class Combined(Variable):
    name: str = 'combined'
    simplification: int = 0

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        super().__post_init__()


    def get(self):
        self.tbl += f'_{self.district_type}'
        self.A = self.g.assignments
        self.A.cols = Levels + District_types
        self.S = self.g.shapes
        self.S.cols = ['aland', 'polygon']
        self.C = self.g.census
        self.C.cols = census_columns['data']
        self.V = self.g.votes_all
        E = get_cols(self.V.tbl)
        self.V.cols = [f'{e}_all' for e in E]
        self.H = self.g.votes_hl
        self.H.cols = [f'{e}_hl' for e in E]
        
        exists = super().get()
        if not exists['tbl']:
            if not exists['raw']:
                print(f'creating raw table', end=concat_str)
                self.process_raw()
            print(f'creating table', end=concat_str)
            self.process()
        return self


    def process_raw(self):
        A_sels = [f'A.{c}' for c in self.A.cols]
        S_sels = [f'S.{c}' for c in self.S.cols]
        C_sels = [f'coalesce(C.{c}, 0) as {c}' for c in self.C.cols]
        V_sels = [f'coalesce(V.{c[:-4]}, 0) as {c}' for c in self.V.cols]
        H_sels = [f'coalesce(H.{c[:-3]}, 0) as {c}' for c in self.H.cols]
        E_sels = [sel for z in zip(V_sels, H_sels) for sel in z]
        sels = A_sels + C_sels + E_sels + S_sels 
        query = f"""
select
    A.geoid,
    {join_str(1).join(sels)},
from
    {self.A.tbl} as A
left join
    {self.S.tbl} as S
on
    A.geoid = S.geoid
left join
    {self.C.tbl} as C
on
    A.geoid = C.geoid
left join
    {self.V.tbl} as V
on
    A.geoid = V.geoid
left join
    {self.H.tbl} as H
on
    A.geoid = H.geoid
"""
        load_table(self.raw, query=query, preview_rows=0)


    def process(self):
        bqclient.copy_table(self.raw, self.tbl).result()
        assign_tbl = self.tbl+'_assign'
        
        if not self.g.county_line:
            query = f"""
select
    geoid,
    {self.level}
from
    {self.g.assignments.tbl}
"""
    
        else:
            query = f"""
select
    geoid,
    case when A=B then cnty else {self.level} end as {self.level}
from (
    select
        geoid,
        cnty,
        cntyvtd,
        max({self.district_type}) over (partition by cnty) as A,
        min({self.district_type}) over (partition by cnty) as B
    from
        {self.g.assignments.tbl}
    )
"""
        
        load_table(assign_tbl, query=query, preview_rows=0)
        self.agg(agg_tbl=assign_tbl, agg_col=self.level, out_tbl=self.tbl, agg_district=True, agg_polygon=True, agg_point=True, simplification=self.simplification, clr_tbl=None)
        delete_table(assign_tbl)

        
        
    def agg(self, agg_tbl, agg_col, out_tbl, agg_district=True, agg_polygon=True, agg_point=False, simplification=0, clr_tbl=None):
######################################
######## join tbl and agg_tbl ########
######################################
        print('joining', end=concat_str)
        temp_tbl = out_tbl + '_temp'
        temp_query = f"""
select
    B.{agg_col} as geoid_new,
    A.*
from
    {self.tbl} as A
left join
    {agg_tbl} as B
on
    A.geoid = B.geoid
"""
        load_table(temp_tbl, query=temp_query, preview_rows=0)

#######################################################
######## agg census and votes########## ###############
#######################################################
        msg = 'aggregating census, votes_all, votes_hl'
        cols = self.C.cols + self.V.cols + self.H.cols
        sels = [f'sum({c}) as {c}' for c in cols]
        query = f"""
select
    geoid_new as geoid,
    {join_str(1).join(sels)}
from
    {temp_tbl}
group by
    1
"""

############################################################################################
######## agg district by most frequent value in each column within each agg region #########
############################################################################################
        col = self.g.district_type    
        if not agg_district:
            query = f"""
select
    A.*,
    '0' as {col}
from (
    {query}
    ) as A
"""
        else:
            msg += f', {col}'
            query = f"""
select
    A.*,
    {col}
from (
    {query}
    ) as A
left join (
    select
        geoid,
        {col}
    from (
        select
            *,
            row_number() over (partition by geoid order by N desc) as r
        from (
            select
                geoid_new as geoid,
                {col},
                count(1) as N
            from
                {temp_tbl}
            group by
                geoid, {col}
            )
        )
    where
        r = 1
    ) as B
on
    A.geoid = B.geoid
"""

#######################################################
######## join colors #################################
#######################################################
        if clr_tbl is None:
            msg += f' without colors'
            query = f"""
select
    C.*,
    0 as color
from (
    {query}
    ) as C
"""
        else:
            msg += f' with colors'
            query = f"""
select
    C.*,
    D.color
from (
    {query}
    ) as C
left join
    {clr_tbl} as D
on
    C.geoid = D.geoid
"""   

#######################################################
######## agg shapes ###################################
#######################################################
        if not agg_polygon:
            msg += ' with polygons'
            query = f"""
select
    E.*,
    0 as aland,
    0 as perim,
    0 as polsby_popper,
    ST_GEOGFROMTEXT("POINT(-100.0 31.0)") as point,
    ST_GEOGFROMTEXT("POLYGON((-100.0 31.0, -100.1 31.0, -100.0 31.1, -100.0 31.0))") as polygon
from (
    {query}
    ) as E
"""
        else:
            msg += f' with polygons with simplification {simplification}'
            if agg_point:
                msg += ' and points'
                sel_point = "st_centroid(polygon) as point"
            else:
                sel_point = f'ST_GEOGFROMTEXT("POINT(-100.0 31.0)") as point'
            query = f"""
select
    E.*,
    aland,
    perim,
    case when perim > 0 then round(4 * acos(-1) * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    {sel_point},
    st_simplify(polygon, {simplification}) as polygon
from (
    {query}
    ) as E
left join (
    select
        *,
        st_perimeter(polygon) as perim
    from (
        select
            geoid_new as geoid,
            st_union_agg(polygon) as polygon,
            sum(aland) as aland
        from
            {temp_tbl}
        group by
            1
        )
    ) as F
on
    E.geoid = F.geoid
"""

#######################################################
######## run query and clean up temp tbls ########
#######################################################
        query += "order by geoid"
        print(msg, end=concat_str)
        load_table(out_tbl, query=query, preview_rows=0)             
        delete_table(temp_tbl)