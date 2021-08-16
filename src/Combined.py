@dataclasses.dataclass
class Combined(Variable):
    name: str = 'combined'
    simplification: int = 0

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        super().__post_init__()


    def get(self):
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
        self.agg(agg_tbl=self.g.assignments.tbl, agg_col=self.level, out_tbl=self.tbl, agg_district=True, agg_polygon=True, agg_point=True, simplification=self.simplification, clr_tbl=None)

        
        
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
######## agg shapes, census, votes, and shapes ########
#######################################################
        msg = 'aggregating census, votes_all, votes_hl'
        cols = self.C.cols + self.V.cols + self.H.cols
        sels = [f'sum({c}) as {c}' for c in cols]
        if not agg_polygon:
            msg += 'without shapes'
            print('aggregating without shapes', end=concat_str)
            main_query = f"""
select
    geoid_new as geoid,
    {join_str(1).join(sels)}
from
    {temp_tbl}
group by
    1
"""

        else:
            msg += ', polygons'
            if agg_point:
                msg += ', and points'
                sel_point = "st_centroid(polygon) as point,"
            else:
                sel_point = ""            
            msg += f' with simplification {simplification}'
            if simplification >= 1:
                sel_polygon = f"st_simplify(polygon, {simplification}) as polygon"
            else:
                sel_polygon = "polygon"
                
            main_query = f"""
select
    geoid,
    aland,
    perim,
    case when perim > 0 then round(4 * acos(-1) * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    {join_str(1).join(cols)},
    {sel_point}
    {sel_polygon}
from (
    select
        *,
        st_perimeter(polygon) as perim
    from (
        select
            geoid_new as geoid,
            {join_str(3).join(sels)},
            sum(aland) as aland,
            st_union_agg(polygon) as polygon
        from
            {temp_tbl}
        group by
            1
        )
    )
"""
            
############################################################################################
######## agg districts by most frequent value in each column within each agg region ########
######## must do this one column at a time, then join ######################################
############################################################################################
        district_tbl = temp_tbl + '_district'    
        if agg_district:
            print(f'aggregating {", ".join(District_types)}', end=concat_str)
            tbls = list()
            c = 64
            for col in District_types:
                t = temp_tbl + f'_{col}'
                tbls.append(t)
                district_query = f"""
select
    geoid,
    {col},
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
"""
                load_table(t, query=district_query, preview_rows=0)
######## create the join query as we do each col so we can run it at the end ########
                c += 1
                if len(tbls) <= 1:
                    join_query = f"""
select
    A.geoid,
    {join_str(1).join(District_types)}
from
    {t} as A
"""
                else:
                    alias = chr(c)
                    join_query +=f"""
left join
    {t} as {alias}
on
    A.geoid = {alias}.geoid
"""
######## run join query ########
            load_table(district_tbl, query=join_query, preview_rows=0)
            for t in tbls:
                delete_table(t)

######## insert join into main query ########
            main_query = f"""
select
    A.*,
    {join_str(1).join(District_types)}
from (
    {main_query}
    ) as A
left join
    {district_tbl} as B
on
    A.geoid = B.geoid
"""
        
#############################
######## join colors ########
#############################
        if clr_tbl is not None:
            msg += f' with colors'
            main_query = f"""
select
    A.*,
    B.color
from (
    {main_query}
    ) as A
left join
    {clr_tbl} as B
on
    A.geoid = B.geoid
"""
        
#######################################################
######## run main_query and clean up temp tbls ########
#######################################################
        print(msg, end=concat_str)
        load_table(out_tbl, query=main_query, preview_rows=0)             
        delete_table(temp_tbl)
        delete_table(district_tbl)