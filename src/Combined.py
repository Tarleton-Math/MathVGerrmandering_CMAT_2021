@dataclasses.dataclass
class Combined(Variable):
    name: str = 'combined'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        super().__post_init__()


    def get(self):
        self.A = self.g.assignments
        self.A.cols = ['tabblock', 'bg', 'tract', 'cnty', 'state', 'cntyvtd', 'cd', 'sldu', 'sldl']
        self.S = self.g.shapes
        self.S.cols = ['aland', 'geography']
        self.C = self.g.census
        self.C.cols = ['total']
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


        
        
        
    def process(self, agg_tbl=None, agg_col=None, out_tbl=None, agg_shapes=True, centroid=True):
        if agg_tbl is None:
            agg_tbl = self.g.assignments.tbl
        if agg_col is None:
            agg_col = self.level
        if out_tbl is None:
            out_tbl = self.tbl

######## join agg_tbl to raw ########
        temp = self.tbl + '_temp'
        query = f"""
select
    B.{agg_col} as geoid_new,
    A.*
from
    {self.raw} as A
left join
    {agg_tbl} as B
on
    A.geoid = B.geoid
"""
        load_table(temp, query=query, preview_rows=0)

######## agg assignments by most frequent value in each column within each agg region ########
######## must compute do this one column at a time, then join ########
        cols_assign = self.A.cols
        tbls = list()
        c = 64
        for col in cols_assign:
            t = temp + f'_{col}'
            tbls.append(t)
            query_assign = f"""
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
            {temp}
        group by
            geoid, {col}
        )
    )
where
    r = 1
"""
            load_table(t, query=query_assign, preview_rows=0)
            
######## create the join query as we do each col so we can run it at the end ########
            c += 1
            if len(tbls) <= 1:
                query_join = f"""
select
    {join_str(1).join(['A.geoid'] + cols_assign)}
from
    {t} as A
"""
            else:
                alias = chr(c)
                query_join +=f"""
left join
    {t} as {alias}
on
    A.geoid = {alias}.geoid
"""
            
######## run join query ########
        temp_assign = temp + '_assign'
        load_table(temp_assign, query=query_join, preview_rows=0)
            

######## agg shapes, census, and votes then join with agg assignments above ########
        cols_data = self.C.cols + self.V.cols + self.H.cols
        sels_data = [f'sum({c}) as {c}' for c in cols]
        
        query_data = f"""
select
    geoid_new as geoid,
    {join_str(1).join(data_sels)},
from
    {temp}
group by
    1
"""
        temp_data = temp + '_data'
        load_table(temp_data, query=query_data, preview_rows=0)


        if centroid:
            centroid_sel = "st_centroid(geography) as point,"
        else:
            centroid_sel = ""
        query_shapes = f"""
select
    *,
    aland,
    perim,
    case when perim > 0 then round(4 * acos(-1) * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    {centroid_sel}
    geography
from (
    select
        *,
        st_perimeter(geography) as perim
    from (
        select
            geoid_new as geoid,
            sum(aland) as aland,
            st_union_agg(geography) as geography
        from
            {temp}
        group by
            1
        )
    )
"""
        temp_shapes = temp + '_shapes'
        if agg_shapes:
            load_table(temp_shapes, query=query_shapes, preview_rows=0)
            query = f"""
select
    A.*,
    aland,
    perim,
    polsby_popper,
    {join_str(1).join(cols_data)},
    point,
    geography
from
    {temp_assign} as A
left join
    {temp_data} as D
on
    A.geoid = D.geoid
left join
    {temp_shape} as S
on
    A.geoid = S.geoid
"""
        else:
            query = f"""
select
    A.*,
    {join_str(1).join(cols_data)},
from
    {temp_assign} as A
left join
    {temp_data} as D
on
    A.geoid = D.geoid
"""
        load_table(out_tbl, query=query, preview_rows=0)
        
######## clean up ########
        delete_table(temp)
        delete_table(temp_assign)
        delete_table(temp_data)
        delete_table(temp_shapes)
        for t in tbls:
            delete_table(t)
        
        
        
        
        
        
        
#     def process(self, agg_tbl=None, agg_col=None, out_tbl=None):
#         if agg_tbl is None:
#             agg_tbl = self.g.assignments.tbl
#         if agg_col is None:
#             agg_col = self.level
#         if out_tbl is None:
#             out_tbl = self.tbl

# ######## join agg_tbl to raw ########
#         temp = self.tbl + '_temp'
#         query = f"""
# select
#     B.{agg_col} as geoid_new,
#     A.*
# from
#     {self.raw} as A
# left join
#     {agg_tbl} as B
# on
#     A.geoid = B.geoid
# """
#         load_table(temp, query=query, preview_rows=0)

# ######## agg assignments by most frequent value in each column within each agg region ########
# ######## must compute do this one column at a time, then join ########
#         cols = self.A.cols
#         tbls = list()
#         c = 64
#         for col in cols:
#             t = temp + f'_{col}'
#             tbls.append(t)
#             query = f"""
# select
#     geoid,
#     {col},
# from (
#     select
#         *,
#         row_number() over (partition by geoid order by N desc) as r
#     from (
#         select
#             geoid_new as geoid,
#             {col},
#             count(1) as N
#         from
#             {temp}
#         group by
#             geoid, {col}
#         )
#     )
# where
#     r = 1
# """
#             load_table(t, query=query, preview_rows=0)
            
# ######## create the join query as we do each col so we can run it at the end ########
#             c += 1
#             if len(tbls) <= 1:
#                 query_join = f"""
# select
#     {join_str(1).join(['A.geoid'] + cols)}
# from
#     {t} as A
# """
#             else:
#                 alias = chr(c)
#                 query_join +=f"""
# left join
#     {t} as {alias}
# on
#     A.geoid = {alias}.geoid
# """
            
# ######## run join query ########
#         temp_assign = temp + '_assign'
#         load_table(temp_assign, query=query_join, preview_rows=0)
            

# ######## agg shapes, census, and votes then join with agg assignments above ########
#         cols = self.C.cols + self.V.cols + self.H.cols
#         sels = [f'sum({c}) as {c}' for c in cols]
#         query = f"""
# select
#     A.*,
#     aland,
#     perim,
#     case when perim > 0 then round(4 * acos(-1) * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
#     {join_str(1).join(cols)},
#     st_centroid(geography) as point,
#     geography
# from
#     {temp_assign} as A
# left join (
#     select
#         *,
#         st_perimeter(geography) as perim
#     from (
#         select
#             geoid_new as geoid,
#             {join_str(1).join(sels)},
#             sum(aland) as aland,
#             st_union_agg(geography) as geography
#         from
#             {temp}
#         group by
#             1
#         )
#     ) as B
# on
#     A.geoid = B.geoid
# order by
#     geoid
# """
#         load_table(out_tbl, query=query, preview_rows=0)
        
# ######## clean up ########
#         delete_table(temp)
#         delete_table(temp_assign)
#         for t in tbls:
#             delete_table(t)