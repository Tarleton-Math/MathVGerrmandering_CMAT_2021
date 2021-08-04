@dataclasses.dataclass
class Combined(Variable):
    name: str = 'combined'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        super().__post_init__()


    def get(self):
        self.A = self.g.assignments
        self.A.cols = Levels + District_types
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
        self.agg(agg_tbl=self.g.assignments.tbl, agg_col=self.level, out_tbl=self.tbl, district_types=District_types, agg_shapes=True, agg_centroids=True, simplification=0)


    def agg(self, agg_tbl, agg_col, out_tbl, district_types=None, agg_shapes=True, agg_centroids=False, simplification=0):
        if district_types is None:
            district_types = self.g.district_type
        district_types = listify(district_types)

######## join tbl and agg_tbl ########
        print(f'joining {self.tbl} and {agg_tbl}', end=concat_str)
        temp = out_tbl + '_temp'
        query = f"""
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
        load_table(temp, query=query, preview_rows=0)

######## agg assignments by most frequent value in each column within each agg region ########
######## must compute do this one column at a time, then join ########
        print(f'aggregating {", ".join(district_types)}', end=concat_str)
        tbls = list()
        c = 64
        for col in district_types:
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
    A.geoid,
    {join_str(1).join(district_types)}
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
        for t in tbls:
            delete_table(t)

######## agg shapes, census, and votes then join with agg assignments above ########
        cols = self.C.cols + self.V.cols + self.H.cols
        sels = [f'sum({c}) as {c}' for c in cols]
        msg = 'aggregating census, votes_all, votes_hl'
        if not agg_shapes:
            query = f"""
select
    A.*,
    {join_str(1).join(cols)}
from
    {temp_assign} as A
left join (
    select
        geoid_new as geoid,
        {join_str(1).join(sels)}
    from
        {temp}
    group by
        1
    ) as B
on
    A.geoid = B.geoid
order by
    geoid
"""
            print(msg, end=concat_str)
            load_table(out_tbl, query=query, preview_rows=0)

        else:
            if agg_centroids:
                msg += ', centroids'
                sel_centroids = "st_centroid(geography) as centroid,"
            else:
                sel_centroids = ""            
            
            msg += f', and shapes with simplification {simplification}'
            if simplification >= 1:
                sel_geography = f"st_simplify(geography, {simplification}) as geography"
            else:
                sel_geography = "geography"
                
            query = f"""
select
    A.*,
    aland,
    perim,
    case when perim > 0 then round(4 * acos(-1) * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    {join_str(1).join(cols)},
    {sel_centroids}
    {sel_geography}
from
    {temp_assign} as A
left join (
    select
        *,
        st_perimeter(geography) as perim
    from (
        select
            geoid_new as geoid,
            {join_str(1).join(sels)},
            sum(aland) as aland,
            st_union_agg(geography) as geography
        from
            {temp}
        group by
            1
        )
    ) as B
on
    A.geoid = B.geoid
order by
    geoid
"""
            print(msg, end=concat_str)
            tbl_color_A = out_tbl + '_color_A'
            load_table(tbl_color_A, query=query, preview_rows=0)
            print('assigning colors', end=concat_str)

            query_edges = f"""
select
    x.geoid as geoid_x,
    y.geoid as geoid_y
from
    {tbl_color_A} as x,
    {tbl_color_A} as y
where
    x.geoid < y.geoid
    and st_intersects(x.geography, y.geography)
"""
            edges = run_query(query_edges)
            G = self.g.graph.edges_to_graph(edges)
            print(1+max(G.degree)[1])
            d = nx.equitable_color(G, num_colors=self.g.num_colors)
            
            colors = pd.DataFrame()
            colors['geoid'] = d.keys()
            colors['color'] = d.values()
            
            tbl_color_B = out_tbl + '_color_B'
            load_table(tbl_color_B, df=colors, preview_rows=0)
            query = f"""
select
    A.*,
    B.color
from
    {tbl_color_A} as A
inner join
    {tbl_color_B} as B
on
    A.geoid = B.geoid
"""
            load_table(out_tbl, query=query, preview_rows=0)
            delete_table(tbl_color_A)
            delete_table(tbl_color_B)
        
######## clean up ########
        delete_table(temp)
        delete_table(temp_assign)
