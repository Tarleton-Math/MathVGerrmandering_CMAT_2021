from . import *

@dataclasses.dataclass
class Analysis(Base):
    nodes_tbl            : str
    max_results          : int = None
    stack_size           : int = 100
    pop_imbalance_thresh : float = 10.0
        
    def __post_init__(self):
        self.results_stem = self.nodes_tbl.split('.')[-1][6:]
        self.abbr, self.yr, self.level, self.district_type = self.results_stem.split('_')
        self.results_bq = f'{root_bq}.{self.results_stem}'
        self.tbl = f'{self.results_bq}.{self.results_stem}_0000000_allresults'
        file = root_path / f'results/{self.results_stem}/{self.results_stem}_0000000_allresults'
        self.pq  = file + '.parquet'
        self.csv = file + '.csv'
        delete_table(self.tbl)

    def compute_results(self):
        self.tbls = dict()
        for src_tbl in bqclient.list_tables(self.results_bq, max_results=self.max_results):
            full  = src_tbl.full_table_id.replace(':', '.')
            short = src_tbl.table_id
            seed = short.split('_')[-2]
            key  = short.split('_')[-1]
            if seed.isnumeric():
                try:
                    self.tbls[seed][key] = full
                except:
                    self.tbls[seed] = {key : full}
        
        cols = [c for c in get_cols(self.nodes_tbl) if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]

#         cols = [c for c in ['total_white', 'total_black', 'total_native', 'total_asian', 'total_pacific', 'total_other'] + get_cols(self.nodes_tbl)[301:] if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]
#         print(cols)
        
        def join(d):
            query = f"""
select
    A.seed,
    A.plan,
    A.{self.district_type},
    A.geoid,
    C.hash_plan,
    C.pop_imbalance_plan,
    C.polsby_popper_plan,
    B.polsby_popper_district,
    B.aland,
    B.total_pop,
    B.total_pop / B.aland as density
from (
    select
        cast(seed as int) as seed,
        cast(plan as int) as plan,
        cast({self.district_type} as int) as {self.district_type},
        geoid
    from
        {d['plans']}
    ) as A
inner join (
    select
        cast(seed as int) as seed,
        cast(plan as int) as plan,
        cast({self.district_type} as int) as {self.district_type},
        aland,
        polsby_popper as polsby_popper_district,
        total_pop
    from
        {d['stats']}
    ) as B
on
    A.seed = B.seed and A.plan = B.plan and A.{self.district_type} = B.{self.district_type}
inner join (
    select
        cast(seed as int) as seed,
        cast(plan as int) as plan,
        Z.hash as hash_plan,
        pop_imbalance pop_imbalance_plan,
        polsby_popper as polsby_popper_plan
    from
        {d['summaries']} as Z
    where
        pop_imbalance < {self.pop_imbalance_thresh}
    ) as C
on
    A.seed = C.seed and A.plan = C.plan
"""
            return query
        
        temp_tbls = list()
        u = '\nunion all\n'
        k = len(self.tbls)
        for seed, tbls in self.tbls.items():
            k -= 1
            if len(tbls) == 3:
                try:
                    stack_query = stack_query + u + join(tbls)
                except:
                    stack_query = join(tbls)
                
            if k % self.stack_size == 0:
                query = f"""
select
    A.seed,
    A.plan,
    A.{self.district_type},
    max(A.hash_plan) as hash_plan,
    max(A.pop_imbalance_plan) as pop_imbalance_plan,
    max(A.polsby_popper_plan) as polsby_popper_plan,
    max(A.polsby_popper_district) as polsby_popper_district,
    max(A.aland) as aland,
    max(A.total_pop) as total_pop,
    max(A.density) as density,
    {join_str(1).join([f'sum(B.{c}) as {c}' for c in cols])}
from (
    select
        *
    from (
        select
            *,
            row_number() over (partition by hash_plan order by plan asc, seed asc) as r
        from (
            {subquery(stack_query, indents=3)}
            )
        )
    where r = 1
    ) as A
inner join
    {self.nodes_tbl} as B
on
    A.geoid = B.geoid
group by
    seed, plan, {self.district_type}
"""
                temp_tbls.append(self.tbl+f'_{k}')
                print(temp_tbls)
                load_table(tbl=temp_tbls[-1], query=query)
                del stack_query
                print(f'at step {k}')
        stack_query = u.join([f'select * from {tbl}' for tbl in temp_tbls])
        query = f"""
select
    *
from (
    select
        *,
        row_number() over (partition by hash_plan order by plan asc, seed asc) as r
    from (
        {subquery(stack_query, indents=3)}
        )
    )
where
    r = 1
order by
    seed, plan, {self.district_type}
"""
        load_table(tbl=self.tbl, query=query)
        for t in temp_tbls:
            delete_table(t)
        self.df = read_table(self.tbl)
        self.df.to_parquet(self.pq)
        self.df.to_csv(self.csv)
        to_gcs(self.pq)
        to_gcs(self.csv)


    def plot(self, show=True):
        try:
            import pandas_bokeh
        except:
            os.system('pip install --upgrade pandas-bokeh')
            import pandas_bokeh

        try:
            df = read_table(tbl=self.tbl+'_plans')
            df = df.pivot(index='geoid', columns='plan').astype(int)
            df.columns = df.columns.droplevel().rename(None)
            d = len(str(df.columns.max()))
            plans = ['plan_'+str(c).rjust(d, '0') for c in df.columns]
            df.columns = plans

            shapes = run_query(f'select geoid, county, total_pop, density, aland, perim, polsby_popper, polygon from {self.nodes}')
            df = df.merge(shapes, on='geoid')
            geo = gpd.GeoSeries.from_wkt(df['polygon'], crs='EPSG:4326').simplify(0.001).buffer(0) #<-- little white space @ .001 ~5.7 mb, minimal at .0001 ~10mb, with no white space ~37mb
#             geo = gpd.GeoSeries.from_wkt(df['polygon'], crs='EPSG:4326').buffer(0) # <-------------------- to not simplify at all
            self.gdf = gpd.GeoDataFrame(df.drop(columns='polygon'), geometry=geo)

            if show:
                pandas_bokeh.output_notebook() #<------------- uncommment to view in notebook
            fig = self.gdf.plot_bokeh(
                figsize = (900, 600),
                slider = plans,
                slider_name = "PLAN #",
                show_colorbar = False,
                colorbar_tick_format="0",
                colormap = "Category20",
                hovertool_string = '@geoid, @county<br>pop=@total_pop<br>density=@density{0.0}<br>land=@aland{0.0}<br>pp=@polsby_popper{0.0}',
                tile_provider = "CARTODBPOSITRON",
                return_html = True,
                show_figure = show,
                **{'fill_alpha' :.5,
                  'line_alpha':.05,}
            )
            fn = self.results_path / f'{self.run}_map.html'
            with open(fn, 'w') as file:
                file.write(fig)
#             rpt(f'map creation for {self.seed} - success')
        except Exception as e:
            rpt(f'map creation for {self.seed} - FAIL {e}')
            fig = None
        return fig