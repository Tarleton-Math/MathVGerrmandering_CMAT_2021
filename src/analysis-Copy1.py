from . import *

try:
    import pandas_bokeh
except:
    os.system('pip install --upgrade pandas-bokeh')
    import pandas_bokeh

@dataclasses.dataclass
class Analysis(Base):
    nodes     : str
    results_bq: str
    seeds     : typing.Any
        
    def __post_init__(self):
        results_stem = self.results_bq.split(".")[-1]
        self.abbr, self.yr, self.level, self.district_type = results_stem.split('_')

        self.seeds_list = list()
        self.bq_list = list()
        for seed in self.seeds:
            tbl = self.results_bq + f'_{seed}_'
            if check_table(tbl + 'plans'):
                self.seeds_list.append(int(seed))
                self.bq_list.append(tbl)

        a, b = min(self.seeds_list), max(self.seeds_list)
        seeds_range = f'{str(a).rjust(4, "0")}_{str(b).rjust(4, "0")}'
        if all([s in self.seeds_list for s in range(a,b)]):
            seeds_range += '_complete'
        else:
            seeds_range += '_incomplete'
        self.tbl = self.results_bq + f'_{seeds_range}'
        self.pq = root_path / f'results/{results_stem}/{seeds_range}.parquet'
#         print(self.tbl, self.pq, self.bq_list)

    def compute_results(self):
        u = "\nunion all\n"
        stack = {key: u.join([f'select * from {bq}{key}' for bq in self.bq_list]) for key in ['plans', 'stats', 'summary']}

#         cols = [c for c in get_cols(self.nodes) if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]
        
        cols = [c for c in ['total_white'] if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]

        
        query = f"""
select
    B.seed,
    B.plan,
    C.{self.district_type},
    max(B.hash) as hash_plan,
    max(B.pop_imbalance) as pop_imbalance_plan,
    max(B.polsby_popper) as polsby_popper_plan,
    max(C.polsby_popper) as polsby_popper_district,
    max(C.aland) as aland,
    max(C.total_pop) as total_pop,
    max(C.total_pop) / sum(E.aland) as density,
    {join_str(1).join([f'sum(E.{c}) as {c}' for c in cols])}
from (
    select
        *
    from (
        select
            *,
            row_number() over (partition by A.hash order by plan asc, seed asc) as r
        from (
            {subquery(stack['summary'], indents=3)}
            ) as A
        )
    where r = 1
    ) as B
inner join (
    {subquery(stack['stats'], indents=1)}
    ) as C
on
    B.seed = C.seed and B.plan = C.plan
inner join (
    select
        *
    from (
        {subquery(stack['plans'], indents=2)}
        )
    ) as D
on
    C.seed = D.seed and C.plan = D.plan and C.{self.district_type} = D.{self.district_type}
inner join
    {self.nodes} as E
on
    D.geoid = E.geoid
group by
    seed, plan, {self.district_type}
order by
    seed, plan, {self.district_type}
"""
        load_table(tbl=self.tbl, query=query)
        self.fetch_results()
        self.save_results()
        
    def fetch_results(self):
        self.results = read_table(tbl=self.tbl)
        idx = ['seed', 'plan', 'cd']
        for col in idx:
            self.results[col] = rjust(self.results[col])
        self.results.sort_values(idx, inplace=True)
        return self.results
        
    def save_results(self):
        self.results.to_parquet(self.pq)
        to_gcs(self.pq)


    def plot(self, show=True):
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
