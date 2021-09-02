from . import *

try:
    import pandas_bokeh
except:
    os.system('pip install --upgrade pandas-bokeh')
    import pandas_bokeh

@dataclasses.dataclass
class Analysis(Base):
    nodes : str
    tbl   : str
        
    def __post_init__(self):
        self.run = self.tbl.split(".")[-1]
        
        self.abbr, self.yr, self.level, self.district_type, _, self.seed = self.run.split('_')
        self.results_path = root_path / f'results/{self.run}'
        self.results_path.mkdir(parents=True, exist_ok=True)
        
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
#                 number_format="1.0 $",
                **{'fill_alpha' :.8,
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


    def get_results(self):
        try:
#             rpt(f'graph copy for {self.seed}')
            graph_source = root_path / f'redistricting_data/graph/{self.abbr}/graph_{self.run}.gpickle'
            graph_target = self.results_path / f'{self.run}_graph.gpickle'
            shutil.copy(graph_source, graph_target)
#             rpt(f'graph copy for {self.seed} - success')
        except Exception as e:
            rpt(f'graph copy for {self.seed} - FAIL {e}')

        try:
            rpt(f'summary copy for {self.seed}')
            self.summary = read_table(tbl=self.tbl+'_summary').sort_values('plan')
            fn = self.results_path / f'{self.run}_summary.csv'
            self.summary.to_csv(fn)
#             rpt(f'summary copy for {self.seed} - success')
        except Exception as e:
            rpt(f'summary copy for {self.seed} - FAIL {e}')
        

        try:
            rpt(f'results calculation for {self.seed}')
            cols = [c for c in get_cols(self.nodes) if c not in Levels + District_types + ['county', 'aland', 'perim', 'polsby_popper', 'density', 'polygon', 'point']]
            query = f"""
select
    D.*,
    {join_str(1).join([f'C.{c} as {c}' for c in cols])}
from (
    select
        A.{self.district_type},
        A.plan,
        {join_str(2).join([f'sum(B.{c}) as {c}' for c in cols])}
    from
        {self.tbl+'_plans'} as A
    left join
        {self.nodes} as B
    on
        A.geoid = B.geoid
    group by 
        1, 2
    ) as C
left join 
    {self.tbl+'_stats'} as D
on 
    C.{self.district_type} = D.{self.district_type} and C.plan = D.plan
order by
    plan, {self.district_type}
"""
            self.results = run_query(query)
            cols = self.results.columns.tolist()
            self.results.columns = [cols[1], cols[0]] + cols[2:]

            fn = self.results_path / f'{self.run}_results.csv'
            self.results.to_csv(fn)
#             rpt(f'results calulation for {self.seed} - SUCCESS')
        except Exception as e:
            rpt(f'results calulation for {self.seed} - FAIL {e}')