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
        
    def plot(self):
        df = read_table(tbl=self.tbl+'_plans')
        df = df.pivot(index='geoid', columns='plan').astype(int)
        df.columns = df.columns.droplevel().rename(None)
        d = len(str(df.columns.max()))
        plans = ['plan_'+str(c).rjust(d, '0') for c in df.columns]
        df.columns = plans
        
        shapes = run_query(f'select geoid, total_pop, aland, polygon from {self.nodes}')
        df = df.merge(shapes, on='geoid')
        geo = gpd.GeoSeries.from_wkt(df['polygon'], crs='EPSG:4326').simplify(0.001).buffer(0) #<-- little white space @ .001 ~5.7 mb, minimal at .0001 ~10mb, with no white space ~37mb
        # geo = gpd.GeoSeries.from_wkt(df['polygon'], crs='EPSG:4326').buffer(0) # <-------------------- to not simplify at all
        self.gdf = gpd.GeoDataFrame(df.drop(columns='polygon'), geometry=geo)

        pandas_bokeh.output_notebook() #<------------- uncommment to view in notebook
        fig = self.gdf.plot_bokeh(
            figsize=(900, 600),
            slider = plans,
            slider_name = "PLAN #",
            show_colorbar=False,
            colormap= "Category20",
            hovertool_columns=['total_pop', 'aland'],
            tile_provider="CARTODBPOSITRON",
            return_html=True,
            show_figure=True,
            **{'fill_alpha' :.8,
              'line_alpha':.05,}
        )
        fn = root_path / f'images/{self.tbl.split(".")[-1]}.html'
        with open(fn, 'w') as file:
            file.write(fig)
        return fig


    def get_results(self):
        cols = [c for c in get_cols(self.nodes) if c not in Levels + District_types + ['polygon', 'aland', 'perim', 'polsby-popper', 'point']]
        query = f"""
select
    A.cd,
    A.plan,
    {join_str(1).join([f'sum(B.{c}) as {c}' for c in cols])}
from
    {self.plans} as A
left join
    {self.nodes} as B
on
    A.geoid = B.geoid
group by
    cd, plan
"""
        self.results = run_query(query)