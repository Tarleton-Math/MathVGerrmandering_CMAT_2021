from . import *

try:
    import pandas_bokeh
except:
    os.system('pip install --upgrade pandas-bokeh')
    import pandas_bokeh

@dataclasses.dataclass
class Analysis(Base):
    nodes : str
    mcmc  : str
    seeds : typing.Any
        
    def __post_init__(self):
        a = self.mcmc.split(".")
        b = a[-1].split('_')[:-2]
        self.base = f'{a[0]}.{a[1]}.{"_".join(b)}_seed'
        b.append(str(pd.Timestamp.now().round("s")).replace(' ','_').replace('-','_').replace(':','_'))
        self.abbr, self.yr, self.level, self.district_type, self.timestamp = b
        self.tbl = f'{a[0]}.{a[1]}.{'_'.join(b)}'
        

    def compute_results(self):
        u = "\nunion all\n"
        summary_stack = u.join([f'select * from {self.base}_{str(seed).rjust(4, "0")}_summary' for seed in self.seeds])
        stats_stack   = u.join([f'select * from {self.base}_{str(seed).rjust(4, "0")}_stats'   for seed in self.seeds])
        plans_stack   = u.join([f'select * from {self.base}_{str(seed).rjust(4, "0")}_plans'   for seed in self.seeds])

        cols = [c for c in get_cols(self.nodes) if c not in Levels + District_types + ['geoid', 'county', 'total_pop', 'polygon', 'aland', 'perim', 'polsby_popper', 'density', 'point']]
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
            {subquery(summary_stack, indents=3)}
            ) as A
        )
    where r = 1
    ) as B
inner join (
    {subquery(stats_stack, indents=1)}
    ) as C
on
    B.seed = C.seed and B.plan = C.plan
inner join (
    select
        *
    from (
        {subquery(plans_stack, indents=2)}
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
        self.file = f'results/{self.tbl.split(".")[-1]}.parquet'
        self.results.to_parquet(str(root_path / self.file), index=False)#, partition_cols='seed')
        self.results.to_parquet(f"gs://{proj_id}-bucket/{self.file}")
        return self.results
        
        
    
#             fn = self.results_path / f'{self.run}_results.csv'
#             self.results.to_csv(fn, index=False)
#             rpt(f'results calulation for {self.seed} - SUCCESS')
#         except Exception as e:
#             rpt(f'results calulation for {self.seed} - FAIL {e}')
    
    
#         try:
# #             rpt(f'graph copy for {self.seed}')
#             graph_source = root_path / f'redistricting_data/graph/{self.abbr}/graph_{self.run}.gpickle'
#             graph_target = self.results_path / f'{self.run}_graph.gpickle'
#             shutil.copy(graph_source, graph_target)
# #             rpt(f'graph copy for {self.seed} - success')
#         except Exception as e:
# #             rpt(f'graph copy for {self.seed} - FAIL {e}')
#             pass
    
    
#         try:
#             self.summary = read_table(tbl=self.tbl+'_summary').sort_values('plan')
#             fn = self.results_path / f'{self.run}_summary.csv'
#             self.summary.to_csv(fn, index=False)
# #             rpt(f'summary copy for {self.seed} - succeed')
#         except Exception as e:
#             rpt(f'summary copy for {self.seed} - FAIL {e}')
        
#         try:
#             self.plans = read_table(tbl=self.tbl+'_plans').sort_values('plan')
#             fn = self.results_path / f'{self.run}_plans.csv'
#             self.plans.to_csv(fn, index=False)
# #             rpt(f'plans copy for {self.seed} - succeed')
#         except Exception as e:
#             rpt(f'plans copy for {self.seed} - FAIL {e}')

#         try:
#             cols = [c for c in get_cols(self.nodes) if c not in Levels + District_types + ['county', 'perim', 'polsby_popper', 'total_pop', 'density', 'polygon', 'point']]
#             query = f"""
# select
#     D.*,
#     case when aland > 0 then total_pop / aland else 0 end as density,
#     {join_str(1).join([f'C.{c} as {c}' for c in cols])}
# from (
#     select
#         A.{self.district_type},
#         A.plan,
#         {join_str(2).join([f'sum(B.{c}) as {c}' for c in cols])}
#     from
#         {self.tbl+'_plans'} as A
#     left join
#         {self.nodes} as B
#     on
#         A.geoid = B.geoid
#     group by 
#         1, 2
#     ) as C
# left join 
#     {self.tbl+'_stats'} as D
# on 
#     C.{self.district_type} = D.{self.district_type} and C.plan = D.plan
# order by
#     plan, {self.district_type}
# """
#             self.results = run_query(query)
# #             cols = self.results.columns.tolist()
# #             self.results.columns = [cols[1], cols[0]] + cols[2:]

#             fn = self.results_path / f'{self.run}_results.csv'
#             self.results.to_csv(fn, index=False)
# #             rpt(f'results calulation for {self.seed} - SUCCESS')
#         except Exception as e:
#             rpt(f'results calulation for {self.seed} - FAIL {e}')





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
