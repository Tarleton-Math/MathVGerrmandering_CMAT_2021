@dataclasses.dataclass
class Gerry(Base):
    # These are default values that can be overridden when you create the object
    abbr              : str = 'TX'
    level             : str = 'tract'
    shapes_yr         : int = 2020
    census_yr         : int = 2020
    district_type     : str = 'cd'
    agg_polygon       : bool = True
    agg_point         : bool = False
    simplification    : int = 600
    num_colors        : int = 8
    refresh_tbl       : typing.Tuple = ()
    refresh_all       : typing.Tuple = ()
    election_filters  : typing.Tuple = (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'")
    pop_imbalance_tol : float = 10.0
    node_attrs        : typing.Tuple = ('total_pop', 'aland', 'perim', 'polsby_popper')
    
    def __post_init__(self):
        check_level(self.level)
        check_district_type(self.district_type)
        check_year(self.census_yr)
        check_year(self.shapes_yr)
        self.state = states[states['abbr']==self.abbr].iloc[0]
        self.__dict__.update(self.state)

    def get_data(self):
        self.tbl = f'{bq_dataset}.{user_name}_plans_{self.state.abbr}_{self.census_yr}_{self.level}_{self.district_type}'
        self.crosswalks    = Crosswalks(g=self)
        self.assignments   = Assignments(g=self)
        self.shapes        = Shapes(g=self)
        self.census        = Census(g=self)
        self.elections     = Elections(g=self)
        self.votes_all     = Votes(g=self, group='all')
        self.votes_hl      = Votes(g=self, group='hl')
        self.combined      = Combined(g=self, simplification=0)
#         self.combined_simp = Combined(g=self, simplification=self.simplification, name='combined_simp')
        self.edges       = Edges(g=self)
        self.nodes         = Nodes(g=self)
        self.graph         = Graph(g=self)
        self.districts     = Districts(g=self)


    def get_components(self, G=None):
        if G is None:
            G = self.graph.graph
        return sorted([tuple(c) for c in nx.connected_components(G)], key=lambda x:len(x), reverse=True)

        
    def get_colors(self, G=None):
        if G is None:
            G = self.graph.graph
        k = max([d for n, d in G.degree]) + 1
        return pd.Series(nx.equitable_color(G, k)) + 1

        
    def MCMC(self, steps=10):
        self.get_data()

        d = len(str(steps))
        f = lambda k: f"plan_{str(k).rjust(d, '0')}"
        g = lambda k: self.nodes.df[self.districts.name].copy().astype(str).rename(f(k))

        self.plans  = [g(0)]
        self.hashes = [self.districts.hash]
        self.stats  = [self.districts.stats.copy()]
        for step in range(1,steps+1):
            print(f'MCMC {f(step)}', end=concat_str)
            while True:
                if self.graph.recomb():
                    self.districts.update()
                    if self.districts.hash not in self.hashes:
                        print(self.districts.hash, end=concat_str)
                        self.hashes.append(self.districts.hash)
                        self.plans.append(g(step))
                        
                        self.districts.stats['plan'] = step
                        self.stats.append(self.districts.stats.copy())
                        print('success')
                        break
                    else:
                        print(f'Found a duplicate plan at {f(step)} - discarding and trying again; hash = {self.districts.hash}')
                else:
                    print(f'No suitable recomb found at {f(step)} - trying again')
                
        print('MCMC done')
        self.plans = pd.concat(self.plans, axis=1)
        load_table(tbl=self.tbl, df=self.plans.reset_index(), preview_rows=0)
        
        self.stats = pd.concat(self.stats, axis=0).reset_index()
        cols = self.stats.columns.to_list()
        cols[0], cols[1] = cols[1], cols[0]
        load_table(tbl=self.tbl+'_stats', df=self.stats[cols], preview_rows=0)
        self.agg_plans()
        
    def agg_plans(self):
        plans = self.plans.columns
        tbls = list()
        for col in plans:
            out_tbl = self.tbl + f"_{col.split('_')[-1]}"
            tbls.append(out_tbl)
            print(f'Post-processing {col} to make {out_tbl}', end=concat_str)
            self.combined.agg(agg_tbl=self.tbl, agg_col=col, out_tbl=out_tbl, agg_district=False, agg_polygon=self.agg_polygon, agg_point=self.agg_point, clr_tbl=self.districts.tbl, simplification=self.simplification)
            print('done')

#         cols = ',\n    '.join([s.name for s in bqclient.get_table(tbls[0]).schema if s.name not in ['polygon', 'color']])
        L = [f"""
select
    {int(t.split('_')[-1])} as plan,
    *
from 
    {t}
"""
            for t in tbls]
        j = "\nunion all\n"
        query = j.join(L) + "\norder by plan, geoid"
        load_table(tbl=self.tbl+'_stacked', query=query, preview_rows=0)
        for t in tbls:
            delete_table(t)