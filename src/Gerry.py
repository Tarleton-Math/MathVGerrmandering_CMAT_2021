@dataclasses.dataclass
class Gerry(Base):
    # These are default values that can be overridden when you create the object
    abbr              : str = 'TX'
    level             : str = 'tract'
    shapes_yr         : int = 2020
    census_yr         : int = 2010
    district          : str = 'cd'
    refresh_tbl       : typing.Tuple = ()
    refresh_all       : typing.Tuple = ()
    election_filters  : typing.Tuple = (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'")
    max_pop_imbalance : float = 10.0
    node_attrs        : typing.Tuple = ('geoid', 'total', 'aland', 'perim', 'polsby_popper')
    
    def __post_init__(self):
        check_level(self.level)
        check_district(self.district)
        check_year(self.census_yr)
        check_year(self.shapes_yr)
        self.state = states[states['abbr']==self.abbr].iloc[0]
        self.__dict__.update(self.state)

    def get_data(self):
        self.crosswalks  = Crosswalks(g=self)
        self.assignments = Assignments(g=self)
        self.shapes      = Shapes(g=self)
        self.census      = Census(g=self)
        self.elections   = Elections(g=self)
        self.votes_all   = Votes(g=self, group='all')
        self.votes_hl    = Votes(g=self, group='hl')
        self.combined    = Combined(g=self)
        self.edges       = Edges(g=self)
        self.nodes       = Nodes(g=self)
        self.graph       = Graph(g=self)


    def recomb_step(self):
        recom_found = False
        best_imbalance = 100
        district_pops = self.nodes.groupby(self.district)['pop'].sum().to_dict()
        D = district_pops.keys()
        P = rng.permutation([(a,b) for a in D for b in D if a < b]).tolist()
        
        for district_pair in rng.permutation(P):#.tolist():
            N = self.nodes.query(f'district in {district_pair}').copy()
            H = self.graph.subgraph(N.index)
            if not nx.is_connected(H):
                print(f'{district_pair} not connected')
                continue
            else:
                print(f'{district_pair} connected')
            pops = self.pops.copy()
            p0 = pops.pop(district_pair[0])
            p1 = pops.pop(district_pair[1])
            pop_pair = p0 + p1
            pop_min, pop_max = pops.min(), pops.max()
            trees = []
            for i in range(100):
                w = {e: rng.uniform() for e in H.edges}
                nx.set_edge_attributes(H, w, "weight")
                T = nx.minimum_spanning_tree(H)
                h = hash(tuple(sorted(T.edges)))
#                 print(h, trees)
                if h not in trees:
                    trees.append(h)
                    d = {e: T.degree[e[0]] + T.degree[e[1]] for e in T.edges}
                    max_tries = 0.02 * len(d)
#                     print(len(d), max_tries)
                    d = sorted(d.items(), key=lambda x:x[1], reverse=True)
                    for i, (e, deg) in enumerate(d):
                        if i > max_tries:
                            print(f'I unsuccessfully tried {i} edge cuts for tree {h} - trying a new tree')
                            break
                        elif i % 100 == 0:
                            print(i, e, deg, f'{best_imbalance:.2f}%')
                        T.remove_edge(*e)
                        comp = nx.connected_components(T)
                        next(comp)
                        s = sum(T.nodes[n]['pop_total'] for n in next(comp))
                        t = pop_pair - s
                        if t < s:
                            s, t = t, s
                        pop_imbalance = (max(t, pop_max) - min(s, pop_min)) / self.pop_ideal * 100
                        best_imbalance = min(best_imbalance, pop_imbalance)
    #                     print(h, s, t, pop_imbalance)
                        if pop_imbalance < self.pop_tolerance:
                            print(f'found split with pop_imbalance={pop_imbalance}')
                            recom_found = True
                            new = [list(c) for c in nx.connected_components(T)]
                            for n, d in zip(new, district_pair):
                                N.loc[n, 'district_new'] = d
                            i = N.groupby(['district','district_new'])['aland'].sum().idxmax()
                            if i[0] != i[1]:
                                new[0], new[1] = new[1], new[0]
                            for n, d in zip(new, district_pair):
                                self.nodes.loc[n, 'district'] = d
                            break
                        T.add_edge(*e)
                else:
                    print(f'Got a repeat spanning tree')
                if recom_found:
                    break
            if recom_found:
                break
        assert recom_found, "No suitable recomb step found"
        return recom_found


    def MCMC(self, steps=10):
        variable, yr, level = 'plans', self.level, self.shapes_yr
        self.get_data()
        self.get_graph()

        self.nodes['district'] = self.nodes[self.district].copy()
        self.pops = self.nodes.groupby('district')['pop_total'].sum()
        self.pop_ideal = self.pops.mean()
        pop_imbalance = (np.max(self.pops) - np.min(self.pops)) / self.pop_ideal * 100
        self.pop_tolerance = max(self.max_pop_imbalance, pop_imbalance)
        print(f'Current population imbalance = {pop_imbalance:.2f}% ... setting population imbalance tolerance = {self.pop_tolerance:.2f}%')
        self.districts = self.nodes['district'].unique()

        self.plans = [self.nodes['district'].copy().rename(f'plan_0')]
        for step in range(1,steps+1):
            if self.recomb_step():
                self.plans.append(self.nodes['district'].copy().rename(f'plan_{step}'))
        self.plans = self.nodes.join(pd.concat(self.plans, axis=1)).drop(columns='district')
        
        self.steps = steps+1
        self.write_results()
        
        
    def write_results(self):
        variable, yr, level = 'plans', self.level, self.shapes_yr
        tbl = self.table_id(variable, level, f'{yr}_{self.district}')
        tbl_temp = tbl + '_temp'
        print(f'loading {tbl_temp}', end=concat_str)
        load_table(tbl_temp, df=self.plans.reset_index(), overwrite=True)
        print(f'temp table written{concat_str}joining shapes', end=concat_str)
        query = f"""
select
    A.*,
    B.geography
from
    {tbl_temp} as A
inner join
    {self.table_id('shapes', yr, level)} as B
on
    A.geoid = B.geoid_{yr}
"""
        load_table(tbl, query=query)
        self.write_viz_tables()


    def write_viz_tables(self):
        variable, yr, level = 'plans', self.level, self.shapes_yr
        tbl = self.table_id(variable, level, f'{yr}_{self.district}')
        for step in range(self.steps):
            print(f'Creating viz table for step {step}')
            query = f"""

select
    plan_{step},
    pop_total,
    R,
    D,
    G,
    L,
    votes_total,
    case when perim > 0 then round(4 * acos(-1) * aland / (perim * perim) * 100) else 0 end as polsby_popper,
    aland,
    perim,
    geography
from (
    select
        *,
        st_perimeter(geography) as perim
    from (
        select
            plan_{step},
            sum(aland) as aland,
            round(sum(pop_total)) as pop_total,
            round(sum(R)) as R,
            round(sum(D)) as D,
            round(sum(G)) as G,
            round(sum(L)) as L,
            round(sum(votes_total)) as votes_total,
            st_union_agg(geography) as geography,
        from
            {tbl}
        group by
            1
        )
    )
"""
            load_table(f'{tbl}_{step}', query=query, overwrite=True)