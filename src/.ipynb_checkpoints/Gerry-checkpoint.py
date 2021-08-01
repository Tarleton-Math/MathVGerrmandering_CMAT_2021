@dataclasses.dataclass
class Gerry(Base):
    # These are default values that can be overridden when you create the object
    abbr              : str = 'TX'
    level             : str = 'tract'
    shapes_yr         : int = 2020
    census_yr         : int = 2010
    district          : str = 'cd'
    refresh_tbl : typing.Tuple = ()
    refresh_all : typing.Tuple = ()
    election_filters  : typing.Tuple = (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'")
    max_pop_imbalance : float = 10.0
    
    def __post_init__(self):
        check_level(self.level)
        check_district(self.district)
        check_year(self.census_yr)
        check_year(self.shapes_yr)
        self.state = states[states['abbr']==self.abbr].iloc[0]
        self.__dict__.update(self.state)


    def get_data(self):
        for Property in [Crosswalks, Assignments, Shapes, Census, Elections]:
            self[Property.name] = Property(g=self)
        for grp in Groups:
            self[f'votes_{grp}'] = Votes(g=self, group=grp)
        self['combined'] = Combined(g=self)




    def get_nodes(self, tbl):
        variable, abbr, yr, level, district = self.table_sep(tbl)
        query = f"""
select
    A.*,
    B.cd,
    B.sldu,
    B.sldl,
    B.pop_total
from
    {self.table_id('shapes', self.shapes_yr, level)} as A
inner join (
    select
        {level},
        min(cd) as cd,
        min(sldu) as sldu,
        min(sldl) as sldl,
        sum(total) as pop_total
    from
        {self.table_id('census', self.census_yr)} as A
    group by
        1
    ) as B
on
    A.{level} = B.{level}
"""
        load_table(tbl, query=query, preview_rows=0)


    def get_edges(self, tbl):
        variable, abbr, yr, level, district = self.table_sep(tbl)
        query = f"""
select
    *
from (
    select
        x.{level} as {level}_x,
        y.{level} as {level}_y,        
        st_distance(x.point, y.point) as distance,
        st_length(st_intersection(x.geography, y.geography)) as shared_perim
    from
        {self.table_id('shapes')} as x,
        {self.table_id('shapes')} as y
    where
        x.{level} < y.{level} and st_intersects(x.geography, y.geography)
    )
where shared_perim > 0.1
"""
        load_table(tbl, query=query, preview_rows=0)


#######################################################################
####################### Make graph and run MCMC #######################
#######################################################################
            
    def edges_to_graph(self, edges):
        edge_attr = ['distance', 'shared_perim']
        return nx.from_pandas_edgelist(edges, source=f'{self.level}_x', target=f'{self.level}_y', edge_attr=edge_attr)


    def get_graph(self):
        variable, yr, level, district = 'graph', self.shapes_yr, self.level, self.district
        file = self.file_id(variable, yr, level, district, suffix='gpickle')
        if variable in self.overwrite:
            if hasattr(self, variable):
                delattr(self, variable)
            if file.is_file():
                file.unlink()
        
        try:
            self.nodes
        except:
            self.nodes = read_table(self.table_id('nodes', yr, self.level), cols=[level, 'cd', 'sldu', 'sldl', 'pop_total', 'aland', 'perim'])
        self.nodes.set_index(level, inplace=True)

        print(f"Get {variable} {self.name} {yr} {level} {self.district}".ljust(44, ' '), end=concat_str)
        try:
            self.graph
            print(f'already defined', end=concat_str)
        except:
            try:
                self.graph = nx.read_gpickle(file)
                print(f'gpickle file exists', end=concat_str)
            except:
                print(f'making graph', end=concat_str)
                try:
                    self.edges
                except:
                    self.edges = read_table(self.table_id('edges', yr, self.level))
                self.graph = self.edges_to_graph(self.edges)
                print(f'connecting districts', end=concat_str)
                tbl_shapes = self.table_id('shapes', yr, self.level)
                for dist, nodes in self.nodes.groupby(self.district):
                    while True:
                        H = self.graph.subgraph(nodes.index)
                        components = sorted([list(c) for c in nx.connected_components(H)], key=lambda x:len(x), reverse=True)
                        print(len(components))
                        print(f'\n{self.name} {level} {yr} {self.district.upper()} district {str(dist).rjust(3, " ")} has {str(len(components)).rjust(3, " ")} connected components with {[len(c) for c in components]} nodes ... adding edges to connect', end=concat_str)
                        if len(components) == 1:
                            break
                        c = ["', '".join(components[i]) for i in range(2)]
                        query = f"""
select
    {level}_x,
    {level}_y,
    distance,
    0.0 as shared_perim
from (
    select
        *,
        min(distance) over () as m
    from (
        select
            A.{level} as {level}_x,
            B.{level} as {level}_y,
            st_distance(A.point, B.point) as distance
        from
            {tbl_shapes} as A,
            {tbl_shapes} as B
        where
            A.{level} in ('{c[0]}') and B.{level} in ('{c[1]}')
        )
    )
where distance < 1.05 * m
"""
                        new_edges = bqclient.query(query).result().to_dataframe()
                        self.graph.update(self.edges_to_graph(new_edges))
                        print(f'done', end='', flush=True)
                nx.set_node_attributes(self.graph, self.nodes.to_dict('index'))
                file.parent.mkdir(parents=True, exist_ok=True)
                nx.write_gpickle(self.graph, file)
        print(f'success \n-----------------------------------------------------------------------------------')


    def recomb_step(self):
        recom_found = False
        min_imbalance = 100
        for district_pair in rng.permutation([(a,b) for a in self.districts for b in self.districts if a < b]).tolist():
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
                            print(i, e, deg, f'{min_imbalance:.2f}%')
                        T.remove_edge(*e)
                        comp = nx.connected_components(T)
                        next(comp)
                        s = sum(T.nodes[n]['pop_total'] for n in next(comp))
                        t = pop_pair - s
                        if t < s:
                            s, t = t, s
                        pop_imbalance = (max(t, pop_max) - min(s, pop_min)) / self.pop_ideal * 100
                        min_imbalance = min(min_imbalance, pop_imbalance)
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