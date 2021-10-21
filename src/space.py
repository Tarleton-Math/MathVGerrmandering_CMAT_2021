from . import *
# proposal_default = 

@dataclasses.dataclass
class Space(Base):
    proposal   : str = 'planh2100'
    contract   : str = 'proposal'
    node_attr  : typing.Tuple = ()
    edge_attr  : typing.Tuple = ()

        
    def __post_init__(self):
        self.Sources = ('proposal', 'nodes', 'graph')
        super().__post_init__()
        self.district_type = self.District_types[self.proposal[4]]
        stem = f'{self.state.abbr}_{self.district_type}_{self.proposal}'
        self.dataset = f'{root_bq}.{stem}'
        bqclient.create_dataset(self.dataset , exists_ok=True)
        self.path = dict()    
        self.tbls  = dict()
        self.tbls['source'] = f'{data_bq}.{self.state.abbr}_{self.census_yr}_source_all'

        for src in self.Sources:
            self.path[src] = data_path / f'proposals/{stem.replace("_", "/")}'
            if src == 'proposal':
                s = src
                self.csv =  self.path[src] / f'{self.proposal.upper()}.csv'
            else:
                s = f'{self.level}_{self.contract}_{src}'
                if src == 'graph':
                    self.gpickle = self.path[src] / f'{stem}_{s}.gpickle'
            self.tbls[src] = f'{self.dataset}.{s}'

        for src in self.Sources:
            self.get(src)


    def get_graph(self):
        src = 'graph'
        try:
            self.graph = nx.read_gpickle(self.gpickle)
            rpt(f'using existing graph')
            return
        except:
            rpt(f'creating graph')
        # what attributes will be stored in nodes & edges
        self.node_attr = {'geoid', 'county', 'district', 'total_pop', 'seats', 'aland', 'perim'}.union(self.node_attr)
        self.edge_attr = {'distance', 'shared_perim'}.union(self.edge_attr)
        # retrieve node data
        nodes_query = f'select {", ".join(self.node_attr)} from {self.tbls["nodes"]}'
        nodes = run_query(nodes_query).set_index('geoid')

        # find eges = pairs of nodes that border each other
        edges_query = f"""
select
    *
from (
    select
        x.geoid as geoid_x,
        y.geoid as geoid_y,        
        st_distance(x.point, y.point) / {m_per_mi} as distance,
        (x.perim + y.perim - st_perimeter(st_union(x.polygon, y.polygon))/{m_per_mi}) / 2  as shared_perim
    from
        {self.tbls['nodes']} as x,
        {self.tbls['nodes']} as y
    where
        x.geoid < y.geoid
        and st_intersects(x.polygon, y.polygon)
    )
where
    shared_perim > 0.001
"""
        edges = run_query(edges_query)

        # create graph from edges and add node attributes
        self.graph = nx.from_pandas_edgelist(edges, source=f'geoid_x', target=f'geoid_y', edge_attr=tuple(self.edge_attr))
        nx.set_node_attributes(self.graph, nodes.to_dict('index'))
        districts  = set(nodes['district'])

        # Check for disconnected districts & fix
        # This is rare, but can potentially happen during county-node contraction.
        connected = False
        rng = np.random.default_rng(0)
        while not connected:
            connected = True
            for D in districts:
                comp = get_components_district(self.graph, D)
                if len(comp) > 1:
                    # district disconnected - keep largest component and "dissolve" smaller ones into other contiguous districts.
                    # May create population deviation which will be corrected during MCMC.
                    print(f'regrouping to connect components of district {D} with component {[len(c) for c in comp]}')
                    connected = False
                    for c in comp[1:]:
                        for x in c:
                            y = rng.choice(list(self.graph.neighbors(x)))  # chose a random neighbor
                            self.graph.nodes[x]['district'] = self.graph.nodes[y]['district']  # adopt its district

        # Create new districts starting at nodes with high population
        new_districts = self.seats[self.district_type] - len(districts)
        if new_districts > 0:
            new_district_starts = nodes.nlargest(10 * new_districts, 'total_pop').index.tolist()
            D_new = max(districts) + 1
            while new_districts > 0:
                # get most populous remaining node, make it a new district
                # check if this disconnected its old district.  If so, undo and try next node.
                n = new_district_starts.pop(0)
                D_old = self.graph.nodes[n]['district']
                self.graph.nodes[n]['district'] = D_new
                comp = get_components_district(self.graph, D_old)
                if len(comp) == 1:
                    # success
                    D_new += 1
                    new_districts -= 1
                else:
                    # fail - disconnected old district - undo and try again
                    self.graph.nodes[n]['district'] = D_old
        nx.write_gpickle(self.graph, self.gpickle)
            
            
    def get_proposal(self):
        src = 'proposal'
        rpt(f'creating proposal table from {self.csv}')
        df = pd.read_csv(self.csv, skiprows=1, names=('geoid', self.district_type), dtype={'geoid':str})
        load_table(self.tbls[src], df=df)
        
        
    def aggegrate(self, qry, show=False):
        data_sums = [f'sum({c}) as {c}' for c in self.data_cols]
####### Builds a deeply nested SQL query to generate nodes
####### We build the query one level of nesting at a time store the "cumulative query" at each step
####### Python builds the SQL query using f-strings.  If you haven't used f-string, they are f-ing amazing.
####### Note we keep a dedicated "cntyvtd_temp" even though typically level = cntyvtd
####### so that, when we run with level != cntyvtd, we still have access to ctnyvtd via ctnyvtd_temp
        
####### Contraction can cause ambiguities.
####### Suppose some block of a cntyvtd are in county 1 while others are in county 2.
####### Or some blocks of a contracting county are in district A while others are in district B.
####### We will assign the contracted node to the county/district/cntyvtd that contains the largest population.
####### But because we need seats for other purposes AND seats is proportional to total_pop,
####### it's more convenient to implement this using seats in leiu of total_pop.
####### We must apply this tie-breaking rule to all categorical variables.
####### First, find the total seats in each (geoid_new, unit) intersection

        query = qry.copy()
        query.append(f"""
select
    *,
    sum(seats) over (partition by geoid_new, district) as seats_district,
    sum(seats) over (partition by geoid_new, county  ) as seats_county,
    sum(seats) over (partition by geoid_new, cntyvtd ) as seats_cntyvtd,
from (
    {subquery(query[-1])}
    )
""")
####### Now, we find the max over all units in a given geoid ###########
        query.append(f"""
select
    *,
    max(seats_district) over (partition by geoid_new) seats_district_max,
    max(seats_county  ) over (partition by geoid_new) seats_county_max,
    max(seats_cntyvtd ) over (partition by geoid_new) seats_cntyvtd_max,
from (
    {subquery(query[-1])}
    )
""")
####### Now, we create temporary columns that are null except on the rows of the unit achieving the max value found above
####### When we do the "big aggegration" below, max() will grab the name of the correct unit (one with max seat)
        query.append(f"""
select
    *,
    case when seats_district = seats_district_max then district else null end as district_new,
    case when seats_county   = seats_county_max   then county   else null end as county_new,
    case when seats_cntyvtd  = seats_cntyvtd_max  then cntyvtd  else null end as cntyvtd_new,
from (
    {subquery(query[-1])}
    )
""")
####### Time for the big aggregration step.
####### Join source, groupby geoid_new, and aggregate categorical variable with max, numerical variables with sum,
####### and geospatial polygon with st_union_agg.
        query.append(f"""
select
    geoid_new as geoid,
    max(district_new) as district,
    max(county_new  ) as county,
    max(cntyvtd_new ) as cntyvtd,
    sum(seats       ) as seats,
    {join_str().join(data_sums)},
    st_union_agg(polygon_simp) as polygon,
    sum(aland) as aland,
from (
    {subquery(query[-1])}
    )
group by
    geoid_new
""")
####### Get polygon perimeter #######
        query.append(f"""
select
    *,
    st_perimeter(polygon) / {m_per_mi} as perim,
from (
    {subquery(query[-1])}
    )
""")
####### Compute density, polsby-popper, and centroid #######
        query.append(f"""
select
    *,
    case when perim > 0 then round(4 * {np.pi} * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    case when aland > 0 then total_pop / aland else 0 end as density,
    st_centroid(polygon) as point,
from (
    {subquery(query[-1])}
    )
""")
        if show:
            for k, q in enumerate(query):
                print(f'\n=====================================================================================\nstage {k}')
                print(q)
        return query

        
        
    def get_nodes(self, show=False):
        src = 'nodes'
        cols = get_cols(self.tbls['source'])
        a = cols.index('total_pop')
        b = cols.index('polygon')
        self.data_cols = cols[a:b]
        query = list()
        query.append(f"""
select
    *,
    cntyvtd as cntyvtd_temp,
    seats_{self.district_type} as seats,
from
    {self.tbls['source']}
""")
        
        query.append(f"""
select
    A.*,
    B.{self.district_type} as district,
from (
    {subquery(query[-1])}
    ) as A
inner join
    {self.tbls['proposal']} as B
on
    A.geoid = B.geoid
""")


        tbl_data = self.tbls['proposal'] + '_data'
        if check_table(tbl_data):
            rpt('using existing data table')
        else:
            rpt('creating data table')
            query.append(f"""
select
    geoid,
    district,
    county,
    cntyvtd,
    seats,
    {join_str().join(self.data_cols)},
from (
    {subquery(query[-1])}
    )
""")
            load_table(tbl_data, query=query[-1])
            query.pop(-1)

        
        tbl_districts = self.tbls['proposal'] + '_districts'
        if check_table(tbl_districts):
            rpt('using existing districts table')
        else:
            rpt('creating districts table')
            query.append(f"""
select
    *,
    district as geoid_new,
from (
    {subquery(query[-1])}
    )
""")
            load_table(tbl_districts, query=self.aggegrate(query)[-1])
            query.pop(-1)


        if check_table(self.tbls[src]):
            rpt('using nodes table')
        else:
            rpt('creating nodes table')

####### No county contraction #######
            if str(self.contract) == '0':
                query.append(f"""
select
    *,
    {self.level} as geoid_new,
from (
    {subquery(query[-1])}
    )
""")


####### Contract county iff it is wholly contained in a single district in the proposed plan #######
            elif self.contract == 'proposal':
                query.append(f"""
select
    *, 
    case when count(distinct district) over (partition by cnty) = 1 then cnty else {self.level} end as geoid_new,
from (
    {subquery(query[-1])}
    )
""")
####### Contract county iff its seats_share < contract / 10 #######
####### seats_share = county pop / ideal district pop #######
####### ideal district pop = state pop / # districts #######
####### Note: contract = "tenths of a seat" rather than "seats" so that contract is an integer #######
####### Why? To avoid decimals in table & file names.  No other reason. #######
            else:
                try:
                    c = float(self.contract) / 10
                except:
                    raise Exception(f'contract must be "proposal" or "{proposal_default}" or an integer >= 0 ... got {self.contract}')
                query.append(f"""
select
    *,
    case when sum(seats) over (partition by cnty) < {c} then cnty else {self.level} end as geoid_new,
from (
    {subquery(query[-1])}
    )
""")
            load_table(self.tbls[src], query=self.aggegrate(query)[-1])