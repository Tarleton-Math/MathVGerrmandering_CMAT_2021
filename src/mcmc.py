from . import *
import networkx as nx
proposal_default = 'plansenacted2010'

################# graph utilities #################

def get_components(G):
    # get and sorted connected components by size
    return sorted([tuple(x) for x in nx.connected_components(G)], key=lambda x:len(x), reverse=True)

def district_view(G, D):
    # get subgraph of a given district
    return nx.subgraph_view(G, lambda n: G.nodes[n]['district'] == D)

def get_components_district(G, D):
    # get connected components of a district
    return get_components(district_view(G, D))

def get_hash(G):
    # Partition hashing provides a unique integer label for each distinct plan
    # For each district, get sorted tuple of nodes it contains.  Then sort this tuple of tuples.
    # Produces a sorted tuple of sorted tuples called "partition" that does not care about:
    # permutations of the nodes within a district OR permutations of the district labels.
    # WARNING - Python inserts randomness into its hash function for security reasons.
    # However, this means the same partition gets a different hash in different runs.
    # The first lines of this .py file fix this issue by setting the hashseen
    # But this solution does NOT work in a Jupyter notebook, AFAIK.
    # I have not found a way to force deterministic hashing in Jupyter.
    districts = set(d for n, d in G.nodes(data='district'))
    partition = tuple(sorted(tuple(sorted(district_view(G, D).nodes)) for D in districts))
    return partition.__hash__()


@dataclasses.dataclass
class MCMC(Base):
    proposal   : str = proposal_default
    contract   : str = '0'
    node_attr  : typing.Tuple = ()
    edge_attr  : typing.Tuple = ()
    random_seed: int = 0
        
    def __post_init__(self):
        self.Sources = ('proposals', 'nodes', 'graphs', 'adjs')
        
        super().__post_init__()
        self.refresh_all.discard(self.proposal)
        self.random_seed = int(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)

        self.tbl  = dict()
        self.gp   = dict()
        self.csv  = dict()
        self.path = dict()
        self.tbl['source'] = f'{data_bq}.{self.state.abbr}_{self.census_yr}_source_all'
        for src in self.Sources:
            stem = f'{self.state.abbr}_{self.census_yr}_{self.district_type}_{self.proposal}'
            if src == 'proposals':
                self.csv[src] = data_path / f'{src}/{self.state.abbr}/{self.district_type}/{self.proposal.upper()}.csv'
            else:
                stem += f'_{self.level}_{self.contract}_{src}'
                if src != 'nodes':
                    stem += f'_{self.random_seed}'
            self.tbl [src] = f'{data_bq}.{stem}'
            self.gp  [src] = data_path / f'{src}/{self.state.abbr}/{self.district_type}/{stem}.gpickle'
            self.path[src] = self.gp[src].parent

        for src in self.Sources:
            if src in ('proposals', 'nodes'):
                self.get(src)


    def get_graphs(self):
        src = 'graphs'
        try:
            self.graph = nx.read_gpickle(self.gp[src])
            rpt(f'using existing graph')
            return
        except:
            rpt(f'creating graph')
        # what attributes will be stored in nodes & edges
        self.node_attr = {'geoid', 'county', 'district', 'total_pop', 'seats', 'aland', 'perim'}.union(self.node_attr)
        self.edge_attr = {'distance', 'shared_perim'}.union(self.edge_attr)
        # retrieve node data
        nodes_query = f'select {", ".join(self.node_attr)} from {self.tbl["nodes"]}'
        nodes = run_query(nodes_query).set_index('geoid')

        # get unique districts & counties
        self.districts = set(nodes['district'])
        self.counties  = set(nodes['county'  ])

        # find eges = pairs of nodes that border each other
        edges_query = f"""
select
    *
from (
    select
        x.geoid as geoid_x,
        y.geoid as geoid_y,        
        st_distance(x.point, y.point) as distance,
        st_perimeter(st_intersection(x.polygon, y.polygon)) as shared_perim
    from
        {self.tbl['nodes']} as x,
        {self.tbl['nodes']} as y
    where
        x.geoid < y.geoid
        and st_intersects(x.polygon, y.polygon)
    )
where
    shared_perim > 0.01
"""
        edges = run_query(edges_query)

        # create graph from edges and add node attributes
        self.graph = nx.from_pandas_edgelist(edges, source=f'geoid_x', target=f'geoid_y', edge_attr=tuple(self.edge_attr))
        nx.set_node_attributes(self.graph, nodes.to_dict('index'))

        # Check for disconnected districts & fix
        # This is rare, but can potentially happen during county-node contraction.
        connected = False
        while not connected:
            connected = True
            for D in self.districts:
                comp = get_components_district(self.graph, D)
                if len(comp) > 1:
                    # district disconnected - keep largest component and "dissolve" smaller ones into other contiguous districts.
                    # May create population deviation which will be corrected during MCMC.
                    print(f'regrouping to connect components of district {D} with component {[len(c) for c in comp]}')
                    connected = False
                    for c in comp[1:]:
                        for x in c:
                            y = self.rng.choice(list(self.graph.neighbors(x)))  # chose a random neighbor
                            self.graph.nodes[x]['district'] = self.graph.nodes[y]['district']  # adopt its district

        # Create new districts starting at nodes with high population
        new_districts = self.seats[self.district_type] - len(self.districts)
        if new_districts > 0:
            new_district_starts = nodes.nlargest(10 * new_districts, 'total_pop').index.tolist()
            D_new = max(self.districts) + 1
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
        nx.write_gpickle(self.graph, self.gp[src])


    def get_adjs(self):
        src = 'adjs'
        try:
            self.adj = nx.read_gpickle(self.gp[src])
            rpt(f'using existing adj')
            return
        except:
            rpt(f'creating adj')

        # Create the county-district bi-partite adjacency graph.
        # This graph has 1 node for each county and district &
        # an edge for all (county, district) that intersect (share land).
        # It is an efficient tool to track map defect and other properties.
        self.adj = nx.Graph()
        for n, data in self.graph.nodes(data=True):
            D = data['district']
            self.adj.add_node(D)  # adds district node if not already present
            self.adj.nodes[D]['polsby_popper'] = 0
            for k in ['total_pop', 'aland', 'perim']:
                try:
                    self.adj.nodes[D][k] += data[k]  # add to attribute if exists
                except:
                    self.adj.nodes[D][k] = data[k]  # else create attribute

            C = data['county']
            self.adj.add_node(C)  # adds county node if not already present
            for k in ['total_pop', 'seats']:
                try:
                    self.adj.nodes[C][k] += data[k]  # add to attribute if exists
                except:
                    self.adj.nodes[C][k] = data[k]  # else create attribute

            self.adj.add_edge(C, D)  # create edge

        # get defect targets
        for C in self.counties:
            self.adj.nodes[C]['whole_target']     = int(np.floor(self.adj.nodes[C]['seats']))
            self.adj.nodes[C]['intersect_target'] = int(np.ceil (self.adj.nodes[C]['seats']))
        nx.write_gpickle(self.adj, self.gp[src])
            
            
    def get_proposals(self):
        src = 'proposals'
        if self.proposal != proposal_default:
            rpt(f'creating proposal table from {self.csv[src]}')
            df = pd.read_csv(self.csv[src], skiprows=1, names=('geoid', self.district_type), dtype={'geoid':str})
            load_table(self.tbl['proposals'], df=df)
        
        
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
    st_perimeter(polygon) as perim,
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
        cols = get_cols(self.tbl['source'])
        a = cols.index('total_pop')
        b = cols.index('aland')
        self.data_cols = cols[a:b]
        query = list()
        query.append(f"""
select
    *,
    cntyvtd as cntyvtd_temp,
    seats_{self.district_type} as seats,
from
    {self.tbl['source']}
""")
        
        if self.proposal == proposal_default:
            query.append(f"""
select
    A.*,
    A.{self.district_type} as district,
from (
    {subquery(query[-1])}
    ) as A
""")
        else:
            query.append(f"""
select
    A.*,
    B.{self.district_type} as district,
from (
    {subquery(query[-1])}
    ) as A
inner join
    {self.tbl['proposals']} as B
on
    A.geoid = B.geoid
""")


        tbl_data = self.tbl['proposals'] + '_all'
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

        
        tbl_districts = self.tbl['proposals'] + '_districts'
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


        if check_table(self.tbl[src]):
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
####### Contract county iff it was wholly contained in a single district in 2010 #######
            elif self.contract == proposal_default:
                query.append(f"""
select
    *, 
    case when count(distinct {self.district_type}) over (partition by cnty) = 1 then cnty else {self.level} end as geoid_new,
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
                    c = int(self.contract) / 10
                except:
                    raise Exception(f'contract must be "proposal" or "{proposal_default}" or an integer >= 0 ... got {self.contract}')
                query.append(f"""
select
    geoid,
    case when sum(seats) over (partition by cnty) < {c} then cnty else {self.level} end as geoid_new,
from (
    {subquery(query[-1])}
    )
""")
            load_table(self.tbl[src], query=self.aggegrate(query)[-1])
#             query.pop(-1)
            
            
            
            
            


                
        
#             query.append(f"""
# select
#     geoid,
#     district,
#     {self.level},
#     cntyvtd as cntyvtd_temp,
#     substring(cnty, 3) as cnty,
#     county,
#     seats_{self.district_type} as seats,
#     {join_str.join(data_cols)}
# from (
#     {subquery(query[-1])}
#     )
# """)
            
            
#             plan_query = f"""
# select

            
            
# """
            
#             load_table(tbl_raw, query=query)

        

#         # join the proposal table if specificied; else use currently enacted plan
#         if self.proposal == proposal_default:
#             query.append(f"""
# select
#     A.*,
#     A.district_2010 as district,
# from (
#     {subquery(query[-1])}
#     ) as A
# """)
    
#         else:
#             cols = get_cols(self.tbl['proposals'])
#             query.append(f"""
# select
#     A.*,
#     cast(B.{cols[1]} as int) as district,
# from (
#     {subquery(query[-1])}
#     ) as A
# inner join
#     {self.tbl['proposals']} as B
# on
#     A.geoid = cast(B.{cols[0]} as string)
# """)

        # source data is at the census block level, but our MCMC usually runs at the cntyvtd level
        # So, we already need one round of contraction to combined all blocks in a cntyvtd into a single node.
        # However, we may want a second round of contraction combining all cntyvtds in a "small" county into a single node.
        # Here are several options for this second contraction, which I'll call "county contraction".
    
        # No county contraction
#         rpt(f'creating table')
#         if str(self.contract) == '0':
#             query.append(f"""
# select
#     geoid,
#     {self.level} as geoid_new,
#     district,
#     county,
#     cntyvtd_temp as cntyvtd,
#     seats,
# from (
#     {subquery(query[-1])}
#     )
# """)

#         # Contract county iff it was wholly contained in a single district in 2010
#         elif self.contract == proposal_default:
#             query.append(f"""
# select
#     geoid,
#     case when ct = 1 then cnty else {self.level} end as geoid_new,
#     district,
#     county,
#     cntyvtd_temp as cntyvtd,
#     seats,
# from (
#     select
#         geoid,
#         {self.level},
#         district,
#         cnty,
#         county,
#         cntyvtd_temp,
#         seats,
#         count(distinct district_2010) over (partition by cnty) as ct,
#     from (
#         {subquery(query[-1], indents=2)}
#         )
# """)
        
    
#         # Contract county iff it is wholly contained in a single district in the proposed plan
#         elif self.contract == 'proposal':
#             query.append(f"""
# select
#     geoid,
#     case when ct = 1 then cnty else {self.level} end as geoid_new,
#     district,
#     county,
#     cntyvtd_temp as cntyvtd,
#     seats,
# from (
#     select
#         geoid,
#         {self.level},
#         district,
#         cnty,
#         county,
#         cntyvtd_temp,
#         seats,
#         count(distinct district) over (partition by cnty) as ct,
#     from (
#         {subquery(query[-1], indents=2)}
#         )
# """)
    
    
#         # Contract county iff its seats_share < contract / 10
#         # seats_share = county pop / ideal district pop
#         # ideal district pop = state pop / # districts
#         # Note: contract = "tenths of a seat" rather than "seats" so that contract is an integer
#         # Why? To avoid decimals in table & file names.  No other reason.
#         else:
#             try:
#                 c = int(self.contract) / 10
#             except:
#                 raise Exception(f'contract must be "proposal" or "{proposal_default}" or an integer >= 0 ... got {self.contract}')
#             query.append(f"""
# select
#     geoid,
#     case when seats_temp < {c} then cnty else {self.level} end as geoid_new,
#     district,
#     county,
#     cntyvtd_temp as cntyvtd,
#     seats,
# from (
#     select
#         geoid,
#         {self.level},
#         district,
#         cnty,
#         county,
#         cntyvtd_temp,
#         seats,
#         sum(seats) over (partition by cnty) as seats_temp,
#     from (
#         {subquery(query[-1], indents=2)}
#         )
# """)


