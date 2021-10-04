from . import *
import networkx as nx
proposal_default = 'enacted2010'

@dataclasses.dataclass
class MCMC(Base):
    proposal: str = proposal_default
    contract: str = '0'
        
    def __post_init__(self):
        self.Sources = ('proposal', 'nodes', 'graph', 'results')
        super().__post_init__()
        self.refresh_all.discard('proposal')
        self.seats_col = f'seats_{self.district_type}'

        self.tbl  = dict()
        self.pq   = dict()
        self.path = dict()
        stem = f'{self.state.abbr}_{self.census_yr}'
        self.tbl['source'] = f'{data_bq}.{stem}_source_all'
        
        stem += f'_{self.district_type}_{self.proposal}'
        for src in self.Sources:
            s = stem
            if src != 'proposal':
                s += f'_{self.level}_{self.contract}'
            
            self.tbl [src] = f'{data_bq}.{s}_{src}'
            self.pq  [src] = data_path/f'{stem}/{s}_{src}.parquet'
            self.path[src] = self.pq[src].parent

        for src in ('proposal', 'nodes'):
            rpt(f'Get {src}'.ljust(rpt_just, ' '))
            self.delete_for_refresh(src)
            self[f'get_{src}']()
            print(f'success!')
            os.chdir(code_path)

        
    def get_proposal(self):
        src = self.proposal
        if self.proposal != proposal_default:
            if check_table(self.tbl['proposal']):
                rpt('using existing proposal table')
            else:
                csv = data_path / f'proposals/{self.district_type}/{self.proposal.upper()}.csv'
                rpt(f'creating proposal table from {csv}')
                df = pd.read_csv(csv, skiprows=1, names=('geoid', self.district_type), dtype={'geoid':str})
                load_table(self.tbl['proposal'], df=df)
        
        
    def get_nodes(self, show=False):
        # Builds a deeply nested SQL query to generate nodes
        # We build the query one level of nesting at a time store the "cumulative query" at each step
        # Python builds the SQL query using f-strings.  If you haven't used f-string, they are f-ing amazing.
        # Get critical columns from source table
        # Note we keep a dedicated "cntyvtd_temp" even though typically level = cntyvtd
        # so that, when we run with level != cntyvtd, we still have access to ctnyvtd via ctnyvtd_temp
        query = list()
        query.append(f"""
select
    geoid,
    {self.level},
    cast({self.district_type} as int) as district_2010,
    substring(cnty,3) as cnty,
    county,
    cntyvtd as cntyvtd_temp,
    {self.seats_col} as seats,
from
    {self.tbl['source']}
""")


        # join the proposal table if specificied; else use currently enacted plan
        if self.proposal == proposal_default:
            query.append(f"""
select
    A.*,
    A.district_2010 as district,
from (
    {subquery(query[-1])}
    ) as A
""")
    
        else:
            cols = get_cols(self.tbl['proposal'])
            query.append(f"""
select
    A.*,
    cast(B.{cols[1]} as int) as district,
from (
    {subquery(query[-1])}
    ) as A
inner join
    {self.tbl['proposal']} as B
on
    A.geoid = cast(B.{cols[0]} as string)
""")

        # source data is at the census block level, but our MCMC usually runs at the cntyvtd level
        # So, we already need one round of contraction to combined all blocks in a cntyvtd into a single node.
        # However, we may want a second round of contraction combining all cntyvtds in a "small" county into a single node.
        # Here are several options for this second contraction, which I'll call "county contraction".
    
        # No county contraction
        if str(self.contract) == '0':
            query.append(f"""
select
    geoid,
    {self.level} as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    {subquery(query[-1])}
    )
""")

        # Contract county iff it was wholly contained in a single district in 2010
        elif self.contract == proposal_default:
            query.append(f"""
select
    geoid,
    case when ct = 1 then cnty else {self.level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {self.level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        count(distinct district_2010) over (partition by cnty) as ct,
    from (
        {subquery(query[-1], indents=2)}
        )
    )
""")
        
    
        # Contract county iff it is wholly contained in a single district in the proposed plan
        elif self.contract == 'proposal':
            query.append(f"""
select
    geoid,
    case when ct = 1 then cnty else {self.level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {self.level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        count(distinct district) over (partition by cnty) as ct,
    from (
        {subquery(query[-1], indents=2)}
        )
    )
""")
    
    
        # Contract county iff its seats_share < contract / 10
        # seats_share = county pop / ideal district pop
        # ideal district pop = state pop / # districts
        # Note: contract = "tenths of a seat" rather than "seats" so that contract is an integer
        # Why? To avoid decimals in table & file names.  No other reason.
        else:
            try:
                c = int(self.contract) / 10
            except:
                raise Exception(f'contract must be "proposal" or "{proposal_default}" or an integer >= 0 ... got {self.contract}')
            query.append(f"""
select
    geoid,
    case when seats_temp < {c} then cnty else {self.level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {self.level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        sum(seats) over (partition by cnty) as seats_temp,
    from (
        {subquery(query[-1], indents=2)}
        )
    )
""")


        # Contraction can cause ambiguities.
        # Suppose some block of a cntyvtd are in county 1 while others are in county 2.
        # Or some blocks of a contracting county are in district A while others are in district B.
        # We will assign the contracted node to the county/district/cntyvtd that contains the largest population.
        # But because we need seats for other purposes AND seats is proportional to total_pop,
        # it's more convenient to implement this using seats in leiu of total_pop.
        # We must apply this tie-breaking rule to all categorical variables.

        # First, find the total seats in each (geoid_new, unit) intersection
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


        # Now, we find the max over all units in a given geoid
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
    

        # Now, we create temporary columns that are null except on the rows of the unit achieving the max value found above
        # When we do the "big aggegration" below, max() will grab the name of the correct unit (one with max seat)
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


        # Time for the big aggregration step.
        # Get names of the remaining data columns of source
        cols = get_cols(self.tbl['source'])
        a = cols.index('total_pop_prop')
        b = cols.index('aland')
        # Create a list of sum statements for these columns to use in the select
        sels = ',\n    '.join([f'sum({c}) as {c}' for c in cols[a:b]])

        # Join source, groupby geoid_new, and aggregate categorical variable with max, numerical variables with sum,
        # and geospatial polygon with st_union_agg.
        query.append(f"""
select
    A.geoid_new as geoid,
    max(district_new) as district,
    max(county_new  ) as county,
    max(cntyvtd_new ) as cntyvtd,
    {sels},
    st_union_agg(polygon) as polygon,
    sum(aland) as aland
from (
    {subquery(query[-1])}
    ) as A
inner join
    {self.tbl['source']} as B
on
    A.geoid = B.geoid
group by
    geoid_new
""")


        # Get polygon perimeter
        query.append(f"""
select
    *,
    st_perimeter(polygon) as perim,
from (
    {subquery(query[-1])}
    )
""")


        # Compute density, polsby-popper, and centroid.
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
    
        load_table(self.tbl['nodes'], query=query[-1])
        
################# create graph #################

    def get_components(self, G=None):
        # get and sorted connected components by size
        if G is None:
            G = self.graph
        return sorted([tuple(x) for x in nx.connected_components(G)], key=lambda x:len(x), reverse=True)

    def district_view(self, D, G=None):
        # get subgraph of a given district
        if G is None:
            G = self.graph
        return nx.subgraph_view(G, lambda n: G.nodes[n]['district'] == D)

    def get_components_district(self, D, G=None):
        # get connected components of a district
        if G is None:
            G = self.graph
        return get_components(district_view(D, G))

    def get_hash(self, G=None):
        # Partition hashing provides a unique integer label for each distinct plan
        # For each district, get sorted tuple of nodes it contains.  Then sort this tuple of tuples.
        # Produces a sorted tuple of sorted tuples called "partition" that does not care about:
        # permutations of the nodes within a district OR
        # permutations of the district labels

        # WARNING - Python inserts randomness into its hash function for security reasons.
        # However, this means the same partition gets a different hash in different runs.
        # The first lines of this .py file fix this issue by setting the hashseen
        # But this solution does NOT work in a Jupyter notebook, AFAIK.
        # I have not found a way to force deterministic hashing in Jupyter.

        if G is None:
            G = self.graph
        districts = set(d for n, d in G.nodes(data='district'))
        partition = tuple(sorted(tuple(sorted(district_view(G, D).nodes)) for D in districts))
        return partition.__hash__()



    def get_graph(nodes_tbl, new_districts=0, node_attr=(), edge_attr=(), random_seed=0):
        # what attributes will be stored in nodes & edges
        node_attr = {'geoid', 'county', 'district', 'total_pop', seats, 'aland', 'perim'}.union(node_attr)
        edge_attr = {'distance', 'shared_perim'}.union(edge_attr)
        # retrieve node data
        nodes_query = f'select {", ".join(node_attr)} from {nodes_tbl}'
        nodes = run_query(nodes_query).set_index('geoid')

        # get unique districts & counties
        districts = set(nodes['district'])
        counties  = set(nodes['county'  ])

        # set random number generator for reproducibility
        rng = np.random.default_rng(random_seed)

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
        {nodes_tbl} as x,
        {nodes_tbl} as y
    where
        x.geoid < y.geoid
        and st_intersects(x.polygon, y.polygon)
    )
where
    shared_perim > 0.01
"""
        edges = run_query(edges_query)

        # create graph from edges and add node attributes
        G = nx.from_pandas_edgelist(edges, source=f'geoid_x', target=f'geoid_y', edge_attr=tuple(edge_attr))
        nx.set_node_attributes(G, nodes.to_dict('index'))


        # Check for disconnected districts & fix
        # This is rare, but can potentially happen during county-node contraction.
        connected = False
        while not connected:
            connected = True
            for D in districts:
                comp = get_components_district(G, D)
                if len(comp) > 1:
                    # district disconnected - keep largest component and "dissolve" smaller ones into other contiguous districts.
                    # May create population deviation which will be corrected during MCMC.
                    print(f'regrouping to connect components of district {D} with component {[len(c) for c in comp]}')
                    connected = False
                    for c in comp[1:]:
                        for x in c:
                            y = rng.choice(list(G.neighbors(x)))  # chose a random neighbor
                            G.nodes[x]['district'] = G.nodes[y]['district']  # adopt its district

        # Create new districts starting at nodes with high population
        new_district_starts = nodes.nlargest(10 * new_districts, 'total_pop').index.tolist()
        D_new = max(districts) + 1
        while new_districts > 0:
            # get most populous remaining node, make it a new district
            # check if this disconnected its old district.  If so, undo and try next node.
            n = new_district_starts.pop(0)
            D_old = G.nodes[n]['district']
            G.nodes[n]['district'] = D_new
            comp = get_components_district(G, D_old)
            if len(comp) == 1:
                # success
                D_new += 1
                new_districts -= 1
            else:
                # fail - disconnected old district - undo and try again
                G.nodes[n]['district'] = D_old


        # Create the county-district bi-partite adjacency graph.
        # This graph has 1 node for each county and district &
        # an edge for all (county, district) that intersect (share land).
        # It is an efficient tool to track map defect and other properties.
        A = nx.Graph()
        for n, data in G.nodes(data=True):
            D = data['district']
            A.add_node(D)  # adds district node if not already present
            A.nodes[D]['polsby_popper'] = 0
            for k in ['total_pop', 'aland', 'perim']:
                try:
                    A.nodes[D][k] += data[k]  # add to attribute if exists
                except:
                    A.nodes[D][k] = data[k]  # else create attribute

            C = data['county']
            A.add_node(C)  # adds county node if not already present
            for k in ['total_pop', seats]:
                try:
                    A.nodes[C][k] += data[k]  # add to attribute if exists
                except:
                    A.nodes[C][k] = data[k]  # else create attribute

            A.add_edge(C, D)  # create edge

        # get defect targets
        for C in counties:
            A.nodes[C]['whole_target']     = int(np.floor(A.nodes[C][seats]))
            A.nodes[C]['intersect_target'] = int(np.ceil (A.nodes[C][seats]))
        return G, A