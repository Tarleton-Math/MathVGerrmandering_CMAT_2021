################# set parameters #################
district_type = 'sldu'
level         = 'cntyvtd'
proposal      = 'PLANS2101'
# proposal      = ''  # if you want to want from the 2010 districts
contract      = 10
compute_nodes = False
graph_path    = '/home/jupyter/'
proj_id       = 'cmat-315920'
dataset       = f'{proj_id}.redistricting_data'

# assumes there is a table in dataset called "nodes_TX_2020_raw"
# if proposal is not '', there must also be a table with this name
# writes graph files to the specified local directory


################# Set hashseed for reproducibility #################
HASHSEED = '0'
import os, sys
if os.getenv('PYTHONHASHSEED') != HASHSEED:
    os.environ['PYTHONHASHSEED'] = HASHSEED
    os.execv(sys.executable, [sys.executable] + sys.argv)


################# imports and references #################
import numpy as np, pandas as pd, geopandas as gpd, networkx as nx
try:
    import google.cloud.bigquery, google.cloud.bigquery_storage
except:
    os.system('pip install --upgrade google-cloud-bigquery-storage')
    import google.cloud.bigquery, google.cloud.bigquery_storage

cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient   = google.cloud.bigquery.Client(credentials=cred, project=proj)
raw_tbl    = f'{dataset}.nodes_TX_2020_raw'
stem       = f'TX_2020_{level}_{district_type}_contract{contract}'
graph_file = graph_path + f'graph_{stem}.gpickle'
adj_file   = graph_path + f'adj_{stem}.gpickle'
nodes_tbl  = f'{dataset}.nodes_{stem}'
if proposal != '':
    nodes_tbl += f'_{proposal}'
seats = f'seats_{district_type}'

################# utilities #################
def run_query(query):
    res = bqclient.query(query).result()
    try:
        return res.to_dataframe()
    except:
        return True

def get_cols(tbl):
    return [s.name for s in bqclient.get_table(tbl).schema]

def subquery(query, indents=1):
    s = '\n' + indents * '    '
    return query.strip().replace('\n', s)

def delete_table(tbl):
    query = f"drop table {tbl}"
    try:
        run_query(query)
    except google.api_core.exceptions.NotFound:
        pass

def load_table(tbl, df=None, query=None, overwrite=True, preview_rows=0):
    if overwrite:
        delete_table(tbl)
    if df is not None:
        job = bqclient.load_table_from_dataframe(df, tbl).result()
    elif query is not None:
        job = bqclient.query(query, job_config=google.cloud.bigquery.QueryJobConfig(destination=tbl)).result()
    else:
        raise Exception('at least one of df, file, or query must be specified')
    if preview_rows > 0:
        print(head(tbl, preview_rows))
    return tbl


################# get nodes #################
def get_nodes_query(show=True):
    # Builds a deeply nested SQL query to generate nodes table
    # Query is returned, but not run by this function because it takes a while
    # and I got really pissed off by accidentally running it and waiting forever.
    
    # We build the query one level of nesting at a time store the "cumulative query" at each step
    
    query = list()
    
    # Python builds the SQL query using f-strings.  If you haven't used f-string, they are f-ing amazing.
    
    # Get critical columns from nodes_raw
    # Note we keep a dedicated "cntyvtd_temp" even though typically level=cntyvtd
    # so that, when we run with level<>cntyvtd, we still have access to ctnyvtd via ctnyvtd_temp
    query.append(f"""
select
    geoid,
    {level},
    cast({district_type} as int) as district_2010,
    substring(cnty,3) as cnty,
    county,
    cntyvtd as cntyvtd_temp,
    {seats} as seats,
from
    {raw_tbl}""")


    # Joins the proposal's table is given.  Else, uses the 2010 districts.
    if proposal != '':
        proposal_tbl = f'{dataset}.{proposal}'
        cols = get_cols(proposal_tbl)
        query.append(f"""
select
    A.*,
    cast(B.{cols[1]} as int) as district,
from (
    {subquery(query[-1])}
    ) as A
inner join
    {proposal_tbl} as B
on
    A.geoid = cast(B.{cols[0]} as string)""")

    else:
        query.append(f"""
select
    A.*,
    A.district_2010 as district,
from (
    {subquery(query[-1], indents=1)}
    ) as A""")


    # Nodes_raw is at the census block level, but our MCMC usually runs at the cntyvtd level
    # So, we already need one round of contraction to combined all blocks in a cntyvtd into a single node.
    # However, we may want a second round of contraction combining all cntyvtds in a "small" county into a single node.
    # Here are several options for this second contraction, which I'll call "county contraction".
    
    # No county contraction
    if contract == 0:
        query.append(f"""
select
    geoid,
    {level} as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    {subquery(query[-1])}
    )""")

    # Contract county iff it was wholly contained in a single district in 2010
    elif contract == 2010:
        query.append(f"""
select
    geoid,
    case when ct = 1 then cnty else {level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        count(distinct district_2010) over (partition by cnty) as ct,
    from (
        {subquery(query[-1])}
        )
    )""")
        
    
    # Contract county iff it is wholly contained in a single district in the proposed plan
    elif contract == 'proposal':
        query.append(f"""
select
    geoid,
    case when ct = 1 then cnty else {level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        count(distinct district) over (partition by cnty) as ct,
    from (
        {subquery(query[-1], indents=2)}
        )
    )""")
    
    
    # Contract county iff its seats_share < contract / 10
    # seats_share = county pop / ideal district pop
    # ideal district pop = state pop / # districts
    # Note: contract = "tenths of a seat" rather than "seats" so that contract is an integer
    # Why? To avoid decimals in table & file names.  No other reason.
    else:
        query.append(f"""
select
    geoid,
    case when 10 * seats_temp < {contract} then cnty else {level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        sum(seats) over (partition by cnty) as seats_temp,
    from (
        {subquery(query[-1], indents=2)}
        )
    )""")


    # Contraction leads to ambiguities.
    # Suppose some block of a cntyvtd are in county 1 while others are in county 2.
    # Or some blocks of a contracting county are in district A while others are in district B.
    # We will chose to assigned the contracted node to the county/district/cntyvtd that contains
    # the largest population of the contracting geographic unit unit.
    # Because we need seats for other purposes AND seats is proportional to total_pop,
    # it's equivalent and more convenient to implement this using seats in leiu of total_pop.
    # We must apply this tie-breaking rule to all categorical variables.
    
    # First, find the total seats in each (geoid_new, unit) intersection
    query.append(f"""
select
    *,
    sum(seats) over (partition by geoid_new, district) as seats_district,
    sum(seats) over (partition by geoid_new, county  ) as seats_county,
    sum(seats) over (partition by geoid_new, cntyvtd ) as seats_cntyvtd,
from (
    {subquery(query[-1], indents=1)}
    )""")


    # Now, we find the max over all units in a given geoid
    query.append(f"""
select
    *,
    max(seats_district) over (partition by geoid_new) seats_district_max,
    max(seats_county  ) over (partition by geoid_new) seats_county_max,
    max(seats_cntyvtd ) over (partition by geoid_new) seats_cntyvtd_max,
from (
    {subquery(query[-1])}
    )""")
    

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
    )""")



    # Time for the big aggregration step.
    # Get names of the remaining data columns of nodes_raw
    cols = get_cols(raw_tbl)
    a = cols.index('total_pop_prop')
    b = cols.index('aland')
    # Create a list of sum statements for these columns to use in the select
    sels = ',\n    '.join([f'sum({c}) as {c}' for c in cols[a:b]])
    
    # Join nodes_raw, groupby geoid_new, and aggregate categorical variable with max, numerical variables with sum,
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
    {raw_tbl} as B
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
    )""")


    # Compute density, polsby-popper, and centroid.
    query.append(f"""
select
    *,
    case when perim > 0 then round(4 * {np.pi} * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    case when aland > 0 then total_pop / aland else 0 end as density,
    st_centroid(polygon) as point,
from (
    {subquery(query[-1])}
    )""")


    if show:
        for k, q in enumerate(query):
            print(f'\n\nquery {k}')
            print(q)
    
    return query[-1]



################# get graph#################

### utility functions ###
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
    # permutations of the nodes within a district OR
    # permutations of the district labels
    
    # WARNING - Python inserts randomness into its hash function for security reasons.
    # However, this means the same partition gets a different hash in different runs.
    # The first lines of this .py file fix this issue by setting the hashseen
    # But this solution does NOT work in a Jupyter notebook, AFAIK.
    # I have not found a way to force deterministic hashing in Jupyter.
    
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


################# Run it #################

if compute_nodes:
    nodes_query = get_nodes_query(False)
    load_table(tbl=nodes_tbl, query=nodes_query)
G, A = get_graph(nodes_tbl)
nx.write_gpickle(G, graph_file)
nx.write_gpickle(A, adj_file)