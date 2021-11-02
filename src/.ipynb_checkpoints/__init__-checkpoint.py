proj_id   = 'cmat-315920'
root_path = '/home/jupyter/'
gcs_path  = 'math_for_unbiased_maps_tx'

import os, pathlib, shutil, time, datetime, dataclasses, typing, google.cloud.bigquery
import numpy as np, pandas as pd, geopandas as gpd, networkx as nx

try:
    import google.cloud.bigquery_storage
except:
    os.system('pip install --upgrade google-cloud-bigquery-storage')
    import google.cloud.bigquery_storage

import warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
warnings.filterwarnings('ignore', message='.*Pyarrow could not determine the type of columns*')

pd.set_option('display.max_columns', None)
cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient   = google.cloud.bigquery.Client(credentials=cred, project=proj)
gcsclient  = google.cloud.storage .Client(credentials=cred, project=proj)

root_path    = pathlib.Path(root_path)
root_bq      = proj_id
data_bq      = root_bq   +'.redistricting_data'
data_path    = root_path / 'redistricting_data'
code_path    = root_path / 'MathVGerrmandering_CMAT_2021'

# https://gis.stackexchange.com/questions/27702/what-is-the-srid-of-census-gov-shapefiles
crs_census = 'EPSG:4269'
crs_area   = 'ESRI:102003'
crs_length = 'ESRI:102005'
m_per_mi = 1609.344
concat_str = ' ... '
rpt_just   = 15

def listify(x=None):
    if x is None:
        return []
    elif isinstance(x, pd.core.frame.DataFrame):
        x = x.to_dict('split')['data']
    elif isinstance(x, (np.ndarray, pd.Series)):
        x = x.tolist()
    elif isinstance(x, (list, tuple, set)):
        return list(x)
    else:
        return [x]

def setify(x=None):
    return set(listify(x))

def default_factory(x=None):
    return dataclasses.field(default_factory=lambda:x)

def default_set(x=None):
    return default_factory(setify(x))

@dataclasses.dataclass
class Base():
    abbr          : str = 'TX'
    shapes_yr     : int = 2020
    census_yr     : int = 2020
    level         : str = 'cntyvtd'
    Seats         : typing.Dict  = default_factory({'cd':38, 'sldu':31, 'sldl':150})
    refresh_tbl   : typing.Set = default_set()
    refresh_all   : typing.Set = default_set()
    sources       : typing.Tuple = ()
    Years         : typing.Tuple = (2020, 2010)
    Levels        : typing.Tuple = ('tabblock', 'bg', 'tract', 'cntyvtd', 'cnty')
    District_types: typing.Dict  = default_factory({'c':'cd', 's':'sldu', 'h':'sldl'})

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        return setattr(self, key, val)

    def __post_init__(self):
        pass

    def check_inputs(self):
        self.refresh_all = setify(self.refresh_all)
        self.refresh_tbl = setify(self.refresh_tbl).union(setify(self.refresh_all))
        self.state = states[states['abbr']==self.abbr].iloc[0]
        D = {'census_yr'    :self.Years,
             'shapes_yr'    :self.Years,
             'level'        :self.Levels,
             'district_type':tuple(self.District_types.values()),
             # 'refresh_all'  :self.sources,
             # 'refresh_tbl'  :self.sources,
            }
        for key, vals in D.items():
            if hasattr(self, key):
                d = setify(self[key]).difference(vals)
                if len(d) > 0:
                    raise Exception(f'got unknown values {d} for {key} ... must be in {vals}')
    
    def delete_for_refresh(self, src):
        tbl = self.tbls[src]
        if src in self.refresh_all:
            shutil.rmtree(self.path[src], ignore_errors=True)
            dataset = tbl[:tbl.rfind('.')]
            for t in bqclient.list_tables(dataset):
                nm = t.full_table_id.replace(':', '.')
                if tbl in nm:
                    delete_table(nm)
        if src in self.refresh_tbl:
            if check_table(tbl):
                rpt(f'deleting {tbl}')
                delete_table(tbl)

    def get(self, src):
        rpt(f'Get {src}'.ljust(rpt_just, ' '))
        self.delete_for_refresh(src)
        if check_table(self.tbls[src]):
            rpt('using existing table')
        else:
            rpt('processing')
            self[f'get_{src}']()
        print(f'success!')
        os.chdir(code_path)


def rpt(msg):
    print(msg, end=concat_str, flush=True)

def lower_cols(df):
    df.rename(columns = {x:str(x).lower() for x in df.columns}, inplace=True)
    return df

def lower(df):
    if isinstance(df, pd.Series):
        try:
            return df.str.lower()
        except:
            return df
    elif isinstance(df, pd.DataFrame):
        lower_cols(df)
        return df.apply(lower)
    else:
        return df

def time_formatter(t):
    h, m = divmod(t, 3600)
    m, s = divmod(m, 60)
    return f'{int(h)}hrs {int(m)}min {s:.2f}sec'
    
def extract_file(zipfile, fn, **kwargs):
    file = zipfile.extract(fn)
    return lower_cols(pd.read_csv(file, dtype=str, **kwargs))

def join_str(indents=1):
    tab = '    '
    return ',\n' + tab * indents

def subquery(query, indents=1):
    s = join_str(indents)[1:]
    return query.strip().replace('\n', s)

def check_table(tbl):
    try:
        bqclient.get_table(tbl)
        return True
    except:
        return False

def get_cols(tbl):
    """Get list of columns on tbl"""
    t = bqclient.get_table(tbl)
    return [s.name for s in t.schema]
    
def run_query(query):
    res = bqclient.query(query).result()
    try:
        return res.to_dataframe()
    except:
        return True

def delete_table(tbl):
    query = f"drop table {tbl}"
    try:
        run_query(query)
    except google.api_core.exceptions.NotFound:
        pass

def read_table(tbl, rows=99999999999, start=0, cols='*'):
    query = f'select {", ".join(cols)} from {tbl} limit {rows}'
    if start is not None:
        query += f' offset {start}'
    return run_query(query)

def load_table(tbl, df=None, query=None, file=None, overwrite=True, preview_rows=0):
    """Load data into tbl either from a pandas dataframe, sql query, or local csv file"""
    
    if overwrite:
        delete_table(tbl)
    
    if df is not None:
        job = bqclient.load_table_from_dataframe(df, tbl).result()
    elif query is not None:
        job = bqclient.query(query, job_config=google.cloud.bigquery.QueryJobConfig(destination=tbl)).result()
    elif file is not None:
        with open(file, mode='rb') as f:
            job = bqclient.load_table_from_file(f, tbl, job_config=google.cloud.bigquery.LoadJobConfig(autodetect=True)).result()
    else:
        raise Exception('at least one of df, query, or file must be specified')
    
    if preview_rows > 0:
        head(tbl, preview_rows)
    return tbl

################# graph utilities #################
def dict_to_df(D):
    return pd.DataFrame.from_dict(D, orient='index').sort_index()

def graph_to_df(G, index_name='geoid', index_position=2, attr=None):
    if attr is None:
        D = {n: d for n, d in G.nodes(data=True)}
    else:
        D = {n: {x: d[x] for x in attr if x in d.keys()} for n, d in G.nodes(data=True)}
    df = dict_to_df(D)
    df.insert(index_position, index_name, df.index)
    return df.reset_index(drop=True)

def sorter(L, sort=True):
    if sort:
        L = tuple(sorted((tuple(sorted(x)) for x in L), key=lambda x: (len(x), x), reverse=True))
    return L

def get_edges(G, sort=True):
    return sorter(G.edges, sort)

def get_components(G, sort=True):
    return sorter(nx.connected_components(G), sort)
    
def get_partition(G, sort=True):
    districts = set(d for n, d in G.nodes(data='district'))
    return sorter((district_view(G, D).nodes for D in districts), sort)    

def district_view(G, D):
    # get subgraph of given districts
    if isinstance(D, int):
        return nx.subgraph_view(G, lambda n: G.nodes[n]['district'] == D)
    else:
        return nx.subgraph_view(G, lambda n: G.nodes[n]['district'] in D)

def get_components_district(G, D, sort=True):
    # get connected components given districts
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
    return get_partition(G).__hash__()


def get_states():
    query = f"""
select
    state_fips_code as fips
    , state_postal_abbreviation as abbr
    , state_name as name
from
    bigquery-public-data.census_utility.fips_codes_states
where
    state_fips_code <= '56'
"""
    return lower_cols(run_query(query)).set_index('name')

try:
    states
except:
    print('getting states')
    states = get_states()

################# Census definitions #################
Census_columns = {'joins':  ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno']}

Census_columns['geo'] = ({'name':'fileid', 'field_type':'string'}, {'name':'stusab', 'field_type':'string'}, {'name':'sumlev', 'field_type':'string'}, {'name':'geovar', 'field_type':'string'}, {'name':'geocomp', 'field_type':'string'}, {'name':'chariter', 'field_type':'string'}, {'name':'cifsn', 'field_type':'string'}, {'name':'logrecno', 'field_type':'integer'}, {'name':'geoid', 'field_type':'string'}, {'name':'geocode', 'field_type':'string'}, {'name':'region', 'field_type':'string'}, {'name':'division', 'field_type':'string'}, {'name':'state', 'field_type':'string'}, {'name':'statens', 'field_type':'string'}, {'name':'county', 'field_type':'string'}, {'name':'countycc', 'field_type':'string'}, {'name':'countyns', 'field_type':'string'}, {'name':'cousub', 'field_type':'string'}, {'name':'cousubcc', 'field_type':'string'}, {'name':'cousubns', 'field_type':'string'}, {'name':'submcd', 'field_type':'string'}, {'name':'submcdcc', 'field_type':'string'}, {'name':'submcdns', 'field_type':'string'}, {'name':'estate', 'field_type':'string'}, {'name':'estatecc', 'field_type':'string'}, {'name':'estatens', 'field_type':'string'}, {'name':'concit', 'field_type':'string'}, {'name':'concitcc', 'field_type':'string'}, {'name':'concitns', 'field_type':'string'}, {'name':'place', 'field_type':'string'}, {'name':'placecc', 'field_type':'string'}, {'name':'placens', 'field_type':'string'}, {'name':'tract', 'field_type':'string'}, {'name':'blkgrp', 'field_type':'string'}, {'name':'block', 'field_type':'string'}, {'name':'aianhh', 'field_type':'string'}, {'name':'aihhtli', 'field_type':'string'}, {'name':'aianhhfp', 'field_type':'string'}, {'name':'aianhhcc', 'field_type':'string'}, {'name':'aianhhns', 'field_type':'string'}, {'name':'aits', 'field_type':'string'}, {'name':'aitsfp', 'field_type':'string'}, {'name':'aitscc', 'field_type':'string'}, {'name':'aitsns', 'field_type':'string'}, {'name':'ttract', 'field_type':'string'}, {'name':'tblkgrp', 'field_type':'string'}, {'name':'anrc', 'field_type':'string'}, {'name':'anrccc', 'field_type':'string'}, {'name':'anrcns', 'field_type':'string'}, {'name':'cbsa', 'field_type':'string'}, {'name':'memi', 'field_type':'string'}, {'name':'csa', 'field_type':'string'}, {'name':'metdiv', 'field_type':'string'}, {'name':'necta', 'field_type':'string'}, {'name':'nmemi', 'field_type':'string'}, {'name':'cnecta', 'field_type':'string'}, {'name':'nectadiv', 'field_type':'string'}, {'name':'cbsapci', 'field_type':'string'}, {'name':'nectapci', 'field_type':'string'}, {'name':'ua', 'field_type':'string'}, {'name':'uatype', 'field_type':'string'}, {'name':'ur', 'field_type':'string'}, {'name':'cd116', 'field_type':'string'}, {'name':'cd118', 'field_type':'string'}, {'name':'cd119', 'field_type':'string'}, {'name':'cd120', 'field_type':'string'}, {'name':'cd121', 'field_type':'string'}, {'name':'sldu18', 'field_type':'string'}, {'name':'sldu22', 'field_type':'string'}, {'name':'sldu24', 'field_type':'string'}, {'name':'sldu26', 'field_type':'string'}, {'name':'sldu28', 'field_type':'string'}, {'name':'sldl18', 'field_type':'string'}, {'name':'sldl22', 'field_type':'string'}, {'name':'sldl24', 'field_type':'string'}, {'name':'sldl26', 'field_type':'string'}, {'name':'sldl28', 'field_type':'string'}, {'name':'vtd', 'field_type':'string'}, {'name':'vtdi', 'field_type':'string'}, {'name':'zcta', 'field_type':'string'}, {'name':'sdelm', 'field_type':'string'}, {'name':'sdsec', 'field_type':'string'}, {'name':'sduni', 'field_type':'string'}, {'name':'puma', 'field_type':'string'}, {'name':'arealand', 'field_type':'string'}, {'name':'areawatr', 'field_type':'string'}, {'name':'basename', 'field_type':'string'}, {'name':'name', 'field_type':'string'}, {'name':'funcstat', 'field_type':'string'}, {'name':'gcuni', 'field_type':'string'}, {'name':'pop100', 'field_type':'string'}, {'name':'hu100', 'field_type':'string'}, {'name':'intptlat', 'field_type':'string'}, {'name':'intptlon', 'field_type':'string'}, {'name':'lsadc', 'field_type':'string'}, {'name':'partflag', 'field_type':'string'}, {'name':'uga', 'field_type':'string'})

Census_columns['1'] = ({'name':'fileid', 'field_type':'string'}, {'name':'stusab', 'field_type':'string'}, {'name':'chariter', 'field_type':'string'}, {'name':'cifsn', 'field_type':'string'}, {'name':'logrecno', 'field_type':'integer'}, {'name':'total_pop', 'field_type':'integer'}, {'name':'total_1race', 'field_type':'integer'}, {'name':'total_white', 'field_type':'integer'}, {'name':'total_black', 'field_type':'integer'}, {'name':'total_native', 'field_type':'integer'}, {'name':'total_asian', 'field_type':'integer'}, {'name':'total_pacific', 'field_type':'integer'}, {'name':'total_other', 'field_type':'integer'}, {'name':'total_2ormorerace', 'field_type':'integer'}, {'name':'total_2race', 'field_type':'integer'}, {'name':'total_white_black', 'field_type':'integer'}, {'name':'total_white_native', 'field_type':'integer'}, {'name':'total_white_asian', 'field_type':'integer'}, {'name':'total_white_pacific', 'field_type':'integer'}, {'name':'total_white_other', 'field_type':'integer'}, {'name':'total_black_native', 'field_type':'integer'}, {'name':'total_black_asian', 'field_type':'integer'}, {'name':'total_black_pacific', 'field_type':'integer'}, {'name':'total_black_other', 'field_type':'integer'}, {'name':'total_native_asian', 'field_type':'integer'}, {'name':'total_native_pacific', 'field_type':'integer'}, {'name':'total_native_other', 'field_type':'integer'}, {'name':'total_asian_pacific', 'field_type':'integer'}, {'name':'total_asian_other', 'field_type':'integer'}, {'name':'total_pacific_other', 'field_type':'integer'}, {'name':'total_3race', 'field_type':'integer'}, {'name':'total_white_black_native', 'field_type':'integer'}, {'name':'total_white_black_asian', 'field_type':'integer'}, {'name':'total_white_black_pacific', 'field_type':'integer'}, {'name':'total_white_black_other', 'field_type':'integer'}, {'name':'total_white_native_asian', 'field_type':'integer'}, {'name':'total_white_native_pacific', 'field_type':'integer'}, {'name':'total_white_native_other', 'field_type':'integer'}, {'name':'total_white_asian_pacific', 'field_type':'integer'}, {'name':'total_white_asian_other', 'field_type':'integer'}, {'name':'total_white_pacific_other', 'field_type':'integer'}, {'name':'total_black_native_asian', 'field_type':'integer'}, {'name':'total_black_native_pacific', 'field_type':'integer'}, {'name':'total_black_native_other', 'field_type':'integer'}, {'name':'total_black_asian_pacific', 'field_type':'integer'}, {'name':'total_black_asian_other', 'field_type':'integer'}, {'name':'total_black_pacific_other', 'field_type':'integer'}, {'name':'total_native_asian_pacific', 'field_type':'integer'}, {'name':'total_native_asian_other', 'field_type':'integer'}, {'name':'total_native_pacific_other', 'field_type':'integer'}, {'name':'total_asian_pacific_other', 'field_type':'integer'}, {'name':'total_4race', 'field_type':'integer'}, {'name':'total_white_black_native_asian', 'field_type':'integer'}, {'name':'total_white_black_native_pacific', 'field_type':'integer'}, {'name':'total_white_black_native_other', 'field_type':'integer'}, {'name':'total_white_black_asian_pacific', 'field_type':'integer'}, {'name':'total_white_black_asian_other', 'field_type':'integer'}, {'name':'total_white_black_pacific_other', 'field_type':'integer'}, {'name':'total_white_native_asian_pacific', 'field_type':'integer'}, {'name':'total_white_native_asian_other', 'field_type':'integer'}, {'name':'total_white_native_pacific_other', 'field_type':'integer'}, {'name':'total_white_asian_pacific_other', 'field_type':'integer'}, {'name':'total_black_native_asian_pacific', 'field_type':'integer'}, {'name':'total_black_native_asian_other', 'field_type':'integer'}, {'name':'total_black_native_pacific_other', 'field_type':'integer'}, {'name':'total_black_asian_pacific_other', 'field_type':'integer'}, {'name':'total_native_asian_pacific_other', 'field_type':'integer'}, {'name':'total_5race', 'field_type':'integer'}, {'name':'total_white_black_native_asian_pacific', 'field_type':'integer'}, {'name':'total_white_black_native_asian_other', 'field_type':'integer'}, {'name':'total_white_black_native_pacific_other', 'field_type':'integer'}, {'name':'total_white_black_asian_pacific_other', 'field_type':'integer'}, {'name':'total_white_native_asian_pacific_other', 'field_type':'integer'}, {'name':'total_black_native_asian_pacific_other', 'field_type':'integer'}, {'name':'total_6race', 'field_type':'integer'}, {'name':'total_white_black_native_asian_pacific_other', 'field_type':'integer'}, {'name':'total_pop2', 'field_type':'integer'}, {'name':'hisp', 'field_type':'integer'}, {'name':'nonhisp', 'field_type':'integer'}, {'name':'nonhisp_1race', 'field_type':'integer'}, {'name':'nonhisp_white', 'field_type':'integer'}, {'name':'nonhisp_black', 'field_type':'integer'}, {'name':'nonhisp_native', 'field_type':'integer'}, {'name':'nonhisp_asian', 'field_type':'integer'}, {'name':'nonhisp_pacific', 'field_type':'integer'}, {'name':'nonhisp_other', 'field_type':'integer'}, {'name':'nonhisp_2ormorerace', 'field_type':'integer'}, {'name':'nonhisp_2race', 'field_type':'integer'}, {'name':'nonhisp_white_black', 'field_type':'integer'}, {'name':'nonhisp_white_native', 'field_type':'integer'}, {'name':'nonhisp_white_asian', 'field_type':'integer'}, {'name':'nonhisp_white_pacific', 'field_type':'integer'}, {'name':'nonhisp_white_other', 'field_type':'integer'}, {'name':'nonhisp_black_native', 'field_type':'integer'}, {'name':'nonhisp_black_asian', 'field_type':'integer'}, {'name':'nonhisp_black_pacific', 'field_type':'integer'}, {'name':'nonhisp_black_other', 'field_type':'integer'}, {'name':'nonhisp_native_asian', 'field_type':'integer'}, {'name':'nonhisp_native_pacific', 'field_type':'integer'}, {'name':'nonhisp_native_other', 'field_type':'integer'}, {'name':'nonhisp_asian_pacific', 'field_type':'integer'}, {'name':'nonhisp_asian_other', 'field_type':'integer'}, {'name':'nonhisp_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_3race', 'field_type':'integer'}, {'name':'nonhisp_white_black_native', 'field_type':'integer'}, {'name':'nonhisp_white_black_asian', 'field_type':'integer'}, {'name':'nonhisp_white_black_pacific', 'field_type':'integer'}, {'name':'nonhisp_white_black_other', 'field_type':'integer'}, {'name':'nonhisp_white_native_asian', 'field_type':'integer'}, {'name':'nonhisp_white_native_pacific', 'field_type':'integer'}, {'name':'nonhisp_white_native_other', 'field_type':'integer'}, {'name':'nonhisp_white_asian_pacific', 'field_type':'integer'}, {'name':'nonhisp_white_asian_other', 'field_type':'integer'}, {'name':'nonhisp_white_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_black_native_asian', 'field_type':'integer'}, {'name':'nonhisp_black_native_pacific', 'field_type':'integer'}, {'name':'nonhisp_black_native_other', 'field_type':'integer'}, {'name':'nonhisp_black_asian_pacific', 'field_type':'integer'}, {'name':'nonhisp_black_asian_other', 'field_type':'integer'}, {'name':'nonhisp_black_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_native_asian_pacific', 'field_type':'integer'}, {'name':'nonhisp_native_asian_other', 'field_type':'integer'}, {'name':'nonhisp_native_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_asian_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_4race', 'field_type':'integer'}, {'name':'nonhisp_white_black_native_asian', 'field_type':'integer'}, {'name':'nonhisp_white_black_native_pacific', 'field_type':'integer'}, {'name':'nonhisp_white_black_native_other', 'field_type':'integer'}, {'name':'nonhisp_white_black_asian_pacific', 'field_type':'integer'}, {'name':'nonhisp_white_black_asian_other', 'field_type':'integer'}, {'name':'nonhisp_white_black_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_white_native_asian_pacific', 'field_type':'integer'}, {'name':'nonhisp_white_native_asian_other', 'field_type':'integer'}, {'name':'nonhisp_white_native_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_white_asian_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_black_native_asian_pacific', 'field_type':'integer'}, {'name':'nonhisp_black_native_asian_other', 'field_type':'integer'}, {'name':'nonhisp_black_native_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_black_asian_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_native_asian_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_5race', 'field_type':'integer'}, {'name':'nonhisp_white_black_native_asian_pacific', 'field_type':'integer'}, {'name':'nonhisp_white_black_native_asian_other', 'field_type':'integer'}, {'name':'nonhisp_white_black_native_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_white_black_asian_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_white_native_asian_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_black_native_asian_pacific_other', 'field_type':'integer'}, {'name':'nonhisp_6race', 'field_type':'integer'}, {'name':'nonhisp_white_black_native_asian_pacific_other', 'field_type':'integer'})

Census_columns['2'] = ({'name':'fileid', 'field_type':'string'}, {'name':'stusab', 'field_type':'string'}, {'name':'chariter', 'field_type':'string'}, {'name':'cifsn', 'field_type':'string'}, {'name':'logrecno', 'field_type':'integer'}, {'name':'vap', 'field_type':'integer'}, {'name':'vap_1race', 'field_type':'integer'}, {'name':'vap_white', 'field_type':'integer'}, {'name':'vap_black', 'field_type':'integer'}, {'name':'vap_native', 'field_type':'integer'}, {'name':'vap_asian', 'field_type':'integer'}, {'name':'vap_pacific', 'field_type':'integer'}, {'name':'vap_other', 'field_type':'integer'}, {'name':'vap_2ormorerace', 'field_type':'integer'}, {'name':'vap_2race', 'field_type':'integer'}, {'name':'vap_white_black', 'field_type':'integer'}, {'name':'vap_white_native', 'field_type':'integer'}, {'name':'vap_white_asian', 'field_type':'integer'}, {'name':'vap_white_pacific', 'field_type':'integer'}, {'name':'vap_white_other', 'field_type':'integer'}, {'name':'vap_black_native', 'field_type':'integer'}, {'name':'vap_black_asian', 'field_type':'integer'}, {'name':'vap_black_pacific', 'field_type':'integer'}, {'name':'vap_black_other', 'field_type':'integer'}, {'name':'vap_native_asian', 'field_type':'integer'}, {'name':'vap_native_pacific', 'field_type':'integer'}, {'name':'vap_native_other', 'field_type':'integer'}, {'name':'vap_asian_pacific', 'field_type':'integer'}, {'name':'vap_asian_other', 'field_type':'integer'}, {'name':'vap_pacific_other', 'field_type':'integer'}, {'name':'vap_3race', 'field_type':'integer'}, {'name':'vap_white_black_native', 'field_type':'integer'}, {'name':'vap_white_black_asian', 'field_type':'integer'}, {'name':'vap_white_black_pacific', 'field_type':'integer'}, {'name':'vap_white_black_other', 'field_type':'integer'}, {'name':'vap_white_native_asian', 'field_type':'integer'}, {'name':'vap_white_native_pacific', 'field_type':'integer'}, {'name':'vap_white_native_other', 'field_type':'integer'}, {'name':'vap_white_asian_pacific', 'field_type':'integer'}, {'name':'vap_white_asian_other', 'field_type':'integer'}, {'name':'vap_white_pacific_other', 'field_type':'integer'}, {'name':'vap_black_native_asian', 'field_type':'integer'}, {'name':'vap_black_native_pacific', 'field_type':'integer'}, {'name':'vap_black_native_other', 'field_type':'integer'}, {'name':'vap_black_asian_pacific', 'field_type':'integer'}, {'name':'vap_black_asian_other', 'field_type':'integer'}, {'name':'vap_black_pacific_other', 'field_type':'integer'}, {'name':'vap_native_asian_pacific', 'field_type':'integer'}, {'name':'vap_native_asian_other', 'field_type':'integer'}, {'name':'vap_native_pacific_other', 'field_type':'integer'}, {'name':'vap_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_4race', 'field_type':'integer'}, {'name':'vap_white_black_native_asian', 'field_type':'integer'}, {'name':'vap_white_black_native_pacific', 'field_type':'integer'}, {'name':'vap_white_black_native_other', 'field_type':'integer'}, {'name':'vap_white_black_asian_pacific', 'field_type':'integer'}, {'name':'vap_white_black_asian_other', 'field_type':'integer'}, {'name':'vap_white_black_pacific_other', 'field_type':'integer'}, {'name':'vap_white_native_asian_pacific', 'field_type':'integer'}, {'name':'vap_white_native_asian_other', 'field_type':'integer'}, {'name':'vap_white_native_pacific_other', 'field_type':'integer'}, {'name':'vap_white_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_black_native_asian_pacific', 'field_type':'integer'}, {'name':'vap_black_native_asian_other', 'field_type':'integer'}, {'name':'vap_black_native_pacific_other', 'field_type':'integer'}, {'name':'vap_black_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_native_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_5race', 'field_type':'integer'}, {'name':'vap_white_black_native_asian_pacific', 'field_type':'integer'}, {'name':'vap_white_black_native_asian_other', 'field_type':'integer'}, {'name':'vap_white_black_native_pacific_other', 'field_type':'integer'}, {'name':'vap_white_black_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_white_native_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_black_native_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_6race', 'field_type':'integer'}, {'name':'vap_white_black_native_asian_pacific_other', 'field_type':'integer'}, {'name':'vap2', 'field_type':'integer'}, {'name':'vap_hisp', 'field_type':'integer'}, {'name':'vap_nonhisp', 'field_type':'integer'}, {'name':'vap_nonhisp_1race', 'field_type':'integer'}, {'name':'vap_nonhisp_white', 'field_type':'integer'}, {'name':'vap_nonhisp_black', 'field_type':'integer'}, {'name':'vap_nonhisp_native', 'field_type':'integer'}, {'name':'vap_nonhisp_asian', 'field_type':'integer'}, {'name':'vap_nonhisp_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_other', 'field_type':'integer'}, {'name':'vap_nonhisp_2ormorerace', 'field_type':'integer'}, {'name':'vap_nonhisp_2race', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black', 'field_type':'integer'}, {'name':'vap_nonhisp_white_native', 'field_type':'integer'}, {'name':'vap_nonhisp_white_asian', 'field_type':'integer'}, {'name':'vap_nonhisp_white_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_white_other', 'field_type':'integer'}, {'name':'vap_nonhisp_black_native', 'field_type':'integer'}, {'name':'vap_nonhisp_black_asian', 'field_type':'integer'}, {'name':'vap_nonhisp_black_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_black_other', 'field_type':'integer'}, {'name':'vap_nonhisp_native_asian', 'field_type':'integer'}, {'name':'vap_nonhisp_native_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_native_other', 'field_type':'integer'}, {'name':'vap_nonhisp_asian_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_asian_other', 'field_type':'integer'}, {'name':'vap_nonhisp_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_3race', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_native', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_asian', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_native_asian', 'field_type':'integer'}, {'name':'vap_nonhisp_white_native_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_white_native_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_asian_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_white_asian_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_black_native_asian', 'field_type':'integer'}, {'name':'vap_nonhisp_black_native_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_black_native_other', 'field_type':'integer'}, {'name':'vap_nonhisp_black_asian_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_black_asian_other', 'field_type':'integer'}, {'name':'vap_nonhisp_black_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_native_asian_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_native_asian_other', 'field_type':'integer'}, {'name':'vap_nonhisp_native_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_4race', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_native_asian', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_native_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_native_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_asian_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_asian_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_native_asian_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_white_native_asian_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_native_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_black_native_asian_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_black_native_asian_other', 'field_type':'integer'}, {'name':'vap_nonhisp_black_native_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_black_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_native_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_5race', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_native_asian_pacific', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_native_asian_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_native_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_white_native_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_black_native_asian_pacific_other', 'field_type':'integer'}, {'name':'vap_nonhisp_6race', 'field_type':'integer'}, {'name':'vap_nonhisp_white_black_native_asian_pacific_other', 'field_type':'integer'}, {'name':'housing_total', 'field_type':'integer'}, {'name':'housing_occupied', 'field_type':'integer'}, {'name':'housing_vacant', 'field_type':'integer'})

Census_columns['3'] = ({'name':'fileid', 'field_type':'string'}, {'name':'stusab', 'field_type':'string'}, {'name':'chariter', 'field_type':'string'}, {'name':'cifsn', 'field_type':'string'}, {'name':'logrecno', 'field_type':'integer'}, {'name':'groupquarters', 'field_type':'integer'}, {'name':'groupquarters_institute', 'field_type':'integer'}, {'name':'groupquarters_institute_jail_adult', 'field_type':'integer'}, {'name':'groupquarters_institute_jail_juvenile', 'field_type':'integer'}, {'name':'groupquarters_institute_nursing', 'field_type':'integer'}, {'name':'groupquarters_institute_other', 'field_type':'integer'}, {'name':'groupquarters_noninstitute', 'field_type':'integer'}, {'name':'groupquarters_noninstitute_college', 'field_type':'integer'}, {'name':'groupquarters_noninstitute_military', 'field_type':'integer'}, {'name':'groupquarters_noninstitute_other', 'field_type':'integer'})


Census_columns['data'] = [x['name'] for x in Census_columns['1'] + Census_columns['2'] + Census_columns['3'] if x['name'] not in Census_columns['joins'] and x['name'][-1]!='2']

# census_columns['geo'] = (('fileid','string'), ('stusab','string'), ('sumlev','string'), ('geovar','string'), ('geocomp','string'), ('chariter','string'), ('cifsn','string'), ('logrecno','integer'), ('geoid','string'), ('geocode','string'), ('region','string'), ('division','string'), ('state','string'), ('statens','string'), ('county','string'), ('countycc','string'), ('countyns','string'), ('cousub','string'), ('cousubcc','string'), ('cousubns','string'), ('submcd','string'), ('submcdcc','string'), ('submcdns','string'), ('estate','string'), ('estatecc','string'), ('estatens','string'), ('concit','string'), ('concitcc','string'), ('concitns','string'), ('place','string'), ('placecc','string'), ('placens','string'), ('tract','string'), ('blkgrp','string'), ('block','string'), ('aianhh','string'), ('aihhtli','string'), ('aianhhfp','string'), ('aianhhcc','string'), ('aianhhns','string'), ('aits','string'), ('aitsfp','string'), ('aitscc','string'), ('aitsns','string'), ('ttract','string'), ('tblkgrp','string'), ('anrc','string'), ('anrccc','string'), ('anrcns','string'), ('cbsa','string'), ('memi','string'), ('csa','string'), ('metdiv','string'), ('necta','string'), ('nmemi','string'), ('cnecta','string'), ('nectadiv','string'), ('cbsapci','string'), ('nectapci','string'), ('ua','string'), ('uatype','string'), ('ur','string'), ('cd116','string'), ('cd118','string'), ('cd119','string'), ('cd120','string'), ('cd121','string'), ('sldu18','string'), ('sldu22','string'), ('sldu24','string'), ('sldu26','string'), ('sldu28','string'), ('sldl18','string'), ('sldl22','string'), ('sldl24','string'), ('sldl26','string'), ('sldl28','string'), ('vtd','string'), ('vtdi','string'), ('zcta','string'), ('sdelm','string'), ('sdsec','string'), ('sduni','string'), ('puma','string'), ('arealand','string'), ('areawatr','string'), ('basename','string'), ('name','string'), ('funcstat','string'), ('gcuni','string'), ('pop100','string'), ('hu100','string'), ('intptlat','string'), ('intptlon','string'), ('lsadc','string'), ('partflag','string'), ('uga','string'))

                  
# census_columns['1'] = ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno', 'total_pop', 'total_1race', 'total_white', 'total_black', 'total_native', 'total_asian', 'total_pacific', 'total_other', 'total_2ormorerace', 'total_2race', 'total_white_black', 'total_white_native', 'total_white_asian', 'total_white_pacific', 'total_white_other', 'total_black_native', 'total_black_asian', 'total_black_pacific', 'total_black_other', 'total_native_asian', 'total_native_pacific', 'total_native_other', 'total_asian_pacific', 'total_asian_other', 'total_pacific_other', 'total_3race', 'total_white_black_native', 'total_white_black_asian', 'total_white_black_pacific', 'total_white_black_other', 'total_white_native_asian', 'total_white_native_pacific', 'total_white_native_other', 'total_white_asian_pacific', 'total_white_asian_other', 'total_white_pacific_other', 'total_black_native_asian', 'total_black_native_pacific', 'total_black_native_other', 'total_black_asian_pacific', 'total_black_asian_other', 'total_black_pacific_other', 'total_native_asian_pacific', 'total_native_asian_other', 'total_native_pacific_other', 'total_asian_pacific_other', 'total_4race', 'total_white_black_native_asian', 'total_white_black_native_pacific', 'total_white_black_native_other', 'total_white_black_asian_pacific', 'total_white_black_asian_other', 'total_white_black_pacific_other', 'total_white_native_asian', 'total_white_native_asian_other', 'total_white_native_pacific_other', 'total_white_asian_pacific_other', 'total_black_native_asian_pacific', 'total_black_native_asian_other', 'total_black_native_pacific_other', 'total_black_asian_pacific_other', 'total_native_asian_pacific_other', 'total_5race', 'total_white_black_native_asian_pacific', 'total_white_black_native_asian_other', 'total_white_black_native_pacific_other', 'total_white_black_asian_pacific_other', 'total_white_native_asian_pacific_other', 'total_black_native_asian_pacific_other', 'total_6race', 'total_white_black_native_asian_pacific_other', 'total_pop', 'hisp', 'nonhisp', 'nonhisp_1race', 'nonhisp_white', 'nonhisp_black', 'nonhisp_native', 'nonhisp_asian', 'nonhisp_pacific', 'nonhisp_other', 'nonhisp_2ormorerace', 'nonhisp_2race', 'nonhisp_white_black', 'nonhisp_white_native', 'nonhisp_white_asian', 'nonhisp_white_pacific', 'nonhisp_white_other', 'nonhisp_black_native', 'nonhisp_black_asian', 'nonhisp_black_pacific', 'nonhisp_black_other', 'nonhisp_native_asian', 'nonhisp_native_pacific', 'nonhisp_native_other', 'nonhisp_asian_pacific', 'nonhisp_asian_other', 'nonhisp_pacific_other', 'nonhisp_3race', 'nonhisp_white_black_native', 'nonhisp_white_black_asian', 'nonhisp_white_black_pacific', 'nonhisp_white_black_other', 'nonhisp_white_native_asian', 'nonhisp_white_native_pacific', 'nonhisp_white_native_other', 'nonhisp_white_asian_pacific', 'nonhisp_white_asian_other', 'nonhisp_white_pacific_other', 'nonhisp_black_native_asian', 'nonhisp_black_native_pacific', 'nonhisp_black_native_other', 'nonhisp_black_asian_pacific', 'nonhisp_black_asian_other', 'nonhisp_black_pacific_other', 'nonhisp_native_asian_pacific', 'nonhisp_native_asian_other', 'nonhisp_native_pacific_other', 'nonhisp_asian_pacific_other', 'nonhisp_4race', 'nonhisp_white_black_native_asian', 'nonhisp_white_black_native_pacific', 'nonhisp_white_black_native_other', 'nonhisp_white_black_asian_pacific', 'nonhisp_white_black_asian_other', 'nonhisp_white_black_pacific_other', 'nonhisp_white_native_asian', 'nonhisp_white_native_asian_other', 'nonhisp_white_native_pacific_other', 'nonhisp_white_asian_pacific_other', 'nonhisp_black_native_asian_pacific', 'nonhisp_black_native_asian_other', 'nonhisp_black_native_pacific_other', 'nonhisp_black_asian_pacific_other', 'nonhisp_native_asian_pacific_other', 'nonhisp_5race', 'nonhisp_white_black_native_asian_pacific', 'nonhisp_white_black_native_asian_other', 'nonhisp_white_black_native_pacific_other', 'nonhisp_white_black_asian_pacific_other', 'nonhisp_white_native_asian_pacific_other', 'nonhisp_black_native_asian_pacific_other', 'nonhisp_6race', 'nonhisp_white_black_native_asian_pacific_other']

# census_columns['2'] = ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno', 'vap_pop', 'vap_1race', 'vap_white', 'vap_black', 'vap_native', 'vap_asian', 'vap_pacific', 'vap_other', 'vap_2ormorerace', 'vap_2race', 'vap_white_black', 'vap_white_native', 'vap_white_asian', 'vap_white_pacific', 'vap_white_other', 'vap_black_native', 'vap_black_asian', 'vap_black_pacific', 'vap_black_other', 'vap_native_asian', 'vap_native_pacific', 'vap_native_other', 'vap_asian_pacific', 'vap_asian_other', 'vap_pacific_other', 'vap_3race', 'vap_white_black_native', 'vap_white_black_asian', 'vap_white_black_pacific', 'vap_white_black_other', 'vap_white_native_asian', 'vap_white_native_pacific', 'vap_white_native_other', 'vap_white_asian_pacific', 'vap_white_asian_other', 'vap_white_pacific_other', 'vap_black_native_asian', 'vap_black_native_pacific', 'vap_black_native_other', 'vap_black_asian_pacific', 'vap_black_asian_other', 'vap_black_pacific_other', 'vap_native_asian_pacific', 'vap_native_asian_other', 'vap_native_pacific_other', 'vap_asian_pacific_other', 'vap_4race', 'vap_white_black_native_asian', 'vap_white_black_native_pacific', 'vap_white_black_native_other', 'vap_white_black_asian_pacific', 'vap_white_black_asian_other', 'vap_white_black_pacific_other', 'vap_white_native_asian', 'vap_white_native_asian_other', 'vap_white_native_pacific_other', 'vap_white_asian_pacific_other', 'vap_black_native_asian_pacific', 'vap_black_native_asian_other', 'vap_black_native_pacific_other', 'vap_black_asian_pacific_other', 'vap_native_asian_pacific_other', 'vap_5race', 'vap_white_black_native_asian_pacific', 'vap_white_black_native_asian_other', 'vap_white_black_native_pacific_other', 'vap_white_black_asian_pacific_other', 'vap_white_native_asian_pacific_other', 'vap_black_native_asian_pacific_other', 'vap_6race', 'vap_white_black_native_asian_pacific_other', 'vap_pop', 'vap_hisp', 'vap_nonhisp', 'vap_nonhisp_1race', 'vap_nonhisp_white', 'vap_nonhisp_black', 'vap_nonhisp_native', 'vap_nonhisp_asian', 'vap_nonhisp_pacific', 'vap_nonhisp_other', 'vap_nonhisp_2ormorerace', 'vap_nonhisp_2race', 'vap_nonhisp_white_black', 'vap_nonhisp_white_native', 'vap_nonhisp_white_asian', 'vap_nonhisp_white_pacific', 'vap_nonhisp_white_other', 'vap_nonhisp_black_native', 'vap_nonhisp_black_asian', 'vap_nonhisp_black_pacific', 'vap_nonhisp_black_other', 'vap_nonhisp_native_asian', 'vap_nonhisp_native_pacific', 'vap_nonhisp_native_other', 'vap_nonhisp_asian_pacific', 'vap_nonhisp_asian_other', 'vap_nonhisp_pacific_other', 'vap_nonhisp_3race', 'vap_nonhisp_white_black_native', 'vap_nonhisp_white_black_asian', 'vap_nonhisp_white_black_pacific', 'vap_nonhisp_white_black_other', 'vap_nonhisp_white_native_asian', 'vap_nonhisp_white_native_pacific', 'vap_nonhisp_white_native_other', 'vap_nonhisp_white_asian_pacific', 'vap_nonhisp_white_asian_other', 'vap_nonhisp_white_pacific_other', 'vap_nonhisp_black_native_asian', 'vap_nonhisp_black_native_pacific', 'vap_nonhisp_black_native_other', 'vap_nonhisp_black_asian_pacific', 'vap_nonhisp_black_asian_other', 'vap_nonhisp_black_pacific_other', 'vap_nonhisp_native_asian_pacific', 'vap_nonhisp_native_asian_other', 'vap_nonhisp_native_pacific_other', 'vap_nonhisp_asian_pacific_other', 'vap_nonhisp_4race', 'vap_nonhisp_white_black_native_asian', 'vap_nonhisp_white_black_native_pacific', 'vap_nonhisp_white_black_native_other', 'vap_nonhisp_white_black_asian_pacific', 'vap_nonhisp_white_black_asian_other', 'vap_nonhisp_white_black_pacific_other', 'vap_nonhisp_white_native_asian', 'vap_nonhisp_white_native_asian_other', 'vap_nonhisp_white_native_pacific_other', 'vap_nonhisp_white_asian_pacific_other', 'vap_nonhisp_black_native_asian_pacific', 'vap_nonhisp_black_native_asian_other', 'vap_nonhisp_black_native_pacific_other', 'vap_nonhisp_black_asian_pacific_other', 'vap_nonhisp_native_asian_pacific_other', 'vap_nonhisp_5race', 'vap_nonhisp_white_black_native_asian_pacific', 'vap_nonhisp_white_black_native_asian_other', 'vap_nonhisp_white_black_native_pacific_other', 'vap_nonhisp_white_black_asian_pacific_other', 'vap_nonhisp_white_native_asian_pacific_other', 'vap_nonhisp_black_native_asian_pacific_other', 'vap_nonhisp_6race', 'vap_nonhisp_white_black_native_asian_pacific_other', 'housing_total', 'housing_occupied', 'housing_vacant']

# census_columns['3'] = ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno', 'groupquarters', 'groupquarters_institute', 'groupquarters_institute_jail_adult', 'groupquarters_institute_jail_juvenile', 'groupquarters_institute_nursing', 'groupquarters_institute_other', 'groupquarters_noninstitute', 'groupquarters_noninstitute_college', 'groupquarters_noninstitute_military', 'groupquarters_noninstitute_other']