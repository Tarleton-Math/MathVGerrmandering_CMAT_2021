import google, time, datetime, dataclasses, typing, os, pathlib, shutil, urllib
import zipfile as zf, numpy as np, pandas as pd, geopandas as gpd, networkx as nx
import matplotlib.pyplot as plt, plotly.express as px
from shapely.ops import orient
from google.cloud import aiplatform, bigquery
try:
    from google.cloud.bigquery_storage import BigQueryReadClient
except:
    os.system('pip install --upgrade google-cloud-bigquery-storage')
    from google.cloud.bigquery_storage import BigQueryReadClient
    
import warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
warnings.filterwarnings('ignore', message='.*Pyarrow could not determine the type of columns*')

user_name = 'cook'

# user_name = input('user_name (default=cook)')
# if user_name == '':
#     user_name = 'cook'
default_random_seed = 1
random_seed = input(f'random_seed (default={default_random_seed})')
try:
    rng = np.random.default_rng(int(random_seed))    
except:
    rng = np.random.default_rng(default_random_seed)

pd.set_option('display.max_columns', None)
cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient   = bigquery.Client(credentials=cred, project=proj)
root_path  = pathlib.Path(root_path)
data_path  = root_path / 'redistricting_data'
bq_dataset = proj_id   +'.redistricting_data'

Levels = ['tabblock', 'bg', 'tract', 'cnty', 'state', 'cntyvtd']
District_types = ['cd', 'sldu', 'sldl']
Years = [2010, 2020]
Groups = ['all', 'hl']
concat_str = ' ... '

def check_level(level):
    assert level in Levels, f"level must be one of {Levels}, got {level}"

def check_district_type(district_type):
    assert district_type in District_types, f"district must be one of {District_types}, got {district_type}"

def check_year(year):
    assert year in Years, f"year must be one of {Years}, got {year}"

def check_group(group):
    assert group in Groups, f"group must be one of {Groups}, got {group}"
    
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

def listify(x):
    if x is None:
        return []
    if isinstance(x, pd.core.frame.DataFrame):
        x = x.to_dict('split')['data']
    if isinstance(x, (np.ndarray, pd.Series)):
        x = x.tolist()
    if isinstance(x, (list, tuple, set)):
        return list(x)
    else:
        return [x]

def extract_file(zipfile, fn, **kwargs):
    file = zipfile.extract(fn)
    return lower_cols(pd.read_csv(file, dtype=str, **kwargs))
#     return lower(pd.read_csv(file, dtype=str, **kwargs))

def check_table(tbl):
    try:
        bqclient.get_table(tbl)
        return True
    except:
        return False

def get_cols(tbl):
    print('hi')
    t = bqclient.get_table(tbl)
    cols = [s.name for s in t.schema]
    return cols
    
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
    except:
#         print(f'{tbl} not found', end=concat_str)
        pass

def read_table(tbl, rows=99999999999, start=0, cols='*'):
    query = f'select {", ".join(cols)} from {tbl} limit {rows}'
    if start is not None:
        query += f' offset {start}'
    return run_query(query)

def head(tbl, rows=10):
    return read_table(tbl, rows)

def get_cols(tbl):
    return [s.name for s in bqclient.get_table(tbl).schema if s.name.lower() != 'geoid']

def load_table(tbl, df=None, file=None, query=None, overwrite=True, preview_rows=0):
#     print(f'loading BigQuery table {tbl}', end=concat_str)
    if overwrite:
        delete_table(tbl)
    if df is not None:
        job = bqclient.load_table_from_dataframe(df, tbl).result()
    elif file is not None:
        with open(file, mode="rb") as f:
            job = bqclient.load_table_from_file(f, tbl, job_config=bigquery.LoadJobConfig(autodetect=True)).result()
    elif query is not None:
        job = bqclient.query(query, job_config=bigquery.QueryJobConfig(destination=tbl)).result()
    else:
        raise Exception('at least one of df, file, or query must be specified')
    if preview_rows > 0:
        display(head(tbl, preview_rows))
    return tbl

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

def yr_to_congress(yr):
    return min(116, int(yr-1786)/2)

def join_str(k=1):
    tab = '    '
    return ',\n' + k * tab

@dataclasses.dataclass
class Base():
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__dict__[key] = val

try:
    states
except:
    print('getting states')
    states = get_states()