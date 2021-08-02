import google, time, datetime, dataclasses, typing, os, pathlib, shutil, urllib
import zipfile as zf, numpy as np, pandas as pd, geopandas as gpd, networkx as nx
import matplotlib.pyplot as plt
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

pd.set_option('display.max_columns', None)
cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient   = bigquery.Client(credentials=cred, project=proj)
root_path  = pathlib.Path(root_path)
data_path  = root_path / 'redistricting_data'
bq_dataset = proj_id   +'.redistricting_data'
try:
    rng = np.random.RandomState(seed)
except:
    rng = np.random.RandomState(42)

Levels = ['tabblock', 'bg', 'tract', 'cnty', 'state', 'cntyvtd']
Districts = ['cd', 'sldu', 'sldl']
Years = [2010, 2020]
Groups = ['all', 'hl']
concat_str = ' ... '

def check_level(level):
    assert level in Levels, f"level must be one of {Levels}, got {level}"

def check_district(district):
    assert district in Districts, f"district must be one of {Districts}, got {district}"

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

def run_query(query):
    return bqclient.query(query).result().to_dataframe()

def read_table(tbl, rows=99999999999, start=0, cols='*'):
    query = f'select {", ".join(cols)} from {tbl} limit {rows}'
    if start is not None:
        query += f' offset {start}'
    return run_query(query)

def delete_table(tbl):
    bqclient.delete_table(tbl, not_found_ok=True)
    
def check_table(tbl):
    try:
        bqclient.get_table(tbl)
        return True
    except:
        return False

def get_cols(tbl):
    return [s.name for s in bqclient.get_table(tbl).schema if s.name.lower() != 'geoid']
        

def head(tbl, rows=10):
    return read_table(tbl, rows)

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

census_columns = {
    'joins':  ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno'],

    'widths': [6, 2, 3, 2, 3, 2, 7, 1, 1, 2, 3, 2, 2, 5, 2, 2, 5, 2, 2, 6, 1, 4, 2, 5, 2, 2, 4, 5, 2, 1, 3, 5, 2, 6, 1, 5, 2, 5, 2, 5, 3, 5, 2, 5, 3, 1, 1, 5, 2, 1, 1, 2, 3, 3, 6, 1, 3, 5, 5, 2, 5, 5, 5, 14, 14, 90, 1, 1, 9, 9, 11, 12, 2, 1, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 5, 18],

    'geo': ['fileid', 'stusab', 'sumlev', 'geocomp', 'chariter', 'cifsn', 'logrecno', 'region', 'division', 'state', 'county', 'countycc', 'countysc', 'cousub', 'cousubcc', 'cousubsc', 'place', 'placecc', 'placesc', 'tract', 'blkgrp', 'block', 'iuc', 'concit', 'concitcc', 'concitsc', 'aianhh', 'aianhhfp', 'aianhhcc', 'aihhtli', 'aitsce', 'aits', 'aitscc', 'ttract', 'tblkgrp', 'anrc', 'anrccc', 'cbsa', 'cbsasc', 'metdiv', 'csa', 'necta', 'nectasc', 'nectadiv', 'cnecta', 'cbsapci', 'nectapci', 'ua', 'uasc', 'uatype', 'ur', 'cd', 'sldu', 'sldl', 'vtd', 'vtdi', 'reserve2', 'zcta5', 'submcd', 'submcdcc', 'sdelm', 'sdsec', 'sduni', 'arealand', 'areawatr', 'name', 'funcstat', 'gcuni', 'pop100', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'partflag', 'reserve3', 'uga', 'statens', 'countyns', 'cousubns', 'placens', 'concitns', 'aianhhns', 'aitsns', 'anrcns', 'submcdns', 'cd113', 'cd114', 'cd115', 'sldu2', 'sldu3', 'sldu4', 'sldl2', 'sldl3', 'sldl4', 'aianhhsc', 'csasc', 'cnectasc', 'memi', 'nmemi', 'puma', 'reserved'],
                  
    '1': ['total', 'population_of_one_race', 'white_alone', 'black_or_african_american_alone', 'american_indian_and_alaska_native_alone', 'asian_alone', 'native_hawaiian_and_other_pacific_islander_alone', 'some_other_race_alone', 'population_of_two_or_more_races', 'population_of_two_races', 'white_black_or_african_american', 'white_american_indian_and_alaska_native', 'white_asian', 'white_native_hawaiian_and_other_pacific_islander', 'white_some_other_race', 'black_or_african_american_american_indian_and_alaska_native', 'black_or_african_american_asian', 'black_or_african_american_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_some_other_race', 'american_indian_and_alaska_native_asian', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander', 'asian_some_other_race', 'native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_three_races', 'white_black_or_african_american_american_indian_and_alaska_native', 'white_black_or_african_american_asian', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_some_other_race', 'white_american_indian_and_alaska_native_asian', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander', 'white_asian_some_other_race', 'white_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_asian_some_other_race', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_asian_some_other_race', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_four_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_asian_some_other_race', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_asian_some_other_race', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_five_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_six_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'total_hl', 'hispanic_or_latino_hl', 'not_hispanic_or_latino_hl', 'population_of_one_race_hl', 'white_alone_hl', 'black_or_african_american_alone_hl', 'american_indian_and_alaska_native_alone_hl', 'asian_alone_hl', 'native_hawaiian_and_other_pacific_islander_alone_hl', 'some_other_race_alone_hl', 'population_of_two_or_more_races_hl', 'population_of_two_races_hl', 'white_black_or_african_american_hl', 'white_american_indian_and_alaska_native_hl', 'white_asian_hl', 'white_native_hawaiian_and_other_pacific_islander_hl', 'white_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_hl', 'black_or_african_american_asian_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_some_other_race_hl', 'american_indian_and_alaska_native_asian_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_hl', 'asian_some_other_race_hl', 'native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_three_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_hl', 'white_black_or_african_american_asian_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_asian_some_other_race_hl', 'white_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_asian_some_other_race_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_asian_some_other_race_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_four_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_asian_some_other_race_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_five_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_six_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl'],

    '2': ['total_18', 'population_of_one_race_18', 'white_alone_18', 'black_or_african_american_alone_18', 'american_indian_and_alaska_native_alone_18', 'asian_alone_18', 'native_hawaiian_and_other_pacific_islander_alone_18', 'some_other_race_alone_18', 'population_of_two_or_more_races_18', 'population_of_two_races_18', 'white__black_or_african_american_18', 'white__american_indian_and_alaska_native_18', 'white__asian_18', 'white__native_hawaiian_and_other_pacific_islander_18', 'white__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native_18', 'black_or_african_american__asian_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__some_other_race_18', 'american_indian_and_alaska_native__asian_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander_18', 'asian__some_other_race_18', 'native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_three_races_18', 'white__black_or_african_american__american_indian_and_alaska_native_18', 'white__black_or_african_american__asian_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__some_other_race_18', 'white__american_indian_and_alaska_native__asian_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander_18', 'white__asian__some_other_race_18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__asian__some_other_race_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__asian__some_other_race_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_four_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__asian__some_other_race_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__asian__some_other_race_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_five_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_six_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'total_hl18', 'hispanic_or_latino_hl18', 'not_hispanic_or_latino_hl18', 'population_of_one_race_hl18', 'white_alone_hl18', 'black_or_african_american_alone_hl18', 'american_indian_and_alaska_native_alone_hl18', 'asian_alone_hl18', 'native_hawaiian_and_other_pacific_islander_alone_hl18', 'some_other_race_alone_hl18', 'population_of_two_or_more_races_hl18', 'population_of_two_races_hl18', 'white__black_or_african_american_hl18', 'white__american_indian_and_alaska_native_hl18', 'white__asian_hl18', 'white__native_hawaiian_and_other_pacific_islander_hl18', 'white__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native_hl18', 'black_or_african_american__asian_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__some_other_race_hl18', 'american_indian_and_alaska_native__asian_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander_hl18', 'asian__some_other_race_hl18', 'native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_three_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native_hl18', 'white__black_or_african_american__asian_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__asian__some_other_race_hl18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__asian__some_other_race_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__asian__some_other_race_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_four_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__asian__some_other_race_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_five_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_six_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'housing_total', 'housing_occupied', 'housing_vacant'],
}
census_columns['data'] = census_columns['1'] + census_columns['2']
census_columns['starts'] = 1 + np.insert(np.cumsum(census_columns['widths'])[:-1], 0, 0)