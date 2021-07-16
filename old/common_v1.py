import google, time, datetime, dataclasses, typing, os, pathlib, shutil, urllib, zipfile as zf, numpy as np, pandas as pd, geopandas as gpd, networkx as nx
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
pd.set_option('display.max_columns', None)

# crs_map = 'NAD83'
# crs_map = 'epsg:4269'
# crs_area = 'esri:102003'
# crs_length = 'esri:102005'
# input is WKT in NAD83 - https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2020/TGRSHP2020_TechDoc_Ch3.pdf
# use ESRI:102003 for area calculations - https://epsg.io/102003
# use ESRI:102005 for length calculations - https://epsg.io/102005

proj_id = 'cmat-315920'
cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient   = bigquery.Client(credentials=cred, project=proj)
root_path  = '/home/jupyter'
root_path  = pathlib.Path(root_path)
data_path  = root_path / 'redistricting_data'
bq_dataset = proj_id   +'.redistricting_data'
rng = np.random.default_rng(seed)

def lower_cols(df):
    df.rename(columns = {x:str(x).lower() for x in df.columns}, inplace=True)
    return df

def extract_file(zipfile, fn, **kwargs):
    file = zipfile.extract(fn)
    return lower_cols(pd.read_csv(file, dtype=str, **kwargs))

def read_tbl(tbl, rows=1e15, start=0, cols='*'):
    query = f'select {cols} from {tbl} limit {rows}'
    if start is not None:
        query += f' offset {start}'
    try:
        return bqclient.query(query).result().to_dataframe()
    except:
        return None
        
def head(tbl, rows=10):
    return read_tbl(tbl, rows)

def load_table(tbl, df=None, file=None, query=None, overwrite=False, preview_rows=0):
    if overwrite:
        bqclient.delete_table(tbl, not_found_ok=True)
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

def fetch_zip(url, file):
    path = file.parent
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        zipfile = zf.ZipFile(file)
        print(f'zip already exists', end='')
    except:
        try:
            print(f'fetching zip from {url}', end = '')
            zipfile = zf.ZipFile(urllib.request.urlretrieve(url, file)[0])
        except urllib.error.HTTPError:
            print('\n\nFAILED - BAD URL\n\n')
            zipfile = None
    return zipfile

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
    return lower_cols(bqclient.query(query).result().to_dataframe())

def yr_to_congress(yr):
    return min(116, int(yr-1786)/2)

@dataclasses.dataclass
class Gerry:
    # These are default values that can be overridden when you create the object
    abbr              : str
    chunk_size        : int = 10000
#     geo_simplification: float = 0.003
    min_graph_degree  : int = 1
    pop_err_max_pct   : float = 2.0
    clr_seq           : typing.Any = tuple(px.colors.qualitative.Antique)
    
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__dict__[key] = val
        
    def __post_init__(self):
        self.__dict__.update(states[states['abbr']==self['abbr']].iloc[0])
        def rgb_to_hex(c):
            if c[0] == '#':
                return c
            else:
                return '#%02x%02x%02x' % tuple(int(rgb) for rgb in c[4:-1].split(', '))
        self['clr_seq'] = [rgb_to_hex(c) for c in self['clr_seq']]

    def table_id(self, variable, yr=2020):
        return f"{bq_dataset}.{variable}_{yr}_{self['abbr']}"

    def file_id(self, variable, yr=2020, suffix='zip'):
        tbl = self.table_id(variable, yr)
        f = tbl.split('.')[-1]
        return data_path / f"{f.replace('_','/')}/{f}.{suffix}"

    def get_assignments(self, yr=2020, overwrite=False):
        variable = 'assignments'
        tbl = self.table_id(variable, yr)
        if not overwrite and head(tbl) is not None:
            print('BigQuery table exists', end='')
        else:
            url = f"https://www2.census.gov/geo/docs/maps-data/data/baf"
            if yr == 2020:
                url += '2020'
            url += f"/BlockAssign_ST{self['fips']}_{self['abbr']}.zip"
            L = []
            zipfile = fetch_zip(url, self.file_id(variable, yr))
            for fn in zipfile.namelist():
                col = fn.lower().split('_')[-1][:-4]
                if fn[-3:] == 'txt' and col != 'aiannh':
                    if yr == 2020:
                        sep = '|'
                    else:
                        sep = ','
                    df = extract_file(zipfile, fn, sep=sep)
                    if col == 'vtd':
                        df['countyfp'] = df['countyfp'] + df['district'].str[-4:]
                        col = 'cntyvtd'
                    df = df.iloc[:,:2]
                    df.columns = [f'geoid_{yr}', f'{col}_{yr}']
                    L.append(df.set_index(f'geoid_{yr}'))
            df = pd.concat(L, axis=1).reset_index()
            load_table(tbl, df=df, overwrite=overwrite)

    def get_crosswalks(self, yr=2020, overwrite=False):
        variable = 'crosswalks'
        tbl = self.table_id(variable, yr)
        if not overwrite and head(tbl) is not None:
            print('BigQuery table exists', end='')
        else:
            url = f"https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/TAB2010_TAB2020_ST{self['fips']}.zip"
            zipfile = fetch_zip(url, self.file_id(variable, yr))
            for fn in zipfile.namelist():
                df = extract_file(zipfile, fn, sep='|')
                for y in [2010, 2020]:
                    df[f'geoid_{y}'] = df[f'state_{y}'].str.rjust(2,'0') + df[f'county_{y}'].str.rjust(3,'0') + df[f'tract_{y}'].str.rjust(6,'0') + df[f'blk_{y}'].str.rjust(4,'0')
            load_table(tbl, df=df, overwrite=overwrite)
                

    def get_elections(self, yr=2020, overwrite=False):
        if self['abbr'] != 'TX':
            print(f'elections only implemented for TX', end='')
            return
        variable = 'elections'
        tbl = self.table_id(variable, yr)
        if not overwrite and head(tbl) is not None:
            print(f'BigQuery table exists', end='')
        else:
            url = f'https://data.capitol.texas.gov/dataset/aab5e1e5-d585-4542-9ae8-1108f45fce5b/resource/253f5191-73f3-493a-9be3-9e8ba65053a2/download/{yr}-general-vtd-election-data.zip'
            L = []
            zipfile = fetch_zip(url, self.file_id(variable, yr))
            for fn in zipfile.namelist():
                w = fn.split('_')
                if w.pop(-1) == 'Returns.csv':
                    df = extract_file(zipfile, fn, sep=',')
                    df[f'cntyvtd_{yr}'] = df['cntyvtd'].str.rjust(7, '0')
                    df.drop(columns=['cntyvtd'], inplace=True)
                    df['yr'] = w.pop(0)
                    df['race'] = '_'.join(w)
                    L.append(df)
            df = pd.concat(L, axis=0, ignore_index=True).astype({'votes':int, 'yr':int})
            load_table(tbl, df=df, overwrite=overwrite)


    def get_census(self, yr=2020, overwrite=False):
        variable = 'census'
        tbl = self.table_id(variable, yr)
        if not overwrite and head(tbl) is not None:
            print(f'BigQuery table exists', end='')
        else:
            url = f"https://www2.census.gov/programs-surveys/decennial/{yr}/data/01-Redistricting_File--PL_94-171/{self['name'].replace(' ', '_')}/{self['abbr'].lower()}{yr}.pl.zip"
            zipfile = fetch_zip(url, self.file_id(variable, yr))
            for fn in zipfile.namelist():
                if fn[-3:] == '.pl':
                    print(f'{concat_str}{fn}', end='')
                    file = zipfile.extract(fn)
                    if fn[2:5] == 'geo':
                        geo_tbl  = tbl + 'geo'
                        temp_tbl = tbl + 'temp'
                        load_table(temp_tbl, file=file, overwrite=overwrite)
                        sel = [f'trim(substring(string_field_0, {s}, {w})) as {n}' for s, w, n in zip(census_columns['starts'], census_columns['widths'], census_columns['geo'])]
                        query = 'select\n\t' + ',\n\t'.join(sel) + '\nfrom\n\t' + temp_tbl
                        load_table(geo_tbl, query=query, overwrite=overwrite)
                        bqclient.delete_table(temp_tbl)
                    else:
                        i = fn[6]
                        if i in ['1', '2']:
                            cmd = 'sed -i "1s/^/' + ','.join(census_columns['joins'] + census_columns[i]) + '\\n/" ' + file
                            os.system(cmd)
                            load_table(tbl+i, file=file, overwrite=overwrite)

            print(f"{concat_str}joining", end='' )
            t = ',\n    '
            query = f"""
select
    concat(right(concat("00",C.state), 2), right(concat("000",C.county), 3), right(concat("000000",C.tract), 6), right(concat("0000",C.block), 4)) as geoid_{yr}
    , C.*
    , A.{f'{t}A.'.join(census_columns['1'])}
    , B.{f'{t}B.'.join(census_columns['2'])}
from
    {self.table_id(variable, yr)}1 as A
inner join
    {self.table_id(variable, yr)}2 as B
on
    A.fileid = B.fileid
    and A.stusab = B.stusab
    and A.logrecno = B.logrecno
inner join
    {self.table_id(variable, yr)}geo as C
on
    A.fileid = trim(C.fileid)
    and A.stusab = trim(C.stusab)
    and A.logrecno = cast(C.logrecno as int)
where
    C.block != ""
    """

            if yr < 2020:
                query = f"""
select
    E.geoid_2020
    , E.area_prop
    , D.*
from (
    {query}
    ) as D
inner join (
    select
        case when area_{yr} > 0.1 then area_int / area_{yr} else 0 end as area_prop
        , *
    from (
        select
            geoid_{yr}
            , geoid_2020
            , cast(arealand_int as int) as area_int
            , sum(cast(arealand_int as int)) over (partition by geoid_2010) as area_{yr}
        from
            {self.table_id('crosswalks', 2020)}
        )
    ) as E
on
    D.geoid_{yr} = E.geoid_{yr}
    """

                query = f"""
select
    geoid_2020
    , sum(area_prop) as area_prop
    , {t.join([f'max({c}) as {c}'             for c in census_columns['geo']])}
    , {t.join([f'sum(area_prop * {c}) as {c}' for c in census_columns['1'] + census_columns['2']])}
from (
    {query}
    )
group by
    1
    """

#             query = f"""
# select
#     case when cntyvtd_pop > 0 then total / cntyvtd_pop else 0 end as cntyvtd_pop_prop
#     , *
# from (
#     select
#         sum(total) over (partition by cntyvtd_2020) as cntyvtd_pop
#         , cntyvtd_2020
#         , F.*
#     from (
#         {query}
#         ) as F
#     inner join
#         {self.table_id('assignments', 2020)} as G
#     on
#         F.geoid_2020 = G.geoid_2020
#     )
#     """
            load_table(tbl, query=query, overwrite=overwrite)        


    def get_shapes(self, level='tabblock', yr=2020, overwrite=False):
        variable = f'shapes_{level}'
        tbl = self.table_id(variable, yr)
        temp_tbl = tbl + '_temp'
        if not overwrite and head(tbl) is not None:
            print('BigQuery table exists', end='')
        else:
            url = f"https://www2.census.gov/geo/tiger/TIGER{yr}/{level.upper()}"
            if yr == 2010:
                url += '/2010'
            elif yr == 2020 and level == 'tabblock':
                url += '20'
            url += f"/tl_{yr}_{self['fips']}_{level}{str(yr)[-2:]}"
            if yr == 2020 and level in ['tract', 'bg']:
                url = url[:-2]
            url += '.zip'
            file = self.file_id(variable, yr)
            path = file.parent
            zipfile = fetch_zip(url, file)
            zipfile.extractall(path)

            a = 0
            while True:
                df = lower_cols(gpd.read_file(path, rows=slice(a, a+self.chunk_size)))
                df.columns = [x[:-2] if x[-2:].isnumeric() else x for x in df.columns]
                df = df[['geoid', 'aland', 'awater', 'intptlon', 'intptlat', 'geometry']].rename(columns={'geoid':f'geoid_{yr}'})
                df['geometry'] = df['geometry'].apply(lambda p: orient(p, -1))
                load_table(temp_tbl, df=df.to_wkb(), overwrite=a==0)
                if df.shape[0] < self.chunk_size:
                    break
                else:
                    a += self.chunk_size

            query = f"""
select
    geoid_{yr}
    , aland
    , awater
    , area
    , abs(area - aland - awater) / area * 100 as area_err_pct
    , perim
    , case when perim > 0 then 4 * acos(-1) * aland / (perim * perim) else 0 end as polsby_popper
    , point
    , geography
   -- , st_simplify(geography, 10) as geography_simple
from (
    select
        *
        , st_perimeter(geography) as perim
        , st_area(geography) as area
    from (
        select
            *
            , st_geogpoint(cast(intptlon as float64), cast(intptlat as float64)) as point
            , st_geogfrom(geometry) as geography
        from
            {temp_tbl}
        )
    )
order by
    area_err_pct desc
    """
            load_table(tbl, query=query, overwrite=overwrite, preview_rows=0)
            bqclient.delete_table(temp_tbl)

    def get_pairs(self, level='tabblock', yr=2020, overwrite=False):
        variable = f'pairs_{level}'
        tbl = self.table_id(variable, yr)
        if not overwrite and head(tbl) is not None:
            print('BigQuery table exists', end='')
        else:
            shapes_tbl = self.table_id(f'shapes_{level}', yr)
            t = ',\n        '
            query = f"""            
select
    *
from (
    select
        x.geoid_{yr} as geoid_{yr}_x,
        y.geoid_{yr} as geoid_{yr}_y,
        st_distance(x.point, y.point) as distance,
        st_length(st_intersection(x.geography, y.geography)) as shared_perim
    from
        {shapes_tbl} as x,
        {shapes_tbl} as y
    where
        x.geoid_{yr} < y.geoid_{yr} and st_intersects(x.geography, y.geography)
    )
where shared_perim > 0.1
    """
#             print(query)
            load_table(tbl, query=query, overwrite=overwrite)
        self.pairs = read_table(tbl)
        print('success', end='')

    def get_data(self, overwrite=list()):
#         variable = 'crosswalks'
#         for yr in [2020]:
#             print(f"\nGet {self['name']} {variable.ljust(15, ' ')} {yr}", end=concat_str)
#             self[variable] = self.get_crosswalks(yr, variable in overwrite)

#         variable = 'assignments'
#         for yr in [2020, 2010]:
#             print(f"\nGet {self['name']} {variable.ljust(15, ' ')} {yr}", end=concat_str)
#             self[variable] = self.get_assignments(yr, variable in overwrite)
        
#         variable = 'elections'
#         for yr in [2020]:
#             print(f"\nGet {self['name']} {variable.ljust(15, ' ')} {yr}", end=concat_str)
#             self[variable] = self.get_elections(yr, variable in overwrite)
        
#         variable = 'census'
#         for yr in [2010]:
#             print(f"\nGet {self['name']} {variable.ljust(15, ' ')} {yr}", end=concat_str)
#             self[variable] = self.get_census(yr, variable in overwrite)

#         for level in ['tract', 'bg', 'tabblock']:
#             variable = f'shapes_{level}'
#             ovrwrt = variable in overwrite
#             for yr in [2020, 2010]:
#                 print(f"\nGet {self['name']} {variable.ljust(15, ' ')} {yr}", end=concat_str)
#                 self[variable] = self.get_shapes(level, yr, ovrwrt)

        for level in ['tract', 'bg', 'tabblock']:
            variable = f'pairs_{level}'
            ovrwrt = variable in overwrite
            for yr in [2020, 2010]:
                print(f"\nGet {self['name']} {variable.ljust(15, ' ')} {yr}", end=concat_str)
                self[variable] = self.get_pairs(level, yr, ovrwrt)

                

    def read_shape_pairs(self, level='tabblock', yr=2020, rows=None, start=None, cols='*'):
        tbl = self.table_id(f'shapes_{level}', yr)
        df = read_table(tbl, rows, start, cols)
        for c in ['centroid', 'geometry', 'geometry_simple']:
            if c in df.columns:
                df[c] = gpd.GeoSeries.from_wkb(df[c])
        return gpd.GeoDataFrame(df, geometry='geometry', crs=crs_map)

                
                
            
concat_str = ' ... '
census_columns = {
    'joins':  ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno'],

    'widths': [6, 2, 3, 2, 3, 2, 7, 1, 1, 2, 3, 2, 2, 5, 2, 2, 5, 2, 2, 6, 1, 4, 2, 5, 2, 2, 4, 5, 2, 1, 3, 5, 2, 6, 1, 5, 2, 5, 2, 5, 3, 5, 2, 5, 3, 1, 1, 5, 2, 1, 1, 2, 3, 3, 6, 1, 3, 5, 5, 2, 5, 5, 5, 14, 14, 90, 1, 1, 9, 9, 11, 12, 2, 1, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 5, 18],

    'geo': ['fileid', 'stusab', 'sumlev', 'geocomp', 'chariter', 'cifsn', 'logrecno', 'region', 'division', 'state', 'county', 'countycc', 'countysc', 'cousub', 'cousubcc', 'cousubsc', 'place', 'placecc', 'placesc', 'tract', 'blkgrp', 'block', 'iuc', 'concit', 'concitcc', 'concitsc', 'aianhh', 'aianhhfp', 'aianhhcc', 'aihhtli', 'aitsce', 'aits', 'aitscc', 'ttract', 'tblkgrp', 'anrc', 'anrccc', 'cbsa', 'cbsasc', 'metdiv', 'csa', 'necta', 'nectasc', 'nectadiv', 'cnecta', 'cbsapci', 'nectapci', 'ua', 'uasc', 'uatype', 'ur', 'cd', 'sldu', 'sldl', 'vtd', 'vtdi', 'reserve2', 'zcta5', 'submcd', 'submcdcc', 'sdelm', 'sdsec', 'sduni', 'arealand', 'areawatr', 'name', 'funcstat', 'gcuni', 'pop100', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'partflag', 'reserve3', 'uga', 'statens', 'countyns', 'cousubns', 'placens', 'concitns', 'aianhhns', 'aitsns', 'anrcns', 'submcdns', 'cd113', 'cd114', 'cd115', 'sldu2', 'sldu3', 'sldu4', 'sldl2', 'sldl3', 'sldl4', 'aianhhsc', 'csasc', 'cnectasc', 'memi', 'nmemi', 'puma', 'reserved'],
                  
    '1': ['total', 'population_of_one_race', 'white_alone', 'black_or_african_american_alone', 'american_indian_and_alaska_native_alone', 'asian_alone', 'native_hawaiian_and_other_pacific_islander_alone', 'some_other_race_alone', 'population_of_two_or_more_races', 'population_of_two_races', 'white_black_or_african_american', 'white_american_indian_and_alaska_native', 'white_asian', 'white_native_hawaiian_and_other_pacific_islander', 'white_some_other_race', 'black_or_african_american_american_indian_and_alaska_native', 'black_or_african_american_asian', 'black_or_african_american_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_some_other_race', 'american_indian_and_alaska_native_asian', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander', 'asian_some_other_race', 'native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_three_races', 'white_black_or_african_american_american_indian_and_alaska_native', 'white_black_or_african_american_asian', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_some_other_race', 'white_american_indian_and_alaska_native_asian', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander', 'white_asian_some_other_race', 'white_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_asian_some_other_race', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_asian_some_other_race', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_four_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_asian_some_other_race', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_asian_some_other_race', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_five_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_six_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'total_hl', 'hispanic_or_latino_hl', 'not_hispanic_or_latino_hl', 'population_of_one_race_hl', 'white_alone_hl', 'black_or_african_american_alone_hl', 'american_indian_and_alaska_native_alone_hl', 'asian_alone_hl', 'native_hawaiian_and_other_pacific_islander_alone_hl', 'some_other_race_alone_hl', 'population_of_two_or_more_races_hl', 'population_of_two_races_hl', 'white_black_or_african_american_hl', 'white_american_indian_and_alaska_native_hl', 'white_asian_hl', 'white_native_hawaiian_and_other_pacific_islander_hl', 'white_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_hl', 'black_or_african_american_asian_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_some_other_race_hl', 'american_indian_and_alaska_native_asian_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_hl', 'asian_some_other_race_hl', 'native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_three_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_hl', 'white_black_or_african_american_asian_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_asian_some_other_race_hl', 'white_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_asian_some_other_race_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_asian_some_other_race_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_four_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_asian_some_other_race_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_five_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_six_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl'],

    '2': ['total_18', 'population_of_one_race_18', 'white_alone_18', 'black_or_african_american_alone_18', 'american_indian_and_alaska_native_alone_18', 'asian_alone_18', 'native_hawaiian_and_other_pacific_islander_alone_18', 'some_other_race_alone_18', 'population_of_two_or_more_races_18', 'population_of_two_races_18', 'white__black_or_african_american_18', 'white__american_indian_and_alaska_native_18', 'white__asian_18', 'white__native_hawaiian_and_other_pacific_islander_18', 'white__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native_18', 'black_or_african_american__asian_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__some_other_race_18', 'american_indian_and_alaska_native__asian_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander_18', 'asian__some_other_race_18', 'native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_three_races_18', 'white__black_or_african_american__american_indian_and_alaska_native_18', 'white__black_or_african_american__asian_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__some_other_race_18', 'white__american_indian_and_alaska_native__asian_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander_18', 'white__asian__some_other_race_18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__asian__some_other_race_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__asian__some_other_race_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_four_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__asian__some_other_race_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__asian__some_other_race_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_five_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_six_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'total_hl18', 'hispanic_or_latino_hl18', 'not_hispanic_or_latino_hl18', 'population_of_one_race_hl18', 'white_alone_hl18', 'black_or_african_american_alone_hl18', 'american_indian_and_alaska_native_alone_hl18', 'asian_alone_hl18', 'native_hawaiian_and_other_pacific_islander_alone_hl18', 'some_other_race_alone_hl18', 'population_of_two_or_more_races_hl18', 'population_of_two_races_hl18', 'white__black_or_african_american_hl18', 'white__american_indian_and_alaska_native_hl18', 'white__asian_hl18', 'white__native_hawaiian_and_other_pacific_islander_hl18', 'white__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native_hl18', 'black_or_african_american__asian_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__some_other_race_hl18', 'american_indian_and_alaska_native__asian_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander_hl18', 'asian__some_other_race_hl18', 'native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_three_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native_hl18', 'white__black_or_african_american__asian_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__asian__some_other_race_hl18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__asian__some_other_race_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__asian__some_other_race_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_four_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__asian__some_other_race_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_five_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_six_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'housing_total', 'housing_occupied', 'housing_vacant'],
}

census_columns['starts'] = 1 + np.insert(np.cumsum(census_columns['widths'])[:-1], 0, 0)