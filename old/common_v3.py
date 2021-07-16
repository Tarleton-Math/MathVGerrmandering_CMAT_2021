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

def read_table(tbl, rows=99999999999, start=0, cols='*'):
    query = f'select {", ".join(cols)} from {tbl} limit {rows}'
    if start is not None:
        query += f' offset {start}'
    return bqclient.query(query).result().to_dataframe()

def head(tbl, rows=10):
    return read_table(tbl, rows)

def load_table(tbl, df=None, file=None, query=None, overwrite=True, preview_rows=0):
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
            print(f'fetching zip from {url} ', end = '')
            zipfile = zf.ZipFile(urllib.request.urlretrieve(url, file)[0])
            print(f'done', end = '')
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
    level             : str = 'tract'
    census_yr         : int = 2010
    shapes_yr         : int = 2020
    district          : str = 'cd'
    race              : str = 'General_Election'
    race_office       : str = 'U.S. Sen'
    race_yr           : int = 2018
    chunk_size        : int = 50000
    min_graph_degree  : int = 1
    pop_err_max_pct   : float = 2.0
    clr_seq           : typing.Any = tuple(px.colors.qualitative.Antique)
    
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__dict__[key] = val
        
    def __post_init__(self):
        levels = ['tract', 'bg', 'tabblock']
        assert self.level in levels, f"level must be one of {levels}, got {self.level}"
        district_types = ['cd', 'sldu', 'sldl']
        assert self.district in district_types, f"district must be one of {district_types}, got {self.district}"
        census_yrs = list(range(2010, 2021))
        assert self.census_yr in census_yrs, f"census_yr must be one of {census_yrs}, got {self.census_yr}"
        shapes_yrs = [2010, 2020]
        assert self.shapes_yr in shapes_yrs, f"census_yr must be one of {shapes_yrs}, got {self.shapes_yr}"
        
        self.__dict__.update(states[states['abbr']==self.abbr].iloc[0])
        def rgb_to_hex(c):
            if c[0] == '#':
                return c
            else:
                return '#%02x%02x%02x' % tuple(int(rgb) for rgb in c[4:-1].split(', '))
        self.clr_seq = [rgb_to_hex(c) for c in self.clr_seq]

    def table_id(self, variable, level, yr):
        tbl = f"{bq_dataset}.{variable}_{level}_{yr}_{self.abbr}"
        return tbl
    
    def file_id(self, variable=None, level=None, yr=None, tbl=None, suffix='zip'):
        if tbl is None:
            tbl = self.table_id(variable, level, yr)
        f = tbl.split('.')[-1]
        return data_path / f"{f.replace('_','/')}/{f}.{suffix}"


    def get_assignments(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        url = f"https://www2.census.gov/geo/docs/maps-data/data/baf"
        if self.shapes_yr == 2020:
            url += '2020'
        url += f"/BlockAssign_ST{self['fips']}_{self['abbr']}.zip"
        L = []
        zipfile = fetch_zip(url, self.file_id(tbl=tbl))
        for fn in zipfile.namelist():
            col = fn.lower().split('_')[-1][:-4]
            if fn[-3:] == 'txt' and col != 'aiannh':
                df = extract_file(zipfile, fn, sep='|')
                if col == 'vtd':
                    df['countyfp'] = df['countyfp'] + df['district'].str[-4:]
                    col = 'cntyvtd'
                df = df.iloc[:,:2]
                df.columns = [f'geoid_{self.shapes_yr}', f'{col}_{self.shapes_yr}']
                L.append(df.set_index(f'geoid_{self.shapes_yr}'))
        df = pd.concat(L, axis=1).reset_index()
        load_table(tbl, df=df)


    def get_crosswalks(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        url = f"https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/TAB2010_TAB2020_ST{self.fips}.zip"
        zipfile = fetch_zip(url, self.file_id(tbl=tbl))
        for fn in zipfile.namelist():
            df = extract_file(zipfile, fn, sep='|')
            for y in [2010, 2020]:
                df[f'geoid_{y}'] = df[f'state_{y}'].str.rjust(2,'0') + df[f'county_{y}'].str.rjust(3,'0') + df[f'tract_{y}'].str.rjust(6,'0') + df[f'blk_{y}'].str.rjust(4,'0')
        load_table(tbl, df=df)                


    def get_elections(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        if abbr != 'TX':
            print(f'elections only implemented for TX', end='')
            return
        url = f'https://data.capitol.texas.gov/dataset/aab5e1e5-d585-4542-9ae8-1108f45fce5b/resource/253f5191-73f3-493a-9be3-9e8ba65053a2/download/{self.shapes_yr}-general-vtd-election-data.zip'
        L = []
        zipfile = fetch_zip(url, self.file_id(tbl=tbl))
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
        load_table(tbl, df=df)


    def get_census(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        url = f"https://www2.census.gov/programs-surveys/decennial/{self.census_yr}/data/01-Redistricting_File--PL_94-171/{self.name.replace(' ', '_')}/{self.abbr.lower()}{yr}.pl.zip"
        zipfile = fetch_zip(url, self.file_id(tbl=tbl))
        for fn in zipfile.namelist():
            if fn[-3:] == '.pl':
                print(f'{concat_str}{fn}', end='')
                file = zipfile.extract(fn)
                if fn[2:5] == 'geo':
                    geo_tbl  = tbl + 'geo'
                    temp_tbl = tbl + 'temp'
                    load_table(temp_tbl, file=file)
                    sel = [f'trim(substring(string_field_0, {s}, {w})) as {n}' for s, w, n in zip(census_columns['starts'], census_columns['widths'], census_columns['geo'])]
                    query = 'select\n\t' + ',\n\t'.join(sel) + '\nfrom\n\t' + temp_tbl
                    load_table(geo_tbl, query=query)
                    bqclient.delete_table(temp_tbl)
                else:
                    i = fn[6]
                    if i in ['1', '2']:
                        cmd = 'sed -i "1s/^/' + ','.join(census_columns['joins'] + census_columns[i]) + '\\n/" ' + file
                        os.system(cmd)
                        load_table(tbl+i, file=file)
        print(f"{concat_str}joining", end='' )
        t = ',\n    '
        query = f"""
select
    concat(right(concat("00",C.state), 2), right(concat("000",C.county), 3), right(concat("000000",C.tract), 6), right(concat("0000",C.block), 4)) as geoid_{self.shapes_yr},
    C.*,
    A.{f'{t}A.'.join(census_columns['1'])},
    B.{f'{t}B.'.join(census_columns['2'])}
from
    {tbl}1 as A
inner join
    {tbl}2 as B
on
    A.fileid = B.fileid
    and A.stusab = B.stusab
    and A.logrecno = B.logrecno
inner join
    {geo_tbl} as C
on
    A.fileid = trim(C.fileid)
    and A.stusab = trim(C.stusab)
    and A.logrecno = cast(C.logrecno as int)
where
    C.block != ""
    """

        if self.census_yr == 2010:
            query = f"""
select
    E.area_prop,
    D.*
from (
    {query}
    ) as D
inner join (
    select
        case when area_{self.census_yr} > 0.1 then area_int / area_{self.census_yr} else 0 end as area_prop,
        *
    from (
        select
            geoid_{self.census_yr},
            geoid_{self.shapes_yr},
            cast(arealand_int as int) as area_int,
            sum(cast(arealand_int as int)) over (partition by geoid_{self.census_yr}) as area_{self.census_yr}
        from
            {self.table_id('crosswalks', 'tabblock', self.shapes_yr)}
        )
    ) as E
on
    D.geoid_{self.shapes_yr} = E.geoid_{self.shapes_yr}
    """

            query = f"""
select
    geoid_{self.shapes_yr},
    sum(area_prop) as area_prop,
    {t.join([f'max({c}) as {c}'             for c in census_columns['geo']])},
    {t.join([f'sum(area_prop * {c}) as {c}' for c in census_columns['1'] + census_columns['2']])}
from (
    {query}
    )
group by
    1
    """

        query = f"""
select
    case when cntyvtd_pop > 0 then total / cntyvtd_pop else 1 / cntyvtd_blocks end as cntyvtd_pop_prop,
    *
from (
    select
        sum(total) over (partition by cntyvtd_{self.shapes_yr}) as cntyvtd_pop,
        count(*) over (partition by cntyvtd_{self.shapes_yr}) as cntyvtd_blocks,
        cntyvtd_{self.shapes_yr},
        F.*
    from (
        {query}
        ) as F
    inner join
        {self.table_id('assignments', 'tabblock', self.shapes_yr)} as G
    on
        F.geoid_{self.shapes_yr} = G.geoid_{self.shapes_yr}
    )
    """
        load_table(tbl, query=query)
        bqclient.delete_table(geo_tbl)
        bqclient.delete_table(tbl+'1')
        bqclient.delete_table(tbl+'2')


    def get_shapes(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        temp_tbl = tbl + '_temp'
        url = f"https://www2.census.gov/geo/tiger/TIGER{self.shapes_yr}/{self.level.upper()}"
        if self.shapes_yr == 2010:
            url += '/2010'
        elif self.shapes_yr == 2020 and self.level == 'tabblock':
            url += '20'
        url += f"/tl_{self.shapes_yr}_{self.fips}_{self.level}{str(self.shapes_yr)[-2:]}"
        if self.shapes_yr == 2020 and self.level in ['tract', 'bg']:
            url = url[:-2]
        url += '.zip'

        file = self.file_id(tbl=tbl)
        path = file.parent
        zipfile = fetch_zip(url, file)
        zipfile.extractall(path)

        a = 0
        while True:
            print(f'Starting row {a}')
            df = lower_cols(gpd.read_file(path, rows=slice(a, a+self.chunk_size)))
            df.columns = [x[:-2] if x[-2:].isnumeric() else x for x in df.columns]
            df = df[['geoid', 'aland', 'awater', 'intptlon', 'intptlat', 'geometry']].rename(columns={'geoid':f'geoid_{self.shapes_yr}'})
            df['geometry'] = df['geometry'].apply(lambda p: orient(p, -1))
            load_table(temp_tbl, df=df.to_wkb(), overwrite=a==0)
            if df.shape[0] < self.chunk_size:
                break
            else:
                a += self.chunk_size

        query = f"""
select
    geoid_{self.shapes_yr},
    aland,
    awater,
    perim,
    case when perim > 0 then 4 * acos(-1) * aland / (perim * perim) else 0 end as polsby_popper,
    point,
    geography
from (
    select
        *,
        st_perimeter(geography) as perim
    from (
        select
            *,
            st_geogpoint(cast(intptlon as float64), cast(intptlat as float64)) as point,
            st_geogfrom(geometry) as geography
        from
            {temp_tbl}
        )
    )
order by
    geoid_{self.shapes_yr}
    """
        load_table(tbl, query=query)
        bqclient.delete_table(temp_tbl)


    def get_edges(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        shapes_tbl = self.table_id('shapes', level, yr)
        t = ',\n        '
        query = f"""
select
    *
from (
    select
        x.geoid_{self.shapes_yr} as geoid_{self.shapes_yr}_x,
        y.geoid_{self.shapes_yr} as geoid_{self.shapes_yr}_y,
        st_distance(x.point, y.point) as distance,
        st_length(st_intersection(x.geography, y.geography)) as shared_perim
    from
        {shapes_tbl} as x,
        {shapes_tbl} as y
    where
        x.geoid_{self.shapes_yr} < y.geoid_{self.shapes_yr} and st_intersects(x.geography, y.geography)
    )
where shared_perim > 0.1
    """
        load_table(tbl, query=query)
        print('success', end='')


    def get_nodes(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        if abbr != 'TX':
            print(f'elections only implemented for TX', end='')
            return
        
        query = f"""
select
    A.geoid_{self.shapes_yr},
    B.party,
    B.votes * A.cntyvtd_pop_prop as votes,
    A.total as pop_total
from
    {self.table_id('census', 'tabblock', self.census_yr)} as A
inner join (
    select
        *
    from
        {self.table_id('elections', 'cntyvtd', self.shapes_yr)}
    where
        race = "{self.race}" and
        office = "{self.race_office}" and
        yr = {self.race_yr}
    ) as B
on
    A.cntyvtd_{self.shapes_yr} = B.cntyvtd_{self.shapes_yr}
    """

        if self.level != 'tabblock':
            g = 12 if self.level == 'bg' else 11
            query = f"""
select
    substring(geoid_{self.shapes_yr}, 1, {g}) as geoid_{self.shapes_yr},
    party,
    sum(votes) as votes,
    sum(pop_total) as pop_total
from (
    {query}
    )
group by
    1, 2
   """
        query = f"""
select
    A.*,
    B.aland as area,
    B.perim
from (
    {query}
    ) as A
inner join
    {self.table_id('shapes', self.level, self.shapes_yr)} as B
on
    A.geoid_{self.shapes_yr} = B.geoid_{self.shapes_yr}
    """
        N = bqclient.query(query).result().to_dataframe()
        i = N.columns.drop(['party', 'votes', 'candidate', 'office', 'yr'], errors='ignore')
        N = N.pivot(index=i, columns='party', values='votes').reset_index()
        load_table(tbl, df=N)
        print('success', end='')


    def get_data(self, overwrite=list()):
        for variable, level, yr in [
            ('crosswalks', 'tabblock', self.shapes_yr),
            ('assignments', 'tabblock', self.shapes_yr),
            ('elections', 'cntyvtd', self.shapes_yr),
            ('census', 'tabblock', self.census_yr),
            ('shapes', self.level, self.shapes_yr),
            ('edges', self.level, self.shapes_yr),
            ('nodes', self.level, self.shapes_yr),
        ]:
            print(f"\nGet {self.name} " + f"{variable} {level} {yr}".ljust(25, ' '), end=concat_str)
            tbl = self.table_id(variable, level, yr)
            try:
                assert variable not in overwrite
                bqclient.get_table(tbl)
                print('BigQuery table exists', end='')
            except:
                getattr(self, f'get_{variable}')(tbl)


    def make_graph(self):
        self.nodes = read_table(self.table_id('nodes', self.level, self.shapes_yr))
        self.edges = read_table(self.table_id('edges', self.level, self.shapes_yr))
        self.graph = nx.from_pandas_edgelist(self.edges, source=f'geoid_{self.shapes_yr}_x', target=f'geoid_{self.shapes_yr}_y', edge_attr=['distance', 'shared_perim'])
        nx.set_node_attributes(self.graph, self.nodes.set_index(f'geoid_{self.shapes_yr}').to_dict('index'))



            
concat_str = ' ... '
census_columns = {
    'joins':  ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno'],

    'widths': [6, 2, 3, 2, 3, 2, 7, 1, 1, 2, 3, 2, 2, 5, 2, 2, 5, 2, 2, 6, 1, 4, 2, 5, 2, 2, 4, 5, 2, 1, 3, 5, 2, 6, 1, 5, 2, 5, 2, 5, 3, 5, 2, 5, 3, 1, 1, 5, 2, 1, 1, 2, 3, 3, 6, 1, 3, 5, 5, 2, 5, 5, 5, 14, 14, 90, 1, 1, 9, 9, 11, 12, 2, 1, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 5, 18],

    'geo': ['fileid', 'stusab', 'sumlev', 'geocomp', 'chariter', 'cifsn', 'logrecno', 'region', 'division', 'state', 'county', 'countycc', 'countysc', 'cousub', 'cousubcc', 'cousubsc', 'place', 'placecc', 'placesc', 'tract', 'blkgrp', 'block', 'iuc', 'concit', 'concitcc', 'concitsc', 'aianhh', 'aianhhfp', 'aianhhcc', 'aihhtli', 'aitsce', 'aits', 'aitscc', 'ttract', 'tblkgrp', 'anrc', 'anrccc', 'cbsa', 'cbsasc', 'metdiv', 'csa', 'necta', 'nectasc', 'nectadiv', 'cnecta', 'cbsapci', 'nectapci', 'ua', 'uasc', 'uatype', 'ur', 'cd', 'sldu', 'sldl', 'vtd', 'vtdi', 'reserve2', 'zcta5', 'submcd', 'submcdcc', 'sdelm', 'sdsec', 'sduni', 'arealand', 'areawatr', 'name', 'funcstat', 'gcuni', 'pop100', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'partflag', 'reserve3', 'uga', 'statens', 'countyns', 'cousubns', 'placens', 'concitns', 'aianhhns', 'aitsns', 'anrcns', 'submcdns', 'cd113', 'cd114', 'cd115', 'sldu2', 'sldu3', 'sldu4', 'sldl2', 'sldl3', 'sldl4', 'aianhhsc', 'csasc', 'cnectasc', 'memi', 'nmemi', 'puma', 'reserved'],
                  
    '1': ['total', 'population_of_one_race', 'white_alone', 'black_or_african_american_alone', 'american_indian_and_alaska_native_alone', 'asian_alone', 'native_hawaiian_and_other_pacific_islander_alone', 'some_other_race_alone', 'population_of_two_or_more_races', 'population_of_two_races', 'white_black_or_african_american', 'white_american_indian_and_alaska_native', 'white_asian', 'white_native_hawaiian_and_other_pacific_islander', 'white_some_other_race', 'black_or_african_american_american_indian_and_alaska_native', 'black_or_african_american_asian', 'black_or_african_american_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_some_other_race', 'american_indian_and_alaska_native_asian', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander', 'asian_some_other_race', 'native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_three_races', 'white_black_or_african_american_american_indian_and_alaska_native', 'white_black_or_african_american_asian', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_some_other_race', 'white_american_indian_and_alaska_native_asian', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander', 'white_asian_some_other_race', 'white_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_asian_some_other_race', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_asian_some_other_race', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_four_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_asian_some_other_race', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_asian_some_other_race', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_five_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_six_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'total_hl', 'hispanic_or_latino_hl', 'not_hispanic_or_latino_hl', 'population_of_one_race_hl', 'white_alone_hl', 'black_or_african_american_alone_hl', 'american_indian_and_alaska_native_alone_hl', 'asian_alone_hl', 'native_hawaiian_and_other_pacific_islander_alone_hl', 'some_other_race_alone_hl', 'population_of_two_or_more_races_hl', 'population_of_two_races_hl', 'white_black_or_african_american_hl', 'white_american_indian_and_alaska_native_hl', 'white_asian_hl', 'white_native_hawaiian_and_other_pacific_islander_hl', 'white_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_hl', 'black_or_african_american_asian_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_some_other_race_hl', 'american_indian_and_alaska_native_asian_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_hl', 'asian_some_other_race_hl', 'native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_three_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_hl', 'white_black_or_african_american_asian_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_asian_some_other_race_hl', 'white_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_asian_some_other_race_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_asian_some_other_race_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_four_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_asian_some_other_race_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_five_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_six_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl'],

    '2': ['total_18', 'population_of_one_race_18', 'white_alone_18', 'black_or_african_american_alone_18', 'american_indian_and_alaska_native_alone_18', 'asian_alone_18', 'native_hawaiian_and_other_pacific_islander_alone_18', 'some_other_race_alone_18', 'population_of_two_or_more_races_18', 'population_of_two_races_18', 'white__black_or_african_american_18', 'white__american_indian_and_alaska_native_18', 'white__asian_18', 'white__native_hawaiian_and_other_pacific_islander_18', 'white__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native_18', 'black_or_african_american__asian_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__some_other_race_18', 'american_indian_and_alaska_native__asian_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander_18', 'asian__some_other_race_18', 'native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_three_races_18', 'white__black_or_african_american__american_indian_and_alaska_native_18', 'white__black_or_african_american__asian_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__some_other_race_18', 'white__american_indian_and_alaska_native__asian_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander_18', 'white__asian__some_other_race_18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__asian__some_other_race_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__asian__some_other_race_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_four_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__asian__some_other_race_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__asian__some_other_race_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_five_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_six_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'total_hl18', 'hispanic_or_latino_hl18', 'not_hispanic_or_latino_hl18', 'population_of_one_race_hl18', 'white_alone_hl18', 'black_or_african_american_alone_hl18', 'american_indian_and_alaska_native_alone_hl18', 'asian_alone_hl18', 'native_hawaiian_and_other_pacific_islander_alone_hl18', 'some_other_race_alone_hl18', 'population_of_two_or_more_races_hl18', 'population_of_two_races_hl18', 'white__black_or_african_american_hl18', 'white__american_indian_and_alaska_native_hl18', 'white__asian_hl18', 'white__native_hawaiian_and_other_pacific_islander_hl18', 'white__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native_hl18', 'black_or_african_american__asian_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__some_other_race_hl18', 'american_indian_and_alaska_native__asian_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander_hl18', 'asian__some_other_race_hl18', 'native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_three_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native_hl18', 'white__black_or_african_american__asian_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__asian__some_other_race_hl18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__asian__some_other_race_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__asian__some_other_race_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_four_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__asian__some_other_race_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_five_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_six_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'housing_total', 'housing_occupied', 'housing_vacant'],
}

census_columns['starts'] = 1 + np.insert(np.cumsum(census_columns['widths'])[:-1], 0, 0)