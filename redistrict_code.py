import google, time, datetime, dataclasses, typing, os, pathlib, shutil, urllib, zipfile as zf, numpy as np, pandas as pd, geopandas as gpd, networkx as nx
import matplotlib.pyplot as plt, plotly.express as px
from pprint import pprint
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

cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient   = bigquery.Client(credentials=cred, project=proj)
root_path  = pathlib.Path(root_path)
data_path  = root_path / 'redistricting_data'
bq_dataset = proj_id   +'.redistricting_data'
rng = np.random.default_rng(seed)

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

def extract_file(zipfile, fn, **kwargs):
    file = zipfile.extract(fn)
    return lower(pd.read_csv(file, dtype=str, **kwargs))

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

def downcast(df, exclude=[]):
    for c in df.columns:
        if c not in exclude:
            df[c] = pd.to_numeric(df[c], downcast='unsigned')
    return df

def fetch_zip(url, file):
    path = file.parent
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        zipfile = zf.ZipFile(file)
        print(f'zip already exists{concat_str}processing', end=concat_str)
    except:
        try:
            print(f'fetching zip from {url}', end=concat_str)
            zipfile = zf.ZipFile(urllib.request.urlretrieve(url, file)[0])
            print(f'processing', end=concat_str)
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
    election_filters  : typing.Tuple = (
        "race='general' and office='u.s. sen'",
        "race='general' and office='president'",
        "race='general' and office like 'u.s. rep%'")
    overwrite         : typing.Tuple = ()
    chunk_size        : int = 50000
    pop_err_max_pct   : float = 10.0
    clr_seq           : typing.Tuple = tuple(px.colors.qualitative.Antique)
    
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__dict__[key] = val
        
    def __post_init__(self):
#         self.overwrite = set(self.overwrite).union({'nodes'})
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
        url += f"/BlockAssign_ST{self.fips}_{self.abbr.upper()}.zip"
        L = []
        zipfile = fetch_zip(url, self.file_id(tbl=tbl))
        for fn in zipfile.namelist():
            col = fn.lower().split('_')[-1][:-4]
            if fn[-3:] == 'txt' and col != 'aiannh':
                df = extract_file(zipfile, fn, sep='|')
                if col == 'vtd':
                    df['countyfp'] = df['countyfp'].str.rjust(3, '0') + df['district'].str.rjust(6, '0')
                    col = 'cntyvtd'
                df = df.iloc[:,:2]
                df.columns = [f'geoid_{self.shapes_yr}', f'{col}_{self.shapes_yr}']
                L.append(df.set_index(f'geoid_{self.shapes_yr}'))
        df = pd.concat(L, axis=1).reset_index()
#         self.assignments = df.copy()
        load_table(tbl, df=df)


    def get_crosswalks(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        url = f"https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/TAB2010_TAB2020_ST{self.fips}.zip"
        zipfile = fetch_zip(url, self.file_id(tbl=tbl))
        for fn in zipfile.namelist():
            df = extract_file(zipfile, fn, sep='|')
            for y in [2010, 2020]:
                df[f'geoid_{y}'] = df[f'state_{y}'].str.rjust(2,'0') + df[f'county_{y}'].str.rjust(3,'0') + df[f'tract_{y}'].str.rjust(6,'0') + df[f'blk_{y}'].str.rjust(4,'0')
#         self.crosswalks = df.copy()
        load_table(tbl, df=df)


    def get_elections(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        if abbr != 'TX':
            print(f'elections only implemented for TX', end=concat_str)
            return
        url = f'https://data.capitol.texas.gov/dataset/aab5e1e5-d585-4542-9ae8-1108f45fce5b/resource/253f5191-73f3-493a-9be3-9e8ba65053a2/download/{self.shapes_yr}-general-vtd-election-data.zip'
        L = []
        zipfile = fetch_zip(url, self.file_id(tbl=tbl))
        for fn in zipfile.namelist():
            w = fn.lower().split('_')
            if w.pop(-1) == 'returns.csv':
                df = extract_file(zipfile, fn, sep=',')
                df['yr'] = int(w.pop(0))
                w.pop(-1)
                df['race'] = '_'.join(w)
                L.append(df)
#                 display(df.head(3))
        df = (pd.concat(L, axis=0, ignore_index=True)
              .astype({'votes':int, 'yr':int, 'fips':str, 'vtd':str})
              .query('votes > 0')
              .query("party in ['r', 'd', 'l', 'g']")
              .reset_index()
             )
        
        c = f'cntyvtd_{self.shapes_yr}'
        df[c] = df['fips'].str.rjust(3, '0') + df['vtd'].str.rjust(6, '0')
        df['alt'] = df['fips'].str.rjust(3, '0') + df['vtd'].str[:-1].str.rjust(6, '0')
        assign = read_table(self.table_id('assignments', 'tabblock', self.shapes_yr))[c].drop_duplicates()
        unmatched = ~df[c].isin(assign)
        df.loc[unmatched, c] = df.loc[unmatched, 'alt']
        unmatched = ~df[c].isin(assign)
        display(df[unmatched].sort_values('votes', ascending=False))
        df.drop(columns=['alt'], inplace=True)
#         self.elections = df.copy()
        load_table(tbl, df=df)


    def get_census(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        url = f"https://www2.census.gov/programs-surveys/decennial/{self.census_yr}/data/01-Redistricting_File--PL_94-171/{self.name.replace(' ', '_')}/{self.abbr.lower()}{yr}.pl.zip"

        zipfile = fetch_zip(url, self.file_id(tbl=tbl))
        for fn in zipfile.namelist():
            if fn[-3:] == '.pl':
                print(fn, end=concat_str)
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
        print('joining', end=concat_str)
        t = ',\n    '
        query = f"""
select
    concat(right(concat("00",C.state), 2), right(concat("000",C.county), 3), right(concat("000000",C.tract), 6), right(concat("0000",C.block), 4)) as geoid_{self.census_yr},
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
#         load_table(tbl+'merge', query=query)
        
        if self.census_yr == 2010:
            query = f"""
select
    E.area_prop,
    E.geoid_{self.shapes_yr},
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
    D.geoid_{self.census_yr} = E.geoid_{self.census_yr}
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
            print(f'starting row {a}', end=concat_str)
            df = lower(gpd.read_file(path, rows=slice(a, a+self.chunk_size)))
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


    def get_nodes(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        geoid = f'geoid_{self.shapes_yr}'
        cntyvtd = f'cntyvtd_{self.shapes_yr}'
        
        if self.level == 'tract':
            g = 11
        elif self.level == 'bg':
            g = 12
        else:
            g = 15
        
        query = f"""
select
    substring(A.{geoid}, 1, {g}) as {geoid},
    A.{geoid} as {geoid}_tabblock,
    B.{cntyvtd},
    A.cntyvtd_pop_prop,
    cast(B.cd_{self.shapes_yr} as int) as cd,
    cast(B.sldu_{self.shapes_yr} as int) as sldu,
    cast(B.sldl_{self.shapes_yr} as int) as sldl,
    A.total as pop_total
from
    {self.table_id('census', 'tabblock', self.census_yr)} as A
inner join
    {self.table_id('assignments', 'tabblock', self.shapes_yr)} as B
on
    A.{geoid} = B.{geoid}
    """
        
        if g < 15:
            query = f"""
select
    {geoid},
    {geoid}_tabblock,
    {cntyvtd},
    cntyvtd_pop_prop,
    min(cd) over (partition by {geoid}) as cd,
    min(sldu) over (partition by {geoid}) as sldu,
    min(sldl) over (partition by {geoid}) as sldl,
    sum(pop_total) over (partition by {geoid}) as pop_total 
from (
    {query}
    )
    """

        if abbr != 'TX':
            query = f"""
select
    C.*,
    's' as party,
    0 as votes
from (
    {query}
    ) as C
    """
        else:
            t = ' or\n        '
            query = f"""
select
    C.*,
    D.party as party,
    D.votes * C.cntyvtd_pop_prop as votes
from (
    {query}
    ) as C
left join (
    select
        *
    from
        {self.table_id('elections', 'cntyvtd', self.shapes_yr)}
    where
        {t.join(f'({x})' for x in self.election_filters)}
    ) as D
on
    C.{cntyvtd} = D.{cntyvtd}
    """

        query = f"""
select
    E.*,
    F.aland as area,
    F.perim,
from (
    select
        {geoid},
        party,
        min(cd) as cd,
        min(sldu) as sldu,
        min(sldl) as sldl,
        min(pop_total) as pop_total,
        sum(votes) as votes,
    from (
        {query}
        )
    group by
        1, 2
    ) as E
inner join
    {self.table_id('shapes', self.level, self.shapes_yr)} as F
on
    E.{geoid} = F.{geoid}
    """
#         print(query)
        df = bqclient.query(query).result().to_dataframe().rename(columns={geoid:'geoid'})
#         display(df.dtypes)
#         mask = df[['cd','sldu','sldl', 'area']].isnull().any(axis=1)
#         display(df[mask])
        df['party'].fillna('r', inplace=True)
        df['votes'].fillna(0, inplace=True)
        i = df.columns.drop(['party', 'votes', 'candidate', 'office', 'yr'], errors='ignore').to_list()
        df = df.pivot_table(index=i, columns='party', values='votes', fill_value=0)
        df['votes_total'] = df.sum(axis=1)
        df.columns.name = None
        self.nodes = df.reset_index()
#         display(self.nodes.head(5))
#         assert 1==2
        load_table(tbl, df=self.nodes, preview_rows=0)
#         self.nodes = df.copy()


    def get_data(self):
        for variable, level, yr in [
            ('crosswalks', 'tabblock', self.shapes_yr),
            ('assignments', 'tabblock', self.shapes_yr),
            ('elections', 'cntyvtd', self.shapes_yr),
            ('census', 'tabblock', self.census_yr),
            ('shapes', self.level, self.shapes_yr),
            ('edges', self.level, self.shapes_yr),
            ('nodes', self.level, self.shapes_yr),
            ('graph', self.level, self.shapes_yr),
        ]:
            tbl = self.table_id(variable, level, yr)            
            msg = f"Get {self.name} " + f"{variable} {level} {yr} "
            if variable == 'graph':
                msg += self.district
            print(msg.ljust(44, ' '), end=concat_str)
            try:
                assert variable not in self.overwrite
                bqclient.get_table(tbl)
                print('BigQuery table exists', end=concat_str)
            except:
                getattr(self, f'get_{variable}')(tbl)
            print('success \n-----------------------------------------------------------------------------------')

    def edges_to_graph(self, edges):
        g = f'geoid_{self.shapes_yr}'
        edge_attr = ['distance', 'shared_perim']
        return nx.from_pandas_edgelist(edges, source=f'{g}_x', target=f'{g}_y', edge_attr=edge_attr)

    def get_graph(self, tbl):
        variable, level, yr, abbr = tbl.split('.')[-1].split('_')
        g = f'geoid_{yr}'
        try:
            self.nodes
        except:
            nodes_tbl = self.table_id('nodes', self.level, yr)
            try:
                self.nodes = read_table(nodes_tbl)
            except:
                self.get_nodes(nodes_tbl)
        self.nodes.set_index('geoid', inplace=True)
                
        file = self.file_id(tbl=tbl+'_'+self.district, suffix='gpickle')
        if variable in self.overwrite and file.is_file():
            file.unlink()
        try:
            self.graph = nx.read_gpickle(file)
            print('pickle file already exists', end=concat_str)
        except:
            print('making graph', end=concat_str)
            self.edges = read_table(self.table_id('edges', self.level, yr))
            self.graph = self.edges_to_graph(self.edges)
            print('connecting districts', end=concat_str)
            shapes_tbl = self.table_id('shapes', self.level, yr)
            for dist, nodes in self.nodes.groupby(self.district):
                while True:
                    H = self.graph.subgraph(nodes.index)
                    components = sorted([list(c) for c in nx.connected_components(H)], key=lambda x:len(x), reverse=True)
                    if len(components) == 1:
                        break
                    print(f'\n{self.district.upper()} district {str(dist).rjust(3, " ")} has {str(len(components)).rjust(3, " ")} connected components with {[len(c) for c in components]} nodes ... adding edges to connect', end=concat_str)
                    c = ["', '".join(components[i]) for i in range(2)]
                    query = f"""
select
    {g}_x,
    {g}_y,
    distance,
    0.0 as shared_perim
from (
    select
        *,
        min(distance) over () as m
    from (
        select
            A.{g} as {g}_x,
            B.{g} as {g}_y,
            st_distance(A.point, B.point) as distance
        from
            {shapes_tbl} as A,
            {shapes_tbl} as B
        where
            A.{g} in ('{c[0]}') and B.{g} in ('{c[1]}')
        )
    )
where distance < 1.05 * m
"""
                    new_edges = bqclient.query(query).result().to_dataframe()
                    self.graph.update(self.edges_to_graph(new_edges))
                    print('done', end='', flush=True)
            file.parent.mkdir(parents=True, exist_ok=True)
            nx.write_gpickle(self.graph, file)
#         drop_districts = {'cd', 'sldu', 'sldl'}.difference({self.district})
#         self.nodes = self.nodes.drop(columns=drop_districts).rename(columns={self.district:'district', g:'geoid'})
        nx.set_node_attributes(self.graph, self.nodes.to_dict('index'))


        
#     def check_pop_balance(self, T):
        

#         comp = nx.connected_components(T)
#         next(comp)
#         s = sum(T.nodes[n]['total_pop'] for n in next(comp))
#         return abs(s - self.pop_target) <= self.pop_err_max
        
#     def recom_step(self):
#         try:
#             return self.graph
#         except:
#             tbl = self.table_id('graph', self.level, self.shapes_yr)
#             self.get_graph(tbl)
        
        
        
#         self.nodes.set_index('geoid', inplace=True)
#         recom_found = False
#         attempts = 0
#         D = self.nodes.groupby('district')
#         for curr in rng.permutation([(a,b) for a in self.cd_names for b in self.cd_names if a < b]).tolist():
#             H = self.graph.subgraph(nodes.index)
            
#             nodes = self.nodes.query(f'district in {curr}').copy()
            
#             trees = []
#             for i in range(1000):
#                 w = {e: self.rng.uniform() for e in H.edges}
#                 nx.set_edge_attributes(H, w, "weight")
#                 T = nx.minimum_spanning_tree(H, "weight")
#                 h = hash(tuple(sorted(T.edges)))
#                 if h not in trees:
#                     trees.append(h)
#                     for e in self['rng'].permutation(T.edges):
#                         attempts += 1
#                         T.remove_edge(*e)
#                         if self.check_pop_balance(T):
#                             recom_found = True
#                             new = [list(c) for c in nx.connected_components(T)]
#                             nodes['cd_new'] = 0
#                             for n, c in zip(new, curr):
#                                 nodes.loc[n, 'cd_new'] = c
#                             i = nodes.groupby(['cd','cd_new'])['area'].sum().idxmax()
#                             if i[0] != i[1]:
#                                 new[0], new[1] = new[1], new[0]
#                             for n, c in zip(new, curr):
#                                 self['bgs'].loc[n, 'cd'] = c
#                             break
#                         T.add_edge(*e)
#                 else:
#                     print('Got a repeat spanning tree')
#                 if recom_found:
#                     break
#             if recom_found:
#                 break
#         self['bgs'].reset_index(inplace=True)
#         assert recom_found, "No suitable recomb step found"
#         return recom_found, attempts, trees
        
            
concat_str = ' ... '
census_columns = {
    'joins':  ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno'],

    'widths': [6, 2, 3, 2, 3, 2, 7, 1, 1, 2, 3, 2, 2, 5, 2, 2, 5, 2, 2, 6, 1, 4, 2, 5, 2, 2, 4, 5, 2, 1, 3, 5, 2, 6, 1, 5, 2, 5, 2, 5, 3, 5, 2, 5, 3, 1, 1, 5, 2, 1, 1, 2, 3, 3, 6, 1, 3, 5, 5, 2, 5, 5, 5, 14, 14, 90, 1, 1, 9, 9, 11, 12, 2, 1, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 5, 18],

    'geo': ['fileid', 'stusab', 'sumlev', 'geocomp', 'chariter', 'cifsn', 'logrecno', 'region', 'division', 'state', 'county', 'countycc', 'countysc', 'cousub', 'cousubcc', 'cousubsc', 'place', 'placecc', 'placesc', 'tract', 'blkgrp', 'block', 'iuc', 'concit', 'concitcc', 'concitsc', 'aianhh', 'aianhhfp', 'aianhhcc', 'aihhtli', 'aitsce', 'aits', 'aitscc', 'ttract', 'tblkgrp', 'anrc', 'anrccc', 'cbsa', 'cbsasc', 'metdiv', 'csa', 'necta', 'nectasc', 'nectadiv', 'cnecta', 'cbsapci', 'nectapci', 'ua', 'uasc', 'uatype', 'ur', 'cd', 'sldu', 'sldl', 'vtd', 'vtdi', 'reserve2', 'zcta5', 'submcd', 'submcdcc', 'sdelm', 'sdsec', 'sduni', 'arealand', 'areawatr', 'name', 'funcstat', 'gcuni', 'pop100', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'partflag', 'reserve3', 'uga', 'statens', 'countyns', 'cousubns', 'placens', 'concitns', 'aianhhns', 'aitsns', 'anrcns', 'submcdns', 'cd113', 'cd114', 'cd115', 'sldu2', 'sldu3', 'sldu4', 'sldl2', 'sldl3', 'sldl4', 'aianhhsc', 'csasc', 'cnectasc', 'memi', 'nmemi', 'puma', 'reserved'],
                  
    '1': ['total', 'population_of_one_race', 'white_alone', 'black_or_african_american_alone', 'american_indian_and_alaska_native_alone', 'asian_alone', 'native_hawaiian_and_other_pacific_islander_alone', 'some_other_race_alone', 'population_of_two_or_more_races', 'population_of_two_races', 'white_black_or_african_american', 'white_american_indian_and_alaska_native', 'white_asian', 'white_native_hawaiian_and_other_pacific_islander', 'white_some_other_race', 'black_or_african_american_american_indian_and_alaska_native', 'black_or_african_american_asian', 'black_or_african_american_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_some_other_race', 'american_indian_and_alaska_native_asian', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander', 'asian_some_other_race', 'native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_three_races', 'white_black_or_african_american_american_indian_and_alaska_native', 'white_black_or_african_american_asian', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_some_other_race', 'white_american_indian_and_alaska_native_asian', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander', 'white_asian_some_other_race', 'white_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_asian_some_other_race', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_asian_some_other_race', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_four_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_asian_some_other_race', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_asian_some_other_race', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_five_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_six_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'total_hl', 'hispanic_or_latino_hl', 'not_hispanic_or_latino_hl', 'population_of_one_race_hl', 'white_alone_hl', 'black_or_african_american_alone_hl', 'american_indian_and_alaska_native_alone_hl', 'asian_alone_hl', 'native_hawaiian_and_other_pacific_islander_alone_hl', 'some_other_race_alone_hl', 'population_of_two_or_more_races_hl', 'population_of_two_races_hl', 'white_black_or_african_american_hl', 'white_american_indian_and_alaska_native_hl', 'white_asian_hl', 'white_native_hawaiian_and_other_pacific_islander_hl', 'white_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_hl', 'black_or_african_american_asian_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_some_other_race_hl', 'american_indian_and_alaska_native_asian_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_hl', 'asian_some_other_race_hl', 'native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_three_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_hl', 'white_black_or_african_american_asian_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_asian_some_other_race_hl', 'white_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_asian_some_other_race_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_asian_some_other_race_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_four_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_asian_some_other_race_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_five_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_six_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl'],

    '2': ['total_18', 'population_of_one_race_18', 'white_alone_18', 'black_or_african_american_alone_18', 'american_indian_and_alaska_native_alone_18', 'asian_alone_18', 'native_hawaiian_and_other_pacific_islander_alone_18', 'some_other_race_alone_18', 'population_of_two_or_more_races_18', 'population_of_two_races_18', 'white__black_or_african_american_18', 'white__american_indian_and_alaska_native_18', 'white__asian_18', 'white__native_hawaiian_and_other_pacific_islander_18', 'white__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native_18', 'black_or_african_american__asian_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__some_other_race_18', 'american_indian_and_alaska_native__asian_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander_18', 'asian__some_other_race_18', 'native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_three_races_18', 'white__black_or_african_american__american_indian_and_alaska_native_18', 'white__black_or_african_american__asian_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__some_other_race_18', 'white__american_indian_and_alaska_native__asian_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander_18', 'white__asian__some_other_race_18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__asian__some_other_race_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__asian__some_other_race_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_four_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__asian__some_other_race_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__asian__some_other_race_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_five_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_six_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'total_hl18', 'hispanic_or_latino_hl18', 'not_hispanic_or_latino_hl18', 'population_of_one_race_hl18', 'white_alone_hl18', 'black_or_african_american_alone_hl18', 'american_indian_and_alaska_native_alone_hl18', 'asian_alone_hl18', 'native_hawaiian_and_other_pacific_islander_alone_hl18', 'some_other_race_alone_hl18', 'population_of_two_or_more_races_hl18', 'population_of_two_races_hl18', 'white__black_or_african_american_hl18', 'white__american_indian_and_alaska_native_hl18', 'white__asian_hl18', 'white__native_hawaiian_and_other_pacific_islander_hl18', 'white__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native_hl18', 'black_or_african_american__asian_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__some_other_race_hl18', 'american_indian_and_alaska_native__asian_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander_hl18', 'asian__some_other_race_hl18', 'native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_three_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native_hl18', 'white__black_or_african_american__asian_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__asian__some_other_race_hl18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__asian__some_other_race_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__asian__some_other_race_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_four_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__asian__some_other_race_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_five_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_six_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'housing_total', 'housing_occupied', 'housing_vacant'],
}

census_columns['starts'] = 1 + np.insert(np.cumsum(census_columns['widths'])[:-1], 0, 0)