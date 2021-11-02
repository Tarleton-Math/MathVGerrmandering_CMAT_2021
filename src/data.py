from . import *
import urllib, zipfile as zf, shapely.ops
try:
    import mechanicalsoup
except:
    os.system('pip install --upgrade mechanicalsoup')
    import mechanicalsoup

@dataclasses.dataclass
class Data(Base):
    election_filters  : typing.Tuple = (
        "office='USSen' and race='general'",
        "office='President' and race='general'",
        "office like 'USRep%' and race='general'")
        
    def __post_init__(self):
        super().__post_init__()
        self.sources = ('crosswalks', 'assignments', 'shapes', 'census', 'elections', 'countries', 'proposals', 'joined', 'tabblock', 'bg', 'tract', 'cntyvtd', 'cnty', 'all')
        self.check_inputs()

        self.tbls = dict()
        self.pq   = dict()
        self.zp   = dict()
        self.path = dict()
        stem = f'{self.state.abbr}_{self.census_yr}'
        dataset = f'{root_bq}.redistricting_data'
        bqclient.create_dataset(dataset, exists_ok=True)
        for src in self.sources:
            self.path[src] = data_path / f'{src}/{stem.replace("_", "/")}'
            self.tbls[src] = f'{dataset}.{stem}_{src}'
            self.zp  [src] = self.path[src] / f'{stem}_{src}.zip'
            self.pq  [src] = self.zp[src].with_suffix('.parquet')
        self.tbls['countries'] = f'{dataset}.countries'

        for src in self.sources:
            self.get(src)
            
#####################################################################################################
#####################################################################################################

    def get_countries(self):
        src = 'countries'
        tbl_raw = self.tbls[src] + '_raw'
        if check_table(tbl_raw):
            rpt(f'using existing raw table')
        else:
            rpt(f'creating raw table')
            import json
            try:
                import datapackage
            except:
                os.system('pip install --upgrade datapackage')
                import datapackage
            package = datapackage.Package('https://datahub.io/core/geo-countries/datapackage.json')
            for resource in package.resources:
                if resource.name == 'countries':
                    js = json.loads(resource.raw_read())
                    break
            L = [{'country' :g['properties']['ADMIN'],
                  'abbr'    :g['properties']['ISO_A3'],
                  'geometry':str(g['geometry'])
                 } for g in js['features']]
            df = pd.DataFrame(L)
            load_table(tbl_raw, df=df)
        query = f"""
select
    country,
    abbr,
    st_geogfromgeojson(geometry, make_valid => TRUE) as polygon
from
    {tbl_raw}
order by
    country
"""
        load_table(self.tbls[src], query=query)
        delete_table(tbl_raw)

#####################################################################################################
#####################################################################################################

    def get_proposals(self):
        src = 'proposals'
        self.proposals = list()
        
        # self.proposals_dict = {dt:list() for dt in self.District_types.values()}
        browser = mechanicalsoup.Browser()
        for abbr, district_type in self.District_types.items():
            rpt(f'fetching {district_type}')
            not_found = 0
            new = 0
            for k in range(1000):
                not_found += 1
                proposal = f'plan{abbr}{2100+k}'.lower()
                root_url = f'https://data.capitol.texas.gov/dataset/{proposal}#'
                login_page = browser.get(root_url)
                tag = login_page.soup.select('a')
                if len(tag) >= 10:
                    not_found = 0
                    proposal_path = self.path[src] / f'{district_type}/{proposal}/proposal'
                    proposal_path.mkdir(parents=True, exist_ok=True)
                    for t in tag:
                        url = t['href']
                        if 'blk.zip' in url:
                            fn = url.split('/')[-1]
                            zp = proposal_path / fn
                            if not zp.is_file():
                                print(f'downloading {proposal}')
                                os.chdir(proposal_path)
                                urllib.request.urlretrieve(url, zp)
                                os.system("unzip -u '*.zip' >/dev/null 2>&1");
                                os.system("unzip -u '*.zip' >/dev/null 2>&1");
                                new += 1
                                os.chdir(code_path)
                    csv = proposal_path / f'{proposal.upper()}.csv'
                    df = pd.read_csv(csv, skiprows=1, names=('geoid', district_type), dtype={'geoid':str})
                    seats = df[district_type].nunique()
                    complete = (seats == self.Seats[district_type]) or (k == 0)
                    print(proposal, seats, complete)
                    self.proposals.append({'district_type':district_type, 'proposal':proposal, 'k':k, 'complete':complete, 'csv':str(csv)})
                if not_found > 15:
                    break
        # rpt({key:len(val) for key, val in self.proposals_dict.items()})
        self.proposals = pd.DataFrame(self.proposals)
        load_table(self.tbls[src], df=self.proposals)

#####################################################################################################
#####################################################################################################
        
    def fetch(self, src, url):
        try:
            df = pd.read_parquet(self.pq[src])
            rpt(f'using existing parquet')
            rpt(f'loading table')
            load_table(self.tbls[src], df=df)
            zipfile = False
        except:
            self.path[src].mkdir(parents=True, exist_ok=True)
            os.chdir(self.path[src])
            try:
                zipfile = zf.ZipFile(self.zp[src])
                rpt(f'using existing zip')
            except:
                try:
                    rpt(f'downloading zip from {url}')
                    zipfile = zf.ZipFile(urllib.request.urlretrieve(url, self.zp[src])[0])
                except urllib.error.HTTPError:
                    raise Exception(f'n\nFAILED - BAD URL {url}\n\n')
        return zipfile

#####################################################################################################
#####################################################################################################
    
    def get_joined(self):
        self.refresh_tbl.update(self.Levels)
        src = 'joined'
        cols = {'A' : self.District_types.values(),
                'C' : ['seats_cd', 'seats_sldu', 'seats_sldl', 'total_pop_prop'] + Census_columns['data'],
                'E' : [c for c in get_cols(self.tbls['elections']) if c not in ['geoid', 'county']],
                'S' : []}
        sels = ([f'A.{c} as {c}'              for c in cols['A']] + 
                [f'coalesce(C.{c}, 0) as {c}' for c in cols['C']] + 
                [f'coalesce(E.{c}, 0) as {c}' for c in cols['E']] + 
                [f'S.{c} as {c}'              for c in cols['S']])
        query = f"""
select
    A.geoid,
    substring(A.geoid, 1, 15) as tabblock,
    substring(A.geoid, 1, 12) as bg,
    substring(A.geoid, 1, 11) as tract,
    A.cntyvtd,
    A.cnty,
    max(E.county) over (partition by A.cnty) as county,  --workaround ... county names come from election data, but not all blocks have election data
    0 as district,
    {join_str().join(sels)},
    st_simplify(S.polygon, 0) as polygon,
    st_simplify(S.polygon, 5) as polygon_simp,
    S.aland
from
    {self.tbls['assignments']} as A
left join
    {self.tbls['census']} as C
on
    A.geoid = C.geoid
left join
    {self.tbls['elections']} as E
on
    A.geoid = E.geoid
left join
    {self.tbls['shapes']} as S
on
    A.geoid = S.geoid
"""
        load_table(self.tbls[src], query=query)

        
    def aggegrate(self, query_in, geo='compute', show=False):
        geo = geo.lower()
        if geo == 'compute':
            rpt(f'computing geo')
        elif geo == 'join':
            rpt(f'attempting to join existing geo')
        else:
            rpt('skipping geo')
        geo_cols = ['cntyvtd', 'county', 'cd', 'sldu', 'sldl', 'district']
        cols = get_cols(self.tbls['joined'])
        a = cols.index('total_pop')
        b = cols.index('polygon')
        data_cols = cols[a:b]
        data_sums = [f'sum({c}) as {c}' for c in data_cols]
####### Builds a nested SQL query to generate nodes
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
        query = [query_in]
        query.append(f"""
select
    geoid,
    geoid_new,
    {join_str().join(geo_cols)},
    {join_str().join([f'sum(total_pop) over (partition by geoid_new, {g}) as pop_{g}' for g in geo_cols])}
from (
    {subquery(query[-1])}
    )
""")
        
####### Now, we find the max over all units in a given geoid ###########
        query.append(f"""
select
    *,
    {join_str().join([f'max(pop_{g}) over (partition by geoid_new) pop_{g}_max' for g in geo_cols])},
from (
    {subquery(query[-1])}
    )
""")
####### Now, we create temporary columns that are null except on the rows of the unit achieving the max value found above
####### When we do the "big aggegration" below, max() will grab the name of the correct unit (one with max seat)
        query.append(f"""
select
    *,
    {join_str().join([f'case when pop_{g} = pop_{g}_max then {g} else null end as {g}_new' for g in geo_cols])},
from (
    {subquery(query[-1])}
    )
""")

####### Time for the big aggregration step.
####### Join source, groupby geoid_new, and aggregate categorical variable with max, numerical variables with sum,
####### and geospatial polygon with st_union_agg.
        if geo == 'compute':
            query.append(f"""
select
    A.*,
    B.* except (geoid, polygon),
from (
    {subquery(query[-1])}
    ) as A
left join
    {self.tbls['joined']} as B
on
    A.geoid = B.geoid
""")
            query.append(f"""
select
    geoid_new as geoid,
    {join_str().join([f'max({g}_new) as {g}' for g in geo_cols])},
    sum(seats_cd) as seats_cd,
    sum(seats_sldu) as seats_sldu,
    sum(seats_sldl) as seats_sldl,
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
        
        else:
            query.append(f"""
select
    A.*,
    B.* except (geoid, polygon, aland)
from (
    {subquery(query[-1])}
    ) as A
left join
    {self.tbls['all']} as B
on
    A.geoid = B.geoid
""")
            
            query.append(f"""
select
    geoid_new as geoid,
    {join_str().join([f'max({g}_new) as {g}' for g in geo_cols])},
    sum(seats_cd) as seats_cd,
    sum(seats_sldu) as seats_sldu,
    sum(seats_sldl) as seats_sldl,
    {join_str().join(data_sums)},
from (
    {subquery(query[-1])}
    )
group by
    geoid_new
""")
            if geo == 'join':
                query.append(f"""
select
    A.*,
    B.polygon,
    B.aland,
    B.perim,
    B.polsby_popper,
    B.density,
    B.point
from (
    {subquery(query[-1])}
    ) as A
left join
    {self.tbls['all']} as B
on
    A.geoid = B.geoid
""")

        if show:
            for k, q in enumerate(query):
                print(f'\n=====================================================================================\nstage {k}')
                print(q)
        return query[-1]
            
    
    def aggegrate_level(self, level, show=False):
        self.refresh_tbl.add('all')
        query = f"""
select
    *,
    {level} as geoid_new
from
    {self.tbls['joined']}
"""
        load_table(self.tbls[level], query=self.aggegrate(query, geo='compute', show=show))

    
    def get_tabblock(self, show=False):
        self.aggegrate_level('tabblock', show=show)
    def get_bg(self, show=False):
        self.aggegrate_level('bg', show=show)
    def get_tract(self, show=False):
        self.aggegrate_level('tract', show=show)
    def get_cntyvtd(self, show=False):
        self.aggegrate_level('cntyvtd', show=show)
    def get_cnty(self, show=False):
        self.aggegrate_level('cnty', show=show)
    def get_all(self):
        query = '\nunion all\n'.join([f'select * from {self.tbls[level]}' for level in self.Levels])
        load_table(self.tbls['all'], query=query)

#####################################################################################################
#####################################################################################################

    def get_elections(self):
        self.refresh_tbl.add('joined')
        src = 'elections'
        url = f'https://data.capitol.texas.gov/dataset/aab5e1e5-d585-4542-9ae8-1108f45fce5b/resource/253f5191-73f3-493a-9be3-9e8ba65053a2/download/{self.census_yr}-general-vtd-election-data.zip'
        zipfile = self.fetch(src, url)
        if zipfile is False:
            return
        
        tbl_raw = self.tbls[src] + '_raw'
        if check_table(tbl_raw):
            rpt(f'using existing raw table')
        else:
            rpt(f'creating raw table')
            ext = '_Returns.csv'
            k = len(ext)
            L = []
            for fn in zipfile.namelist():
                if fn[-k:]==ext:
                    df = extract_file(zipfile, fn, sep=',')
                    df = (df.astype({'votes':int, 'fips':str, 'vtd':str})
                          .query('votes > 0')
                          .query("party in ['R', 'D', 'L', 'G']")
                         )
                    w = fn.lower().split('_')
                    df['election_yr'] = int(w[0])
                    df['race'] = '_'.join(w[1:-2])
                    L.append(df)
#                     os.unlink(fn)

    ######## vertically stack then clean so that joins work correctly later ########
            df = pd.concat(L, axis=0, ignore_index=True).reset_index(drop=True)
            df['fips'] = df['fips'].str.lower()
            df['vtd']  = df['vtd'] .str.lower()
            f = lambda col: col.str.replace('.', '', regex=False).str.replace(' ', '', regex=False).str.replace(',', '', regex=False).str.replace('-', '', regex=False).str.replace("'", '', regex=False)
            df['name'] = f(df['name'])
            df['race'] = f(df['race'])
            df['office'] = f(df['office'])
            mask = ((df['office'].str[:5] == 'USRep') &
                     df['office'].str[-1].str.isnumeric() &
                    ~df['office'].str[-2].str.isnumeric())
            df.loc[mask, 'office'] = df.loc[mask, 'office'].str[:-1] + df.loc[mask, 'office'].str[-1].str.rjust(2, '0')

    ######## correct differences between cntyvtd codes in assignements (US Census) and elections (TX Legislative Council) ########
            c = f'cntyvtd'
            df[c]     = df['fips'].str.rjust(3, '0') + df['vtd']         .str.rjust(6, '0')
            df['alt'] = df['fips'].str.rjust(3, '0') + df['vtd'].str[:-1].str.rjust(6, '0')
            assign = read_table(self.tbls['assignments'])[c].drop_duplicates()

            # find cntyvtd in elections not among assignments
            unmatched = ~df[c].isin(assign)
            # different was usually a simple character shift
            df.loc[unmatched, c] = df.loc[unmatched, 'alt']
            # check for any remaining unmatched
            unmatched = ~df[c].isin(assign)
            if unmatched.any():
                display(df[unmatched].sort_values('votes', ascending=False))
                raise Exception('Unmatched election results')

            df = df.drop(columns=['fips', 'vtd', 'incumbent', 'alt']).rename(columns={'name':'candidate'})
            load_table(tbl_raw, df=df, preview_rows=0)
    

######## Apportion votes from cntyvtd to its tabblock proportional to population ########
######## We computed cntyvtd_pop_prop = pop_tabblock / pop_cntyvtd  during census processing ########
######## Each tabblock gets this proportion of votes cast in its cntyvtd ########
        rpt(f'apportioning votes to blocks proportional to population')
        sep = ' or\n    '
        query = f"""
select
    A.geoid,
    B.county,
    concat(B.office, '_', B.election_yr, '_', B.party, '_', B.candidate, '_', B.race) as election,
    B.votes * A.cntyvtd_pop_prop as votes,
from
    {self.tbls['census']} as A
inner join
    {tbl_raw} as B
on
    A.cntyvtd = B.cntyvtd
where
    {sep.join(f'({x})' for x in self.election_filters)}
order by
    geoid
"""
        tbl_temp = self.tbls[src] + '_temp'
        load_table(tbl_temp, query=query, preview_rows=0)

######## To bring everything into one table, we must pivot from long to wide format (one row per tabblock) ########
######## While easy in Python and Excel, this is delicate in SQl given the number of electionS and tabblocks ########
######## Even BigQuery refuseS to pivot all elections simulatenously ########
######## So we break the elections into chunks, pivot separately, then join horizontally ########
        df = run_query(f'select distinct election from {tbl_temp}')
        elections = tuple(sorted(df['election']))
        stride = 100
        tbl_chunks = list()
        alias_chr = 64 # silly hack to give table aliases A, B, C, ...
        for a in np.arange(0, len(elections), stride):
            b = a + stride
            rpt(f'pivoting columns {a} thru {b}')
            E = elections[a:b]
            t = f'{self.tbls[src]}_{a}'
            tbl_chunks.append(t)
            query = f"""
select
    *
from (
    select
        geoid,
        county,
        election,
        votes
    from
        {tbl_temp}
    )
pivot(
    sum(votes)
    for election in {E})
"""
            load_table(t, query=query, preview_rows=0)
        
######## create the join query as we do each chunk so we can run it at the end ########
            alias_chr += 1
            alias = chr(alias_chr)
            if len(tbl_chunks) == 1:
                query_join = f"""
select
    A.geoid,
    A.county,
    {join_str().join(elections)}
from
    {t} as {alias}
"""
            else:
                query_join += f"""
inner join
    {t} as {alias}
on
    A.geoid = {alias}.geoid
"""
        query_join += f'order by geoid'
        load_table(self.tbls[src], query=query_join, preview_rows=0)
        delete_table(tbl_temp)
        for t in tbl_chunks:
            delete_table(t)

#####################################################################################################
#####################################################################################################
#
#     def add_header(self, file, header):
#         Used for 2010 Census - unused now, but saved for possible future resurrection
#         cmd = 'sed -i "1s/^/' + '|'.join(header) + '\\n/" ' + file
#         os.system(cmd)

    def get_census(self):
        self.refresh_tbl.add('joined')
        src = 'census'
        url = f'https://www2.census.gov/programs-surveys/decennial/{self.census_yr}/data/01-Redistricting_File--PL_94-171/{self.state.name.replace(" ", "_")}/{self.state.abbr.lower()}{self.census_yr}.pl.zip'
        zipfile = self.fetch(src, url)
        if zipfile is False:
            return

        tbl_raw = self.tbls[src] + '_raw'
        if check_table(tbl_raw):
            rpt(f'using existing raw table')
        else:
            rpt(f'creating raw table')
    ######## PL_94-171 involves multiple files - load each into a temp table ########
            temp = dict()
            for fn in zipfile.namelist():
                if fn[-3:] == '.pl':
                    if fn[2:5] == 'geo':
                        i = 'geo'
                    else:
                        i = fn[6]
                    temp[i] = tbl_raw+i
                    if check_table(temp[i]):
                        rpt(f'using existing raw {i} table')
                    else:
                        rpt(f'processing {fn}')
                        file = zipfile.extract(fn)
                        schema = [google.cloud.bigquery.SchemaField(**col) for col in Census_columns[i]]
                        with open(file, mode='rb') as f:
                            bqclient.load_table_from_file(f, temp[i], job_config=google.cloud.bigquery.LoadJobConfig(field_delimiter='|', schema=schema)).result()
        #                 os.unlink(fn)

######## combine census tables into one table ########
            rpt(f'joining')
            query = f"""
select
    concat(right(concat("00", A.state), 2), right(concat("000", A.county), 3), right(concat("000000", A.tract), 6), right(concat("0000", A.block), 4)) as geoid,
    {join_str().join(Census_columns['data'])}
from
    {temp['geo']} as A
inner join
    {temp['1']} as B
on
    A.fileid = B.fileid
    and A.stusab = B.stusab
    and A.chariter = B.chariter
    and A.logrecno = B.logrecno
inner join
    {temp['2']} as C
on
    A.fileid = C.fileid
    and A.stusab = C.stusab
    and A.chariter = C.chariter
    and A.logrecno = C.logrecno
inner join
    {temp['3']} as D
on
    A.fileid = D.fileid
    and A.stusab = D.stusab
    and A.chariter = D.chariter
    and A.logrecno = D.logrecno
where
    A.block != ""
order by
    geoid
"""
            load_table(tbl_raw, query=query, preview_rows=0)
            for t in temp.values():
                delete_table(t)
    
        rpt(f'creating table')
######## Use crosswalks to push 2010 data on 2010 tabblocks onto 2020 tabblocks ########
        if self.census_yr == self.shapes_yr:
            query = f"""
select
    geoid,
    {join_str().join(Census_columns['data'])}
from
    {tbl_raw}
"""
        else:
            query = f"""
select
    E.geoid_{self.shapes_yr} as geoid,
    {join_str().join([f'sum(D.{c} * E.aland_prop) as {c}' for c in Census_columns['data']])}
from
    {tbl_raw} as D
inner join
    {self.tbls['crosswalks']} as E
on
    D.geoid = E.geoid_{self.census_yr}
group by
    geoid
"""

######## Compute cntyvtd_pop_prop = pop_tabblock / pop_cntyvtd ########
######## We will use this later to apportion votes from cntyvtd to its tabblocks  ########
        query = f"""
select
    *,
    case when cntyvtd_pop > 0 then total_pop / cntyvtd_pop else 1 / cntyvtd_count end as cntyvtd_pop_prop,
    total_pop_prop * {self.Seats['cd']} as seats_cd,
    total_pop_prop * {self.Seats['sldu']} as seats_sldu,
    total_pop_prop * {self.Seats['sldl']} as seats_sldl
from (
    select
        G.*,
        F.cntyvtd,
        G.total_pop / sum(G.total_pop) over () as total_pop_prop,
        sum(G.total_pop) over (partition by F.cntyvtd) as cntyvtd_pop,
        count(*) over (partition by F.cntyvtd) as cntyvtd_count
    from 
        {self.tbls['assignments']} as F
    inner join(
        {subquery(query, indents=2)}
        ) as G
    on
        F.geoid = G.geoid
    )
order by
    geoid
"""
        load_table(self.tbls[src], query=query, preview_rows=0)

#####################################################################################################
#####################################################################################################
    
    def get_shapes(self):
        self.refresh_tbl.add('joined')
        src = 'shapes'
        url = f'https://www2.census.gov/geo/tiger/TIGER{self.shapes_yr}/TABBLOCK'
        if self.shapes_yr == 2010:
            url += '/2010'
        elif self.shapes_yr == 2020:
            url += '20'
        url += f'/tl_{self.shapes_yr}_{self.state.fips}_tabblock{str(self.shapes_yr)[-2:]}.zip'
        zipfile = self.fetch(src, url)
        if zipfile is False:
            return
        
        tbl_raw = self.tbls[src] + '_raw'
        if check_table(tbl_raw):
            rpt(f'using existing raw table')
        else:
            rpt(f'creating raw table')
            for fn in zipfile.namelist():
                zipfile.extract(fn)
            a = 0
            chunk_size = 50000
            while True:
                rpt(f'starting row {a}')
                df = lower(gpd.read_file(self.path[src], rows=slice(a, a+chunk_size)))
                df.columns = [x[:-2] if x[-2:].isnumeric() else x for x in df.columns]
                df = df[['geoid', 'aland', 'geometry']]
                # convert to https://spatialreference.org/ref/esri/usa-contiguous-albers-equal-area-conic/ to buffer
                df['geometry'] = df['geometry'].to_crs(crs_area).buffer(5).apply(lambda p: shapely.ops.orient(p, -1)).to_crs(crs_census)
                load_table(tbl_raw, df=df.to_wkb(), overwrite=a==0)
                if df.shape[0] < chunk_size:
                    break
                else:
                    a += chunk_size
#             for fn in zipfile.namelist():
#                 os.unlink(fn)

        rpt(f'creating table')
        query = f"""
select
    geoid,
    st_geogfrom(geometry) as polygon,
    cast(aland as float64) / {m_per_mi**2} as aland,
from
    {tbl_raw}
order by
    geoid
"""
        load_table(self.tbls[src], query=query, preview_rows=0)
        delete_table(tbl_raw)

#####################################################################################################
#####################################################################################################
    
    def get_assignments(self):
        self.refresh_tbl.add('joined')
        src = 'assignments'
        url = f'https://www2.census.gov/geo/docs/maps-data/data/baf'
        if self.census_yr == 2020:
            url += '2020'
        url += f'/BlockAssign_ST{self.state.fips}_{self.state.abbr.upper()}.zip'
        zipfile = self.fetch(src, url)
        if zipfile is False:
            return

        rpt(f'creating table')
        L = []
        for fn in zipfile.namelist():
            col = fn.lower().split('_')[-1][:-4]
            if fn[-3:] == 'txt' and col != 'aiannh':
                df = extract_file(zipfile, fn, sep='|')
                try:
                    df['district'] = df['district'].astype(int)
                except:
                    if col == 'vtd':
                        df['countyfp'] = df['countyfp'].str.rjust(3, '0') + df['district'].str.rjust(6, '0')
                        df = df.iloc[:,:2]
                        col = 'cntyvtd'
                df.columns = ['geoid', col]
                L.append(df.set_index('geoid'))
#                 os.unlink(fn)
        df = lower(pd.concat(L, axis=1).reset_index()).sort_values('geoid')
        df['cnty'] = df['geoid'].str[2:5]
        df = df[['geoid', 'cntyvtd', 'cnty', 'cd', 'sldu', 'sldl']]
        rpt(f'creating table')
        load_table(self.tbls[src], df=df, preview_rows=0)

#####################################################################################################
#####################################################################################################
    
    def get_crosswalks(self):
        self.refresh_tbl.add('joined')
        src = 'crosswalks'
        url = f'https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/TAB2010_TAB2020_ST{self.state.fips}.zip'
        zipfile = self.fetch(src, url)
        if zipfile is False:
            return
            
        rpt(f'creating table')
        geoids = [f'geoid_{yr}' for yr in [2020, 2010]]
        for fn in zipfile.namelist():
            df = extract_file(zipfile, fn, sep='|')
            for geoid in geoids:
                yr = geoid[-4:]
                df[geoid] = df[f'state_{yr}'].str.rjust(2,'0') + df[f'county_{yr}'].str.rjust(3,'0') + df[f'tract_{yr}'].str.rjust(6,'0') + df[f'blk_{yr}'].str.rjust(4,'0')
#             os.unlink(fn)
        df['arealand_int'] = df['arealand_int'].astype(float)
        df['A'] = df.groupby(geoids[1])['arealand_int'].transform('sum')
        df['aland_prop'] = (df['arealand_int'] / df['A']).fillna(0)
        df = df[geoids+['aland_prop']].sort_values(geoids[0])
        rpt(f'creating table')
        load_table(self.tbls[src], df=df, preview_rows=0)