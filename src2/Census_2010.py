@dataclasses.dataclass
class Census(Variable):
    name: str = 'census'
    
    def __post_init__(self):
        self.yr = self.g.census_yr
        super().__post_init__()


    def get(self):
        self.url = f"https://www2.census.gov/programs-surveys/decennial/{self.yr}/data/01-Redistricting_File--PL_94-171/{self.state.name.replace(' ', '_')}/{self.state.abbr.lower()}{self.yr}.pl.zip"
        
        exists = super().get()
        if not exists['tbl']:
            if not exists['raw']:
                self.get_zip()
                print(f'creating raw table', end=concat_str)
                self.process_raw()
            print(f'creating table', end=concat_str)
            self.process()
        return self


    def process_raw(self):
######## In 2010 PL_94-171 involved 3 files - we first load each into a temp table ########
        for fn in self.zipfile.namelist():
            if fn[-3:] == '.pl':
                print(fn, end=concat_str)
                file = self.zipfile.extract(fn)
######## Geo file is fixed width (not delimited) and must be handled carefully ########                
                if fn[2:5] == 'geo':
                    raw_geo = self.raw + '_geo'
                    raw_geo_temp = raw_geo + '_temp'
                    load_table(raw_geo_temp, file=file)
                    sel = [f'trim(substring(string_field_0, {s}, {w})) as {n}' for s, w, n in zip(census_columns['starts'], census_columns['widths'], census_columns['geo'])]
                    query = 'select\n\t' + ',\n\t'.join(sel) + '\nfrom\n\t' + raw_geo_temp
                    load_table(raw_geo, query=query)
######## We must insert the column headers into the 2 data files before upload ########
                else:
                    i = fn[6]
                    if i in ['1', '2']:
                        cmd = 'sed -i "1s/^/' + ','.join(census_columns['joins'] + census_columns[i]) + '\\n/" ' + file
                        os.system(cmd)
                        load_table(self.raw+i, file=file)
                os.unlink(fn)

######## combine 3 census table into one table ########
        print(f'joining', end=concat_str)
        query = f"""
select
    concat(right(concat("00", state), 2), right(concat("000", county), 3), right(concat("000000", tract), 6), right(concat("0000", block), 4)) as geoid,
    {join_str(1).join(census_columns['data'])}
from
    {self.raw}1 as A
inner join
    {self.raw}2 as B
on
    A.fileid = B.fileid
    and A.stusab = B.stusab
    and A.logrecno = B.logrecno
inner join
    {raw_geo} as C
on
    A.fileid = trim(C.fileid)
    and A.stusab = trim(C.stusab)
    and A.logrecno = cast(C.logrecno as int)
where
    C.block != ""
order by
    geoid
"""
        load_table(self.raw, query=query, preview_rows=0)
        
######## clean up ########
        delete_table(raw_geo_temp)
        delete_table(raw_geo)
        delete_table(self.raw+'1')
        delete_table(self.raw+'2')


    def process(self):
######## Use crosswalks to push 2010 data on 2010 tabblocks onto 2020 tabblocks ########
        if self.g.census_yr == self.g.shapes_yr:
            query = f"""
select
    geoid,
    {join_str(1).join(census_columns['data'])}
from
    {self.raw}
"""

        else:
            query = f"""
select
    E.geoid_{self.g.shapes_yr} as geoid,
    {join_str(1).join([f'sum(D.{c} * E.aland_prop) as {c}' for c in census_columns['data']])}
from
    {self.raw} as D
inner join
    {self.g.crosswalks.tbl} as E
on
    D.geoid = E.geoid_{self.g.census_yr}
group by
    geoid
"""

######## Compute cntyvtd_pop_prop = pop_tabblock / pop_cntyvtd ########
######## We will use this later to apportion votes from cntyvtd to its tabblocks  ########
        query = f"""
select
    G.*,
    F.cntyvtd,
    sum(G.total) over (partition by F.cntyvtd) as cntyvtd_pop,
    case when (sum(G.total) over (partition by F.cntyvtd)) > 0 then total / (sum(G.total) over (partition by F.cntyvtd)) else 1 / (count(*) over (partition by F.cntyvtd)) end as cntyvtd_pop_prop,
from 
    {self.g.assignments.tbl} as F
inner join(
    {query}
    ) as G
on
    F.geoid = G.geoid
order by
    geoid
"""
        load_table(self.tbl, query=query, preview_rows=0)
        
        
        
        
census_columns = {
    'joins':  ['fileid', 'stusab', 'chariter', 'cifsn', 'logrecno'],

    'widths': [6, 2, 3, 2, 3, 2, 7, 1, 1, 2, 3, 2, 2, 5, 2, 2, 5, 2, 2, 6, 1, 4, 2, 5, 2, 2, 4, 5, 2, 1, 3, 5, 2, 6, 1, 5, 2, 5, 2, 5, 3, 5, 2, 5, 3, 1, 1, 5, 2, 1, 1, 2, 3, 3, 6, 1, 3, 5, 5, 2, 5, 5, 5, 14, 14, 90, 1, 1, 9, 9, 11, 12, 2, 1, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 5, 18],

    'geo': ['fileid', 'stusab', 'sumlev', 'geocomp', 'chariter', 'cifsn', 'logrecno', 'region', 'division', 'state', 'county', 'countycc', 'countysc', 'cousub', 'cousubcc', 'cousubsc', 'place', 'placecc', 'placesc', 'tract', 'blkgrp', 'block', 'iuc', 'concit', 'concitcc', 'concitsc', 'aianhh', 'aianhhfp', 'aianhhcc', 'aihhtli', 'aitsce', 'aits', 'aitscc', 'ttract', 'tblkgrp', 'anrc', 'anrccc', 'cbsa', 'cbsasc', 'metdiv', 'csa', 'necta', 'nectasc', 'nectadiv', 'cnecta', 'cbsapci', 'nectapci', 'ua', 'uasc', 'uatype', 'ur', 'cd', 'sldu', 'sldl', 'vtd', 'vtdi', 'reserve2', 'zcta5', 'submcd', 'submcdcc', 'sdelm', 'sdsec', 'sduni', 'arealand', 'areawatr', 'name', 'funcstat', 'gcuni', 'pop100', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'partflag', 'reserve3', 'uga', 'statens', 'countyns', 'cousubns', 'placens', 'concitns', 'aianhhns', 'aitsns', 'anrcns', 'submcdns', 'cd113', 'cd114', 'cd115', 'sldu2', 'sldu3', 'sldu4', 'sldl2', 'sldl3', 'sldl4', 'aianhhsc', 'csasc', 'cnectasc', 'memi', 'nmemi', 'puma', 'reserved'],
                  
    '1': ['total', 'population_of_one_race', 'white_alone', 'black_or_african_american_alone', 'american_indian_and_alaska_native_alone', 'asian_alone', 'native_hawaiian_and_other_pacific_islander_alone', 'some_other_race_alone', 'population_of_two_or_more_races', 'population_of_two_races', 'white_black_or_african_american', 'white_american_indian_and_alaska_native', 'white_asian', 'white_native_hawaiian_and_other_pacific_islander', 'white_some_other_race', 'black_or_african_american_american_indian_and_alaska_native', 'black_or_african_american_asian', 'black_or_african_american_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_some_other_race', 'american_indian_and_alaska_native_asian', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander', 'asian_some_other_race', 'native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_three_races', 'white_black_or_african_american_american_indian_and_alaska_native', 'white_black_or_african_american_asian', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_some_other_race', 'white_american_indian_and_alaska_native_asian', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander', 'white_asian_some_other_race', 'white_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_asian_some_other_race', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'american_indian_and_alaska_native_asian_some_other_race', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_four_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_asian_some_other_race', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_american_indian_and_alaska_native_asian_some_other_race', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_five_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'population_of_six_races', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race', 'total_hl', 'hispanic_or_latino_hl', 'not_hispanic_or_latino_hl', 'population_of_one_race_hl', 'white_alone_hl', 'black_or_african_american_alone_hl', 'american_indian_and_alaska_native_alone_hl', 'asian_alone_hl', 'native_hawaiian_and_other_pacific_islander_alone_hl', 'some_other_race_alone_hl', 'population_of_two_or_more_races_hl', 'population_of_two_races_hl', 'white_black_or_african_american_hl', 'white_american_indian_and_alaska_native_hl', 'white_asian_hl', 'white_native_hawaiian_and_other_pacific_islander_hl', 'white_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_hl', 'black_or_african_american_asian_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_some_other_race_hl', 'american_indian_and_alaska_native_asian_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_hl', 'asian_some_other_race_hl', 'native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_three_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_hl', 'white_black_or_african_american_asian_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_asian_some_other_race_hl', 'white_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_asian_some_other_race_hl', 'black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'american_indian_and_alaska_native_asian_some_other_race_hl', 'american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_four_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_asian_some_other_race_hl', 'white_black_or_african_american_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_five_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_some_other_race_hl', 'white_black_or_african_american_american_indian_and_alaska_native_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_black_or_african_american_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'white_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl', 'population_of_six_races_hl', 'white_black_or_african_american_american_indian_and_alaska_native_asian_native_hawaiian_and_other_pacific_islander_some_other_race_hl'],

    '2': ['total_18', 'population_of_one_race_18', 'white_alone_18', 'black_or_african_american_alone_18', 'american_indian_and_alaska_native_alone_18', 'asian_alone_18', 'native_hawaiian_and_other_pacific_islander_alone_18', 'some_other_race_alone_18', 'population_of_two_or_more_races_18', 'population_of_two_races_18', 'white__black_or_african_american_18', 'white__american_indian_and_alaska_native_18', 'white__asian_18', 'white__native_hawaiian_and_other_pacific_islander_18', 'white__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native_18', 'black_or_african_american__asian_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__some_other_race_18', 'american_indian_and_alaska_native__asian_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander_18', 'asian__some_other_race_18', 'native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_three_races_18', 'white__black_or_african_american__american_indian_and_alaska_native_18', 'white__black_or_african_american__asian_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__some_other_race_18', 'white__american_indian_and_alaska_native__asian_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander_18', 'white__asian__some_other_race_18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__asian__some_other_race_18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'american_indian_and_alaska_native__asian__some_other_race_18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_four_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__asian__some_other_race_18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__american_indian_and_alaska_native__asian__some_other_race_18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_five_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'population_of_six_races_18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_18', 'total_hl18', 'hispanic_or_latino_hl18', 'not_hispanic_or_latino_hl18', 'population_of_one_race_hl18', 'white_alone_hl18', 'black_or_african_american_alone_hl18', 'american_indian_and_alaska_native_alone_hl18', 'asian_alone_hl18', 'native_hawaiian_and_other_pacific_islander_alone_hl18', 'some_other_race_alone_hl18', 'population_of_two_or_more_races_hl18', 'population_of_two_races_hl18', 'white__black_or_african_american_hl18', 'white__american_indian_and_alaska_native_hl18', 'white__asian_hl18', 'white__native_hawaiian_and_other_pacific_islander_hl18', 'white__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native_hl18', 'black_or_african_american__asian_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__some_other_race_hl18', 'american_indian_and_alaska_native__asian_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander_hl18', 'asian__some_other_race_hl18', 'native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_three_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native_hl18', 'white__black_or_african_american__asian_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__asian__some_other_race_hl18', 'white__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__asian__some_other_race_hl18', 'black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'american_indian_and_alaska_native__asian__some_other_race_hl18', 'american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_four_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__asian__some_other_race_hl18', 'white__black_or_african_american__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_five_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__some_other_race_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__black_or_african_american__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'white__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'population_of_six_races_hl18', 'white__black_or_african_american__american_indian_and_alaska_native__asian__native_hawaiian_and_other_pacific_islander__some_other_race_hl18', 'housing_total', 'housing_occupied', 'housing_vacant'],
}
census_columns['data'] = census_columns['1'] + census_columns['2']
census_columns['starts'] = 1 + np.insert(np.cumsum(census_columns['widths'])[:-1], 0, 0)