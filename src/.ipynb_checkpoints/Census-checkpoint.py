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