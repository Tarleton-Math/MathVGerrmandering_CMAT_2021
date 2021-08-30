from . import *
@dataclasses.dataclass
class Census(Variable):
    name: str = 'census'
    
    def __post_init__(self):
        self.yr = self.g.census_yr
        super().__post_init__()


    def get(self):
        self.url = f"https://www2.census.gov/programs-surveys/decennial/{self.yr}/data/01-Redistricting_File--PL_94-171/{self.g.state.name.replace(' ', '_')}/{self.g.state.abbr.lower()}{self.yr}.pl.zip"
        
        exists = super().get()
        if not exists['tbl']:
            if not exists['raw']:
                self.get_zip()
                rpt(f'creating raw table')
                self.process_raw()
            rpt(f'creating table')
            self.process()
        return self
    
    def add_header(self, file, header):
        cmd = 'sed -i "1s/^/' + '|'.join(header) + '\\n/" ' + file
        os.system(cmd)

    def process_raw(self):
######## In 2010 PL_94-171 involved 3 files - we first load each into a temp table ########
        for fn in self.zipfile.namelist():
            if fn[-3:] == '.pl':
                rpt(fn)
                file = self.zipfile.extract(fn)
######## Geo file is fixed width (not delimited) and must be handled carefully ########                
                if fn[2:5] == 'geo':
                    i = 'geo'
                else:
                    i = fn[6]
                schema = [bigquery.SchemaField(**col) for col in Census_columns[i]]
                tbl = self.raw+i
                delete_table(tbl)    
                with open(file, mode="rb") as f:
                    bqclient.load_table_from_file(f, tbl, job_config=bigquery.LoadJobConfig(field_delimiter='|', schema=schema)).result()
#                 os.unlink(fn)

######## combine census tables into one table ########
        rpt(f'joining')
        query = f"""
select
    concat(right(concat("00", A.state), 2), right(concat("000", A.county), 3), right(concat("000000", A.tract), 6), right(concat("0000", A.block), 4)) as geoid,
    {join_str(1).join(Census_columns['data'])}
from
    {self.raw}geo as A
inner join
    {self.raw}1 as B
on
    A.fileid = B.fileid
    and A.stusab = B.stusab
    and A.chariter = B.chariter
    and A.logrecno = B.logrecno
inner join
    {self.raw}2 as C
on
    A.fileid = C.fileid
    and A.stusab = C.stusab
    and A.chariter = C.chariter
    and A.logrecno = C.logrecno
inner join
    {self.raw}3 as D
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
        load_table(self.raw, query=query, preview_rows=0)
        
######## clean up ########
        delete_table(self.raw+'geo')
        delete_table(self.raw+'1')
        delete_table(self.raw+'2')
        delete_table(self.raw+'3')


    def process(self):
######## Use crosswalks to push 2010 data on 2010 tabblocks onto 2020 tabblocks ########
        if self.g.census_yr == self.g.shapes_yr:
            query = f"""
select
    geoid,
    {join_str(1).join(Census_columns['data'])}
from
    {self.raw}
"""

        else:
            query = f"""
select
    E.geoid_{self.g.shapes_yr} as geoid,
    {join_str(1).join([f'sum(D.{c} * E.aland_prop) as {c}' for c in Census_columns['data']])}
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
    sum(G.total_pop) over (partition by F.cntyvtd) as cntyvtd_pop,
    case when (sum(G.total_pop) over (partition by F.cntyvtd)) > 0 then G.total_pop / (sum(G.total_pop) over (partition by F.cntyvtd)) else 1 / (count(*) over (partition by F.cntyvtd)) end as cntyvtd_pop_prop,
from 
    {self.g.assignments.tbl} as F
inner join(
    {subquery(query)}
    ) as G
on
    F.geoid = G.geoid
order by
    geoid
"""
        load_table(self.tbl, query=query, preview_rows=0)