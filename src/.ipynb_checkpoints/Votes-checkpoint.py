@dataclasses.dataclass
class Votes(Variable):
    name: str = 'votes'
    group: str = 'all'
        
    def __post_init__(self):
        self.yr = self.g.shapes_yr
        check_group(self.group)
        self.name += f'_{self.group}'
        super().__post_init__()


    def get(self):
        self.raw = self.g.elections.tbl
        exists = super().get()
        if not exists['tbl']:
            if not exists['raw']:
                raise Exception(f'can not find {self.raw} - must create elections object first')
            print(f'creating table', end=concat_str)
            self.process()


    def process(self):
######## To bring everything into one table, we must pivot the long table ########
######## in Elections to a wide format with one row for each tabblock ########
######## To bring everything into one table, we must pivot the long table ########
######## While easy in Python and Excel, this is quite delicate in SQl ########
######## given the number of distinct election and tabblocks ########
        df = run_query(f"select distinct election from {self.raw}")
        elections = tuple(sorted(df['election']))
######## Even BigQuery refused to pivot all elections simulatenously ########
######## So we break the elections into chunks, pivot separately, then join horizontally ########
        stride = 100
        tbls = list()
        c = 64 # silly hack to give table aliases A, B, C, ...
        for r in np.arange(0, len(elections), stride):
            E = elections[r:r+stride]
            t = f'{self.tbl}_{r}'
            tbls.append(t)
            query = f"""
select
    *
from (
    select
        geoid,
        election,
        {self.name}
    from
        {self.raw}
    )
pivot(
    sum({self.name})
    for election in {E})
"""
            load_table(t, query=query, preview_rows=0)
            
######## create the join query as we do each chunk so we can run it at the end ########
            c += 1        
            if len(tbls) == 1:
                query_join = f"""
select
    A.geoid,
    {join_str(1).join(elections)}
from
    {t} as A
"""
            else:
                alias = chr(c)
                query_join += f"""
inner join
    {t} as {alias}
on
    A.geoid = {alias}.geoid
"""
        query_join += f"order by geoid"
        
######## clean up ########
        load_table(self.tbl, query=query_join, preview_rows=0)
        for t in tbls:
            delete_table(t)