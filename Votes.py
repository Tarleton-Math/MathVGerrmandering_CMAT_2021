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
        
        

#     def get_level(self):
#         if self.level == 'tabblock':
#             return
#         query = f"""
# select
#     *,
#     case when votes > 0 then spanish_surname_votes / votes else 0 end as spanish_surname_prop
# from (
#     select
#         A.{self.level} as geoid,
#         B.office,
#         B.election_yr,
#         B.race,
#         B.candidate,
#         B.party,
#         B.election,
#         sum(B.votes) as votes,
#         sum(B.registered_voters) as registered_voters,
#         sum(B.turn_out) as turn_out,
#         sum(B.spanish_surname_votes) as spanish_surname_votes,
#         sum(B.spanish_surname_registered_voters) as spanish_surname_registered_voters,
#         sum(B.spanish_surname_turn_out) as spanish_surname_turn_out,
#     from 
#         {self.g.assignments.tbl_blk} as A
#     inner join
#         {self.tbl_blk} as B
#     on
#         A.geoid = B.geoid
#     group by
#         geoid, B.election_yr, B.race, B.office, B.candidate, B.party, B.election
#     )
# order by
#     geoid
# """
#         load_table(self.tbl_level, query=query, preview_rows=0)

        
#     def long_to_wide(self, variable):
#         L = ['votes', 'spanish_surname_votes']
#         assert variable in L, f'variable must be one of {L}, got {variable}'
        
#         df = run_query(f"select distinct election from {self.tbl}")
#         elections = tuple(sorted(df['election']))
#         stride = 100
        
#         tbls = list()
#         c = 65
#         for r in np.arange(0, len(elections), stride):
#             E = elections[r:r+stride]
#             t = f'{self[variable]}_{r}'
#             tbls.append(t)
#             query = f"""
# select
#     *
# from (
#     select
#         geoid,
#         election,
#         {variable}
#     from
#         {self.tbl}
#     )
# pivot(
#     sum({variable})
#     for election in {E})
# """

#             if len(tbls) == 1:
#                 query_join = f"""
# select
#     A.geoid,
#     {join_str(1).join([f'coalesce({e}, 0) as {e}' for e in elections])}
# from
#     {t} as A
# """
#             else:
#                 alias = chr(c)
#                 query_join += f"""
# inner join
#     {t} as {alias}
# on
#     A.geoid = {alias}.geoid
# """
#             load_table(t, query=query, preview_rows=0)
#             c += 1
#         query_join += f"order by geoid"
#         load_table(self[variable], query=query_join, preview_rows=0)
#         for t in tbls:
#             bqclient.delete_table(t, not_found_ok=True)
#         query_join = f"""
# select
#     A.geoid,
#     {join_str(1).join([f'coalesce({e}, 0) as {e}' for e in elections])}
# from
#     {tbls[0]} as A
# """
#         c = 65
#         for t in tbls[1:]:
#             c += 1
#             alias = chr(c)
#             query_join += f"""
# inner join
#     {t} as {alias}
# on
#     A.geoid = {alias}.geoid
# """
#         query_join += f"order by geoid"
