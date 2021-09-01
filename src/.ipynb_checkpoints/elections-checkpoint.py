from . import *
@dataclasses.dataclass
class Elections(Variable):
    name: str = 'elections'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        super().__post_init__()


    def get(self):
        if self.g.state.abbr != 'TX':
            print(f'elections only implemented for TX')
            return

        self.url = f'https://data.capitol.texas.gov/dataset/aab5e1e5-d585-4542-9ae8-1108f45fce5b/resource/253f5191-73f3-493a-9be3-9e8ba65053a2/download/{self.yr}-general-vtd-election-data.zip'
        exists = super().get()

        if not exists['tbl']:
            if not exists['raw']:
                self.get_zip()
                rpt(f'creating raw table')
                self.process_raw()
            rpt(f'creating table')
            print(self.tbl)
            self.process()
        return self

        
    def process_raw(self):
        ext = '_Returns.csv'
        k = len(ext)
        L = []
        for fn in self.zipfile.namelist():
            if fn[-k:]==ext:
                rpt(fn)
                df = extract_file(self.zipfile, fn, sep=',')
                df = (df.astype({'votes':int, 'fips':str, 'vtd':str})
                      .query('votes > 0')
                      .query("party in ['R', 'D', 'L', 'G']")
                     )
                w = fn.lower().split('_')
                df['election_yr'] = int(w[0])
                df['race'] = "_".join(w[1:-2])
                L.append(df)
                os.unlink(fn)
        
######## vertically stack then clean so that joins work correctly later ########
        df = pd.concat(L, axis=0, ignore_index=True).reset_index(drop=True)
        f = lambda col: col.str.replace(".", "", regex=False).str.replace(" ", "", regex=False).str.replace(",", "", regex=False).str.replace("-", "", regex=False).str.replace("'", "", regex=False)
        df['name'] = f(df['name'])
        df['office'] = f(df['office'])
        df['race'] = f(df['race'])
        df['fips'] = df['fips'].str.lower()
        df['vtd']  = df['vtd'] .str.lower()

######## correct differences between cntyvtd codes in assignements (US Census) and elections (TX Legislative Council) ########
        c = f'cntyvtd'
        df[c]     = df['fips'].str.rjust(3, '0') + df['vtd']         .str.rjust(6, '0')
        df['alt'] = df['fips'].str.rjust(3, '0') + df['vtd'].str[:-1].str.rjust(6, '0')
        assign = read_table(self.g.assignments.tbl)[c].drop_duplicates()
        
        # find cntyvtd in elections not among assignments
        unmatched = ~df[c].isin(assign)
        # different was usually a simple character shift
        df.loc[unmatched, c] = df.loc[unmatched, 'alt']
        # check for any remaining unmatched
        unmatched = ~df[c].isin(assign)
        if unmatched.any():
            display(df[unmatched].sort_values('votes', ascending=False))
            raise Exception('Unmatched election results')
        
        self.df = df.drop(columns=['fips', 'vtd', 'incumbent', 'alt']).rename(columns={'name':'candidate'})
        self.df.to_parquet(self.pq)
        load_table(self.raw, df=self.df, preview_rows=0)
        

    def process(self):
######## Apportion votes from cntyvtd to its tabblock proportional to population ########
######## We computed cntyvtd_pop_prop = pop_tabblock / pop_cntyvtd  during census processing ########
######## Each tabblock gets this proportion of votes cast in its cntyvtd ########

        sep = ' or\n        '
        query = f"""
select
    A.geoid,
    B.county,
    concat(B.office, "_", B.election_yr, "_", B.race, "_", B.party, "_", B.candidate) as election,
    B.votes * A.cntyvtd_pop_prop as votes,
from 
    {self.g.census.tbl} as A
inner join
    {self.raw} as B
on
    A.cntyvtd = B.cntyvtd
where
    {sep.join(f'({x})' for x in self.g.election_filters)}
order by
    geoid
"""
        tbl_temp = self.tbl + '_temp'
        load_table(tbl_temp, query=query, preview_rows=0)

######## To bring everything into one table, we must pivot from long to wide format (one row per tabblock) ########
######## While easy in Python and Excel, this is delicate in SQl given the number of electionS and tabblocks ########
######## Even BigQuery refuseS to pivot all elections simulatenously ########
######## So we break the elections into chunks, pivot separately, then join horizontally ########
        df = run_query(f"select distinct election from {tbl_temp}")
        elections = tuple(sorted(df['election']))
        stride = 100
        tbl_chunks = list()
        alias_chr = 64 # silly hack to give table aliases A, B, C, ...
        for r in np.arange(0, len(elections), stride):
            E = elections[r:r+stride]
            t = f'{self.tbl}_{r}'
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
    {join_str(1).join(elections)}
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
        query_join += f"order by geoid"

######## clean up ########
        load_table(self.tbl, query=query_join, preview_rows=0)
        delete_table(tbl_temp)
        for t in tbl_chunks:
            delete_table(t)