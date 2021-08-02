@dataclasses.dataclass
class Elections(Variable):
    name: str = 'elections'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        super().__post_init__()


    def get(self):
        if self.state.abbr != 'TX':
            print(f'elections only implemented for TX')
            return

        self.url = f'https://data.capitol.texas.gov/dataset/aab5e1e5-d585-4542-9ae8-1108f45fce5b/resource/253f5191-73f3-493a-9be3-9e8ba65053a2/download/{self.yr}-general-vtd-election-data.zip'
        exists = super().get()

        if not exists['tbl']:
            if not exists['raw']:
                self.get_zip()
                print(f'creating raw table', end=concat_str)
                self.process_raw()
            print(f'creating table', end=concat_str)
            self.process()

        
    def process_raw(self):
######## Most races have 2 files with the extensions below - merge horizontally ########
######## Then stack vertically into single dataframe ########
        ext = ['_Returns.csv', '_VRTO.csv']
        k = len(ext[0])
        files = [fn[:-k] for fn in self.zipfile.namelist() if fn[-k:]==ext[0]]
        L = []
        for fn in files:
            msg = fn.ljust(50, ' ')
            try:
                A = extract_file(self.zipfile, fn+ext[0], sep=',')
                B = extract_file(self.zipfile, fn+ext[1], sep=',')
                df = A.merge(B, on='cntyvtd', suffixes=(None,'_y'))
                df = (df.drop(columns=[c for c in df.columns if c[-2:]=='_y'])
                      .query("party in ['R', 'D', 'L', 'G']")
                      .astype({'votes':int, 'fips':str, 'vtd':str})
                      .query('votes > 0')
                     )
                w = fn.lower().split('_')
                df['election_yr'] = int(w[0])
                df['race'] = "_".join(w[1:-1])
                L.append(df)
                os.unlink(fn+ext[0])
                os.unlink(fn+ext[1])
#                 print(msg+'success')
            except:  # ignore races which don't have both files
#                 print(msg+'fail')
                pass
        
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
        
        self.df = (df.drop(columns=['county', 'fips', 'vtd', 'incumbent', 'totalpop', 'alt'])
                     .rename(columns={'name':'candidate', 'totalvr':'registered_voters', 'totalto':'turn_out', 'spanishsurnamepercent':'spanish_surname_pct'})
                  )
        self.df.to_parquet(self.pq)
        load_table(self.raw, df=self.df, preview_rows=0)
        

    def process(self):
######## Apportion votes from cntyvtd to its tabblock proportional to population ########
######## We computed cntyvtd_pop_prop = pop_tabblock / pop_cntyvtd  during census processing ########
######## Each tabblock gets this proportion of votes cast in its cntyvtd ########
######## Moreover, TX Legislative Council provides "spanish_surname_percent" ########
######## We assume the same voting rate among spanish_surname and general population ########
######## Assumes uniform voting behaviors across cntyvtd and between spanish_surname/general population ########
######## Obviously this is a crude approximation, but I don't see a better method ########
######## to do apportionment with the available data ########
        sep = ' or\n        '
        query = f"""
select
    *,
    spanish_surname_prop * votes_all as votes_hl,
    spanish_surname_prop * registered_voters_all as registered_voters_hl,
    spanish_surname_prop * turn_out_all as turn_out_hl,
from (
    select
        A.geoid,
        B.office,
        B.election_yr,
        B.race,
        B.candidate,
        B.party,
        concat(B.office, "_", B.election_yr, "_", B.race, "_", B.party, "_", B.candidate) as election,
        B.votes * A.cntyvtd_pop_prop as votes_all,
        cast(B.registered_voters as int) as registered_voters_all,
        cast(B.turn_out as int) as turn_out_all, 
        cast(B.spanish_surname_pct as float64) / 100 as spanish_surname_prop,
    from 
        {self.g.census.tbl} as A
    inner join
        {self.raw} as B
    on
        A.cntyvtd = B.cntyvtd
    where
        {sep.join(f'({x})' for x in self.g.election_filters)}
    )
order by
    geoid
"""
        load_table(self.tbl, query=query, preview_rows=0)