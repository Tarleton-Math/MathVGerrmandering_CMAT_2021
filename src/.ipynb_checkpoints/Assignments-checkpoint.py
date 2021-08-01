@dataclasses.dataclass
class Assignments(Variable):
    name: str = 'assignments'
    
    def __post_init__(self):
        self.yr = self.g.shapes_yr
        super().__post_init__()


    def get(self):
        self.url = f"https://www2.census.gov/geo/docs/maps-data/data/baf"
        if self.yr == 2020:
            self.url += '2020'
        self.url += f"/BlockAssign_ST{self.state.fips}_{self.state.abbr.upper()}.zip"
        
        exists = super().get()
        if not exists['tbl']:
            self.get_zip()
            print(f'creating table', end=concat_str)
            self.process()


    def process(self):
        L = []
        for fn in self.zipfile.namelist():
            col = fn.lower().split('_')[-1][:-4]
            if fn[-3:] == 'txt' and col != 'aiannh':
                df = extract_file(self.zipfile, fn, sep='|')
                if col == 'vtd':
                    df['countyfp'] = df['countyfp'].str.rjust(3, '0') + df['district'].str.rjust(6, '0')
                    col = 'cntyvtd'
                df = df.iloc[:,:2]
                df.columns = ['geoid', col]
                L.append(df.set_index('geoid'))
                os.unlink(fn)
        self.df = lower(pd.concat(L, axis=1).reset_index()).sort_values('geoid')
        c = ['cd', 'sldu', 'sldl']
        self.df[c] = self.df[c].astype(int)
        c = self.df['geoid'].str
        self.df.insert(1, 'state'   , c[:2])
        self.df.insert(1, 'cnty'    , c[:5])
        self.df.insert(1, 'tract'   , c[:11])
        self.df.insert(1, 'bg'      , c[:12])
        self.df.insert(1, 'tabblock', c[:15])
        self.df.to_parquet(self.pq)
        load_table(tbl=self.tbl, df=self.df, preview_rows=0)