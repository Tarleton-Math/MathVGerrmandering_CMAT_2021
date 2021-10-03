from . import *
@dataclasses.dataclass
class Assignments(Variable):
    name: str = 'assignments'
    
    def __post_init__(self):
        self.yr = self.n.shapes_yr
        super().__post_init__()


    def get(self):
        self.url = f"https://www2.census.gov/geo/docs/maps-data/data/baf"
        if self.yr == 2020:
            self.url += '2020'
        self.url += f"/BlockAssign_ST{self.n.state.fips}_{self.n.state.abbr.upper()}.zip"
        
        exists = super().get()
        if not exists['tbl']:
            self.get_zip()
            rpt(f'creating table')
            self.process()
        return self


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
        c = self.df['geoid'].str
        self.df['state']    = c[:2]
        self.df['cnty']     = c[:5]
        self.df['tract']    = c[:11]
        self.df['bg']       = c[:12]
        self.df['tabblock'] = c[:15]
        self.df['cd_prop']   = self.df['cd']
        self.df['sldu_prop'] = self.df['sldu']
        self.df['sldl_prop'] = self.df['sldl']
        self.df = self.df[['geoid', 'tabblock', 'bg', 'tract', 'cnty', 'cntyvtd', 'cd', 'cd_prop', 'sldu', 'sldu_prop', 'sldl', 'sldl_prop']]
        load_table(tbl=self.tbl, df=self.df, preview_rows=0)
        self.save_tbl()