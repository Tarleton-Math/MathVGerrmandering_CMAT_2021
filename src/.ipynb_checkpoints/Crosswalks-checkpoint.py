@dataclasses.dataclass
class Crosswalks(Variable):
    name: str = 'crosswalks'
        
    def __post_init__(self):
        self.yr = self.g.census_yr
        super().__post_init__()


    def get(self):
        self.url = f"https://www2.census.gov/geo/docs/maps-data/data/rel{self.g.shapes_yr}/t{str(self.g.census_yr)[2:]}t{str(self.g.shapes_yr)[2:]}/TAB{self.g.census_yr}_TAB{self.g.shapes_yr}_ST{self.state.fips}.zip"
        
        exists = super().get()
        if not exists['tbl']:
            self.get_zip()
            print(f'creating table', end=concat_str)
            self.process()


    def process(self):
        yrs = [self.g.census_yr, self.g.shapes_yr]
        ids = [f'geoid_{yr}' for yr in yrs]

        for fn in self.zipfile.namelist():
            df = extract_file(self.zipfile, fn, sep='|')
            for yr, id in zip(yrs, ids):
                df[id] = df[f'state_{yr}'].str.rjust(2,'0') + df[f'county_{yr}'].str.rjust(3,'0') + df[f'tract_{yr}'].str.rjust(6,'0') + df[f'blk_{yr}'].str.rjust(4,'0')
            os.unlink(fn)
        df['arealand_int'] = df['arealand_int'].astype(float)
        df['A'] = df.groupby(ids[0])['arealand_int'].transform('sum')
        df['aland_prop'] = (df['arealand_int'] / df['A']).fillna(0)
        self.df = df[ids+['aland_prop']].sort_values(ids[1])
        self.df.to_parquet(self.pq)
        load_table(tbl=self.tbl, df=self.df, preview_rows=0)