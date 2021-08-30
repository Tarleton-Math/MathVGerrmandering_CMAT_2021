from . import *
@dataclasses.dataclass
class Crosswalks(Variable):
    name: str = 'crosswalks'
        
    def __post_init__(self):
        self.yr = 2010
        super().__post_init__()


    def get(self):
        self.url = f"https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/TAB2010_TAB2020_ST{self.g.state.fips}.zip"
        exists = super().get()
        if not exists['tbl']:
            self.get_zip()
            rpt(f'creating table')
            self.process()
        return self

#         if self.yr < 2020:
#             if not exists['tbl']:
#                 self.get_zip()
#                 rpt(f'creating table')
#                 self.process()
#         else:
#             rpt(f'not necessary')
#         return self


    def process(self):
        yrs = [2010, 2020]
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