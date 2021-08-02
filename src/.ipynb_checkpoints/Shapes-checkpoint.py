@dataclasses.dataclass
class Shapes(Variable):
    name: str = 'shapes'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        super().__post_init__()


    def get(self):
        self.url = f"https://www2.census.gov/geo/tiger/TIGER{self.yr}/{self.level.upper()}"
        if self.yr == 2010:
            self.url += '/2010'
        elif self.yr == 2020 and self.level == 'tabblock':
            self.url += '20'
        self.url += f"/tl_{self.yr}_{self.state.fips}_{self.level}{str(self.yr)[-2:]}"
        if self.yr == 2020 and self.level in ['tract', 'bg']:
            self.url = self.url[:-2]
        self.url += '.zip'
        
        exists = super().get()
        if not exists['tbl']:
            if not exists['raw']:
                self.get_zip()
                print(f'creating raw table', end=concat_str)
                self.process_raw()
            print(f'creating table', end=concat_str)
            self.process()


    def process_raw(self):
        for fn in self.zipfile.namelist():
            self.zipfile.extract(fn)
        a = 0
        chunk_size = 50000
        while True:
            print(f'starting row {a}', end=concat_str)
            df = lower(gpd.read_file(self.path, rows=slice(a, a+chunk_size)))
            df.columns = [x[:-2] if x[-2:].isnumeric() else x for x in df.columns]
            df = df[['geoid', 'aland', 'geometry']]#, 'intptlat', 'intptlon']]
            df['geometry'] = df['geometry'].apply(lambda p: orient(p, -1))
            load_table(self.raw, df=df.to_wkb(), overwrite=a==0)
            if df.shape[0] < chunk_size:
                break
            else:
                a += chunk_size
        for fn in self.zipfile.namelist():
            os.unlink(fn)


    def process(self):                
        query = f"""
select
    geoid,
    cast(aland as float64) as aland,
    st_geogfrom(geometry) as geography
from
    {self.raw}
order by
    geoid
"""
        load_table(self.tbl, query=query, preview_rows=0)