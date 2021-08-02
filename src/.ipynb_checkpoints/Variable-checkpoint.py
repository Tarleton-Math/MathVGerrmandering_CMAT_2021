@dataclasses.dataclass
class Variable(Base):
    g     : typing.Any
    name  : str
    level : str = 'tabblock'

    def __post_init__(self):
        self.state = self.g.state
        a = f'{self.name}/{self.state.abbr}'
        self.path = data_path / a
        a = a.replace('/', '_')
        b = f'{a}_{self.yr}'
        c = f'{b}_{self.level}'
        d = f'{c}_{self.g.district}'
        self.zip     = self.path / f'{b}.zip'
        self.pq      = self.path / f'{b}.parquet'
        self.raw    = f'{bq_dataset}.{b}_raw'
        self.tbl    = f'{bq_dataset}.{c}'
        self.gpickle = self.path / f'{d}.gpickle'

        if self.name in self.g.refresh_tbl:
            delete_table(self.tbl)
            
        if self.name in self.g.refresh_all:
            delete_table(self.tbl)
            delete_table(self.raw)
            shutil.rmtree(self.path, ignore_errors=True)
        self.get()
        print(f'success')
#         delete_table(self.raw)


    def get_zip(self):
        try:
            self.zipfile = zf.ZipFile(self.zip)
            print(f'zip exists', end=concat_str)
        except:
            try:
                self.path.mkdir(parents=True, exist_ok=True)
                os.chdir(self.path)
                print(f'getting zip from {self.url}', end=concat_str)
                self.zipfile = zf.ZipFile(urllib.request.urlretrieve(self.url, self.zip)[0])
                print(f'finished{concat_str}processing', end=concat_str)
            except urllib.error.HTTPError:
                raise Exception(f'n\nFAILED - BAD URL\n\n')


    def get(self):
        print(f"Get {self.name} {self.state.abbr} {self.yr} {self.level}".ljust(32, ' '), end=concat_str)
        exists = dict()
        
        exists['df'] = hasattr(self, 'df')
        if exists['df']:
            print(f'dataframe exists', end=concat_str)
        
        exists['tbl'] = check_table(self.tbl)
        if exists['tbl']:
            print(f'{self.level} table exists', end=concat_str)
        else:
            exists['raw'] = check_table(self.raw)
            if exists['raw']:
                print(f'raw table exists', end=concat_str)
        return exists