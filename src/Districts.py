@dataclasses.dataclass
class Districts(Variable):
    def __post_init__(self):
        self.yr = self.g.census_yr
        self.level = self.g.level
        self.name = self.g.district_type
        self.nodes = self.g.nodes.df
        super().__post_init__()

    def get(self):
        print(f"Get {self.name} {self.state.abbr} {self.yr} {self.level} {self.dt}".ljust(32, ' '), end=concat_str)
        grp = self.nodes.sort_index().groupby(self.name)
        self.dict  = grp.groups
        self.keys  = tuple(self.dict.keys())
        self.tuple = tuple((k, tuple(v)) for k,v in self.dict.items())
        self.hash  = hash(self.tuple)
        self.pops  = grp['pop'].sum()
        self.pop_ideal = self.pops.mean()
        self.pop_imbalance = (self.pops.max() - self.pops.min()) / self.pop_ideal * 100
        return self