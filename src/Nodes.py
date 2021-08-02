@dataclasses.dataclass
class Nodes(Variable):
    name: str = 'nodes'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        self.attrs = listify(self.g.node_attrs) + listify(Districts)
        super().__post_init__()


    def get(self):
        exists = super().get()
        if not exists['df']:
            self.df = (read_table(self.g.combined.tbl, cols=self.attrs)
                       .rename(columns={'total':'pop'}).set_index('geoid'))