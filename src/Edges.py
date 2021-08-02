@dataclasses.dataclass
class Edges(Variable):
    name: str = 'edges'

    def __post_init__(self):
        self.yr = self.g.shapes_yr
        self.level = self.g.level
        super().__post_init__()


    def get(self):
        exists = super().get()
        if not exists['df']:
            if not exists['tbl']:
                print(f'creating table', end=concat_str)
                self.process()
            self.df = read_table(self.tbl)


    def process(self):
        query = f"""
select
    *
from (
    select
        x.geoid as geoid_x,
        y.geoid as geoid_y,        
        st_distance(x.point, y.point) as distance,
        st_length(st_intersection(x.geography, y.geography)) as shared_perim
    from
        {self.g.combined.tbl} as x,
        {self.g.combined.tbl} as y
    where
        x.geoid < y.geoid and st_intersects(x.geography, y.geography)
    )
where
    shared_perim > 0.1
order by
    geoid_x, geoid_y
"""
        load_table(self.tbl, query=query, preview_rows=0)