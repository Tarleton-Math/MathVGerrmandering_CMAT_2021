@dataclasses.dataclass
class Districts(Variable):
    def __post_init__(self):
        self.name = self.g.district_type
        self.yr = self.g.census_yr
        self.level = self.g.level
        self.graph = self.g.graph.graph
        self.pop_imbalance_tol = self.g.pop_imbalance_tol
        super().__post_init__()

    def get(self):
        print(f"Get {self.name} {self.state.abbr} {self.yr} {self.level} {self.district_type}".ljust(33, ' '), end=concat_str)
        self.update()
        H = nx.Graph()
        for i, N in self.tuple:
            for j, M in self.tuple:
                if i < j:
                    if len(nx.node_boundary(self.graph, N, M)) > 0:
                        H.add_edge(i, j)
        self.color = self.g.get_colors(H)
        self.stats['color'] = self.color
        load_table(tbl=self.tbl, df=self.stats.reset_index(), preview_rows=0)
        return self
        
    def update(self):
        grp = self.g.nodes.df.groupby(self.name)
        self.dict  = {k:tuple(v) for k,v in grp.groups.items()}
        self.tuple = tuple(self.dict.items())
        self.hash  = hash(self.tuple)
        self.pops  = grp['total_pop'].sum()
        self.pop_ideal = self.pops.mean()
        self.pop_imbalance = (self.pops.max() - self.pops.min()) / self.pop_ideal * 100

        self.stats = pd.DataFrame().rename_axis('geoid')
        H = nx.Graph()
        for i, N in self.tuple:
            nodes = self.g.nodes.df.loc[list(N)]
            edges = self.g.edges.df.query(f'geoid_x in {N} and geoid_y in {N}')
            self.stats.loc[i, 'total_pop'] = nodes['total_pop'].sum()
            self.stats.loc[i, 'aland']     = nodes['aland'].sum()
            self.stats.loc[i, 'perim']     = nodes['perim'].sum() - 2*edges['shared_perim'].sum()
        self.stats['polsby_popper'] = 4 * np.pi * self.stats['aland'] / (self.stats['perim']**2) * 100
        self.stats.insert(0, 'plan', 0)
        try:
            self.stats['color'] = self.color
        except:
            pass
        
        self.summary = {'plan' : 0,
                        'pop_imbalance' : self.pop_imbalance,
                        'polsby_popper' : self.stats['polsby_popper'].mean()
                       }
        return self