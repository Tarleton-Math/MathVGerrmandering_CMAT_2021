from . import *
import networkx as nx
proposal_default = 'enacted2010'

@dataclasses.dataclass
class MCMC(Base):
    proposal: str = proposal_default
    contract: str = '0'
        
    def __post_init__(self):
        self.Sources = ('proposal', 'nodes', 'graph', 'results')
        super().__post_init__()
        self.refresh_all.discard('proposal')

        self.tbl  = dict()
        self.pq   = dict()
        self.path = dict()
        stem = f'{self.state.abbr}_{self.census_yr}'
        self.tbl['source'] = f'{data_bq}.{stem}_source_all'
        
#         stem += f'_{self.proposal}'
#         self.tbl['proposal'] = f'{data_bq}.{stem}'
#         stem += f'_{self.level}_{self.contract}'

#         stem += f'_{self.district_type}_{self.proposal}_{self.level}_{self.contract}'
#         for src in self.Sources:
#             self.tbl [src] = f'{data_bq}.{stem}_{src}'
#             self.pq  [src] = data_path/f'{stem}/{stem}_{src}.parquet'
#             self.path[src] = self.pq[src].parent
    
        stem += f'_{self.district_type}_{self.proposal}'
        for src in self.Sources:
            s = stem
            if src == 'proposal':
                s += f'_{self.level}_{self.contract}'
            
            self.tbl [src] = f'{data_bq}.{s}_{src}'
            self.pq  [src] = data_path/f'{stem}/{s}_{src}.parquet'
            self.path[src] = self.pq[src].parent


        for src in ('proposal',):
            rpt(f'Get {src}'.ljust(rpt_just, ' '))
            self.delete_for_refresh(src)
            self[f'get_{src}']()
            print(f'success!')
            os.chdir(code_path)

        
#         for src in self.Sources:
#             stem += f'_{self.level}_{self.contract}_{src}'
#             self.tbl [src] = f'{data_bq}.{stem}'
#             self.pq  [src] = data_path / f'{src}/{stem}.parquet'
#             self.path[src] = self.pq[src].parent
            
#         self.seats = f'seats_{self.district_type}'
#         self.get_nodes()
        
        
    def get_proposal(self):
        src = self.proposal
        if self.proposal != proposal_default:
            if check_table(self.tbl['proposal']):
                rpt('using existing proposal table')
            else:
                csv = data_path / f'proposals/{self.district_type}/{self.proposal}.csv'
                rpt(f'creating proposal table from {csv}')
                df = pd.read_csv(csv, skiprows=1, names=('geoid', self.district_type), dtype={'geoid':str})
                load_table(self.tbl['proposal'], df=df)

        
        
    def get_nodes(self, show=True):
        # Builds a deeply nested SQL query to generate nodes
        # We build the query one level of nesting at a time store the "cumulative query" at each step
        # Python builds the SQL query using f-strings.  If you haven't used f-string, they are f-ing amazing.
        # Get critical columns from source table
        # Note we keep a dedicated "cntyvtd_temp" even though typically level = cntyvtd
        # so that, when we run with level != cntyvtd, we still have access to ctnyvtd via ctnyvtd_temp
        query = list()
        query.append(f"""
select
    geoid,
    {self.level},
    cast({self.district_type} as int) as district_2010,
    substring(cnty,3) as cnty,
    county,
    cntyvtd as cntyvtd_temp,
    {self.seats} as seats,
from
    {self.tbl['source']}
""")


        # join the proposal table if specificied; else use currently enacted plan
        if self.proposal == proposal_default:
            query.append(f"""
select
    A.*,
    A.district_2010 as district,
from (
    {subquery(query[-1])}
    ) as A
""")
    
        else:
            cols = get_cols(self.tbl['proposal'])
            query.append(f"""
select
    A.*,
    cast(B.{cols[1]} as int) as district,
from (
    {subquery(query[-1])}
    ) as A
inner join
    {self.tbl['proposal']} as B
on
    A.geoid = cast(B.{cols[0]} as string)
""")

        # source data is at the census block level, but our MCMC usually runs at the cntyvtd level
        # So, we already need one round of contraction to combined all blocks in a cntyvtd into a single node.
        # However, we may want a second round of contraction combining all cntyvtds in a "small" county into a single node.
        # Here are several options for this second contraction, which I'll call "county contraction".
    
        # No county contraction
        if str(self.contract) == '0':
            query.append(f"""
select
    geoid,
    {self.level} as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    {subquery(query[-1])}
    )
""")

        # Contract county iff it was wholly contained in a single district in 2010
        elif self.contract == proposal_default:
            query.append(f"""
select
    geoid,
    case when ct = 1 then cnty else {self.level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {self.level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        count(distinct district_2010) over (partition by cnty) as ct,
    from (
        {subquery(query[-1], indents=2)}
        )
    )
""")
        
    
        # Contract county iff it is wholly contained in a single district in the proposed plan
        elif self.contract == 'proposal':
            query.append(f"""
select
    geoid,
    case when ct = 1 then cnty else {self.level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {self.level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        count(distinct district) over (partition by cnty) as ct,
    from (
        {subquery(query[-1], indents=2)}
        )
    )
""")
    
    
        # Contract county iff its seats_share < contract / 10
        # seats_share = county pop / ideal district pop
        # ideal district pop = state pop / # districts
        # Note: contract = "tenths of a seat" rather than "seats" so that contract is an integer
        # Why? To avoid decimals in table & file names.  No other reason.
        else:
            try:
                c = int(self.contract) / 10
            except:
                raise Exception(f'contract must be "proposal" or "{proposal_default}" or an integer >= 0 ... got {self.contract}')
            query.append(f"""
select
    geoid,
    case when seats_temp < {c} then cnty else {self.level} end as geoid_new,
    district,
    county,
    cntyvtd_temp as cntyvtd,
    seats,
from (
    select
        geoid,
        {self.level},
        district,
        cnty,
        county,
        cntyvtd_temp,
        seats,
        sum(seats) over (partition by cnty) as seats_temp,
    from (
        {subquery(query[-1], indents=2)}
        )
    )
""")


        # Contraction can cause ambiguities.
        # Suppose some block of a cntyvtd are in county 1 while others are in county 2.
        # Or some blocks of a contracting county are in district A while others are in district B.
        # We will assign the contracted node to the county/district/cntyvtd that contains the largest population.
        # But because we need seats for other purposes AND seats is proportional to total_pop,
        # it's more convenient to implement this using seats in leiu of total_pop.
        # We must apply this tie-breaking rule to all categorical variables.

        # First, find the total seats in each (geoid_new, unit) intersection
        query.append(f"""
select
    *,
    sum(seats) over (partition by geoid_new, district) as seats_district,
    sum(seats) over (partition by geoid_new, county  ) as seats_county,
    sum(seats) over (partition by geoid_new, cntyvtd ) as seats_cntyvtd,
from (
    {subquery(query[-1])}
    )
""")


        # Now, we find the max over all units in a given geoid
        query.append(f"""
select
    *,
    max(seats_district) over (partition by geoid_new) seats_district_max,
    max(seats_county  ) over (partition by geoid_new) seats_county_max,
    max(seats_cntyvtd ) over (partition by geoid_new) seats_cntyvtd_max,
from (
    {subquery(query[-1])}
    )
""")
    

        # Now, we create temporary columns that are null except on the rows of the unit achieving the max value found above
        # When we do the "big aggegration" below, max() will grab the name of the correct unit (one with max seat)
        query.append(f"""
select
    *,
    case when seats_district = seats_district_max then district else null end as district_new,
    case when seats_county   = seats_county_max   then county   else null end as county_new,
    case when seats_cntyvtd  = seats_cntyvtd_max  then cntyvtd  else null end as cntyvtd_new,
from (
    {subquery(query[-1])}
    )
""")


        # Time for the big aggregration step.
        # Get names of the remaining data columns of source
        cols = get_cols(self.tbl['source'])
        a = cols.index('total_pop_prop')
        b = cols.index('aland')
        # Create a list of sum statements for these columns to use in the select
        sels = ',\n    '.join([f'sum({c}) as {c}' for c in cols[a:b]])

        # Join source, groupby geoid_new, and aggregate categorical variable with max, numerical variables with sum,
        # and geospatial polygon with st_union_agg.
        query.append(f"""
select
    A.geoid_new as geoid,
    max(district_new) as district,
    max(county_new  ) as county,
    max(cntyvtd_new ) as cntyvtd,
    {sels},
    st_union_agg(polygon) as polygon,
    sum(aland) as aland
from (
    {subquery(query[-1])}
    ) as A
inner join
    {self.tbl['source']} as B
on
    A.geoid = B.geoid
group by
    geoid_new
""")


        # Get polygon perimeter
        query.append(f"""
select
    *,
    st_perimeter(polygon) as perim,
from (
    {subquery(query[-1])}
    )
""")


        # Compute density, polsby-popper, and centroid.
        query.append(f"""
select
    *,
    case when perim > 0 then round(4 * {np.pi} * aland / (perim * perim) * 100, 2) else 0 end as polsby_popper,
    case when aland > 0 then total_pop / aland else 0 end as density,
    st_centroid(polygon) as point,
from (
    {subquery(query[-1])}
    )
""")

        if show:
            for k, q in enumerate(query):
                print(f'\n\nquery {k}')
                print(q)
    
#         return query[-1]
