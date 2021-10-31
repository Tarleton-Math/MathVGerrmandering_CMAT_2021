from . import *

@dataclasses.dataclass
class Agg(Base):
    def __post_init__(self):
        super().__post_init__()
        self.tbls = dict()
        self.tbls['source'] = f'{data_bq}.{self.state.abbr}_{self.census_yr}_source_all'

    def aggegrate(self, agg_query, show=False):
        cols = get_cols(self.tbls['source'])
        a = cols.index('total_pop')
        b = cols.index('polygon')
        data_cols = cols[a:b]
        data_sums = [f'sum({c}) as {c}' for c in data_cols]
        
####### Builds a deeply nested SQL query to generate nodes
####### We build the query one level of nesting at a time store the "cumulative query" at each step
####### Python builds the SQL query using f-strings.  If you haven't used f-string, they are f-ing amazing.
####### Note we keep a dedicated "cntyvtd_temp" even though typically level = cntyvtd
####### so that, when we run with level != cntyvtd, we still have access to ctnyvtd via ctnyvtd_temp
        
####### Contraction can cause ambiguities.
####### Suppose some block of a cntyvtd are in county 1 while others are in county 2.
####### Or some blocks of a contracting county are in district A while others are in district B.
####### We will assign the contracted node to the county/district/cntyvtd that contains the largest population.
####### But because we need seats for other purposes AND seats is proportional to total_pop,
####### it's more convenient to implement this using seats in leiu of total_pop.
####### We must apply this tie-breaking rule to all categorical variables.
####### First, find the total seats in each (geoid_new, unit) intersection

        query = list()
        query.append(f"""
select
    A.*,
    B.geoid_new
from 
    {self.tbls['source']} as A
left join(
    {agg_query}
    ) as B
on
    A.geoid = B.geoid
""")
    
        geo = ['cntyvtd', 'county', 'cd', 'sldu', 'sldl']
        query.append(f"""
select
    *,
    {join_str().join([f'sum(total_pop) over (partition by geoid_new, {g}) as pop_{g}' for g in geo])}
from (
    {subquery(query[-1])}
    )
""")
####### Now, we find the max over all units in a given geoid ###########
        query.append(f"""
select
    *,
    {join_str().join([f'max(pop_{g}) over (partition by geoid_new) pop_{g}_max' for g in geo])},
from (
    {subquery(query[-1])}
    )
""")
####### Now, we create temporary columns that are null except on the rows of the unit achieving the max value found above
####### When we do the "big aggegration" below, max() will grab the name of the correct unit (one with max seat)
        query.append(f"""
select
    *,
    {join_str().join([f'case when pop_{g} = pop_{g}_max then {g} else null end as {g}_new' for g in geo])},
from (
    {subquery(query[-1])}
    )
""")
####### Time for the big aggregration step.
####### Join source, groupby geoid_new, and aggregate categorical variable with max, numerical variables with sum,
####### and geospatial polygon with st_union_agg.
        query.append(f"""
select
    geoid_new as geoid,
    {join_str().join([f'max({g}_new) as {g}' for g in geo])},
    sum(seats_cd) as seats_cd,
    sum(seats_sldu) as seats_sldu,
    sum(seats_sldl) as seats_sldl,
    {join_str().join(data_sums)},
    st_union_agg(polygon_simp) as polygon,
    sum(aland) as aland,
from (
    {subquery(query[-1])}
    )
group by
    geoid_new
""")
####### Get polygon perimeter #######
        query.append(f"""
select
    *,
    st_perimeter(polygon) / {m_per_mi} as perim,
from (
    {subquery(query[-1])}
    )
""")
####### Compute density, polsby-popper, and centroid #######
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
                print(f'\n=====================================================================================\nstage {k}')
                print(q)
        return query