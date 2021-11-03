import json, google.cloud.bigquery, numpy as np, pandas as pd, geopandas as gpd, plotly.express as px
pd.set_option('display.max_columns', None)
path = '/home/jupyter/VRA'

cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient = google.cloud.bigquery.Client(credentials=cred, project=proj)
crs_census = 'EPSG:4269'

races = ['white', 'black', 'hisp']
parties = ['red', 'blue']

def run_query(query):
    return bqclient.query(query).result().to_dataframe()

def prep_data(level='cntyvtd'):
    tbl = f'cmat-315920.VRA.{level}'
    df = run_query(f'select * from {tbl}')
    geo = gpd.GeoSeries.from_wkt(df.pop('polygon'), crs=crs_census).simplify(0.003).buffer(0) #<-- little white space @ .001 ~5.7 mb, minimal at .0001 ~10mb, with no white space ~37mb
    
    s = '_red_pct'
    Elections = {x.replace(s,'') for x in df.columns if s in x}
    L = []
    for elect in Elections:
        repl = {f'{elect}_{attr}':attr for attr in ['votes', 'red_pct']}
        cols = ['county', 'vap_pop'] + [f'vap_{r}_pct' for r in ['black','hisp','white']]
        cols += repl.keys()
        X = df[cols].rename(columns=repl)
        X.insert(0, 'election', elect)
        L.append(X)
    df = pd.concat(L)
    df.insert(df.shape[1]-1, 'vote_rate', df['votes'] / df['vap_pop'] * 100)
    df['vote_rate'].fillna(0, inplace=True)
    df['red_pct'].fillna(50, inplace=True)
    df['blue_pct'] = 100 - df['red_pct']
    for p in parties:
        df[f'{p}_votes'] = (df['votes'] * df[f'{p}_pct']     / 100).astype(int)
    for r in races:
        df[f'{r}_votes'] = (df['votes'] * df[f'vap_{r}_pct'] / 100).astype(int)
    return df, geo

def compute_red_pct(df, white_red_pct=100, black_red_pct=10):
    df['white_red_votes'] = df['white_votes'] * white_red_pct / 100
    df['black_red_votes'] = df['black_votes'] * black_red_pct / 100
    df['hisp_red_votes' ] = np.clip(df['red_votes'] - df['white_red_votes'] - df['black_red_votes'], 0, df['hisp_votes'])
    df['white_red_pct']   = white_red_pct
    df['black_red_pct']   = black_red_pct
    hisp_red_pct_overall  =  100 * df['hisp_red_votes'].sum() / df[f'hisp_votes'].sum()
    df['hisp_red_pct']    = (100 * df['hisp_red_votes'] / df[f'hisp_votes']).fillna(hisp_red_pct_overall).round().astype(int)
    return df

def make_maps(df):
    x_range = [-106.645646, -93.508292]
    y_range = [25.837377, 36.500704]
    aspect = (x_range[1]-x_range[0])/(y_range[1]-y_range[0])
    height = 500
    width = height * aspect
    lon = (x_range[1]+x_range[0])/2
    lat = (y_range[1]+y_range[0])/2
    cmap = 'bluered'
    cmap = 'jet'
    
    for idx, data in df.groupby(['election', 'white_red_pct']):
        fig = px.choropleth_mapbox(data, geojson=geojson, locations='index',
                                   color = 'hisp_red_pct',
                                   animation_frame = 'black_red_pct',
                                   color_continuous_scale=cmap,
                                   range_color = (0, 100),
                                   hover_name = 'county',
                                   mapbox_style="carto-positron",
                                   zoom=3.8, 
                                   center = {"lat": lat, "lon": lon},
                                   width=width,
                                   height=height,
                                   opacity=0.5,
                                   labels={'hisp_red_pct' :'Hispanic-Red% min',
                                           'black_red_pct':'Black-Red% est',
                                           'white_red_pct':'White-Red% est',
                                          },
                                   hover_data = {'index':False, 'black_red_pct':False,}
                                  )
        title   = f'{idx[0]}   White-Red% est={idx[1]}'
        figfile = f'{path}/{level}/hisp_red_min_{level}_{idx[0]}_{str(idx[1]).rjust(3,"0")}.html'
        print(figfile)
        fig.update_layout(title_text=title, title_x=0.5, title_y=0.92)
        with open(figfile, 'w') as f:
            f.write(fig.to_html(auto_play=False))
    return fig


# for level in ['county', 'cntyvtd']:
for level in ['county']:
    df, geo = prep_data(level)
    geojson = json.loads(geo.to_json())
    L = list()
    for b in range(0, 101, 5):
        for w in range(0, 101, 10):
            X = compute_red_pct(df.copy(), black_red_pct=b, white_red_pct=w)#[['county', hr]]
            cols = [
                'election', 'county', 'white_red_pct', 'black_red_pct', 'hisp_red_pct',
                'vap_pop', 'votes', 'vote_rate', 'red_pct', 'blue_pct',
                # 'vap_black_pct', 'vap_hisp_pct', 'vap_white_pct',
                # 'white_votes', 'black_votes', 'hisp_votes',
                # 'white_red_votes', 'black_red_votes', 'hisp_red_votes',
            ]
            L.append(X[cols])
    data = pd.concat(L).reset_index()
    make_maps(data)
