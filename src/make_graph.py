proj_id = 'cmat-315920'

import pandas as pd, geopandas as gpd, networkx as nx

cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
bqclient   = google.cloud.bigquery.Client(credentials=cred, project=proj)



quer
