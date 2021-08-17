import google, time, datetime, dataclasses, typing, os, pathlib, shutil, urllib
import zipfile as zf, numpy as np, pandas as pd, geopandas as gpd, networkx as nx
import matplotlib.pyplot as plt, plotly.express as px
from shapely.ops import orient
from google.cloud import aiplatform, bigquery
import warnings
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
warnings.filterwarnings('ignore', message='.*Pyarrow could not determine the type of columns*')
try:
    from google.cloud.bigquery_storage import BigQueryReadClient
except:
    os.system('pip install --upgrade google-cloud-bigquery-storage')
    from google.cloud.bigquery_storage import BigQueryReadClient
import Core
    
# proj_id = 'cmat-315920'
# root_path = '/home/jupyter'
# code_path = root_path + '/MathVGerrmandering_CMAT_2021/src/'

# user_name = 'cook'
# i = input(f'user_name (default={user_name})')
# if i != '':
#     user_name = i


# default_random_seed = 1
# random_seed = input(f'random_seed (default={default_random_seed})')
# try:
#     rng = np.random.default_rng(int(random_seed))    
# except:
#     rng = np.random.default_rng(default_random_seed)

# pd.set_option('display.max_columns', None)
# cred, proj = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
# bqclient   = bigquery.Client(credentials=cred, project=proj)
# root_path  = pathlib.Path(root_path)
# data_path  = root_path / 'redistricting_data'
# bq_dataset = proj_id   +'.redistricting_data'

# Levels = ['tabblock', 'bg', 'tract', 'cnty', 'state', 'cntyvtd']
# District_types = ['cd', 'sldu', 'sldl']
# Years = [2010, 2020]
# Groups = ['all', 'hl']
# concat_str = ' ... '

# try:
#     states
# except:
#     print('getting states')
#     states = get_states()    








# print(states)