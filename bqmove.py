from src import *
for src_tbl in bqclient.list_tables(f'{proj_id}.results'):
    w = src_tbl.table_id.split('_')
    results_stem = f'{proj_id}.{"_".join(w[:4])}'
    dest = f'{results_stem}.{src_tbl.table_id}'
    try:
        bqclient.get_table(dest)
    except:
        print(f'copying {src_tbl.full_table_id} to {dest}')
        bqclient.create_dataset(results_stem, exists_ok=True)
        bqclient.copy_table(src_tbl, dest).result()
