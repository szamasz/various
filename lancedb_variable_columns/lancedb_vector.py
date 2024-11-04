import lancedb
import lance
import pandas as pa
import pyarrow as py
import uuid
import shutil


home_dir = "./data/lancedb/"
db_name = "lancedb_test"
dataset_name = "text"
path_7day = home_dir+"parquet/7day.parquet"
path_1day = home_dir+"parquet/1day.parquet"

# Function that selects only features(f*) from list of columns in DF
prepare_feature_list_Fs = lambda columns: [col for col in columns if col.startswith('f')]

def create_dataset(df_py,path):
    lance.write_dataset(df_py,path)
    lance_db = lance.dataset(path)
    return lance_db

def get_lancedb_dataset(home_dir,dataset_name,df,feature_selection_f):
    """
    If lancedb doesn't exist yet, use vectorized version of DF to build db together with metadata about vector columns
    """
    path = f"{home_dir}dbs/{dataset_name}"
    try:
        lance_dataset = lance.dataset(path)
    except ValueError:
        print(f"Dataset {dataset_name} doesn't exist yet, creating in {path}")
        df_vct = vectorize_df(df,feature_selection_f)
        df_py = pa2py_with_metadata(df_vct)
        lance_dataset = create_dataset(df_py,path)
    return lance_dataset

def delete_tmp_tables(home_dir,dataset_name):
    """
    Delete tmp tables
    """
    path = f"{home_dir}dbs/{dataset_name}/tmp/"
    shutil.rmtree(path)
    print(f"Tmp dir deleted: {path}")

def vectorize_df(df, feature_selection_f):
    """
    Transform Pandas DF into new DF where all feature columns (f*) are compressed into 1 column 'vector' and then discard
    Additionally metadata about selected feature columns stored in 'vector' is stored in attribute 'selected_columns'
    """
    selected_columns = feature_selection_f(df.columns)
    dfc = df.copy()
    dfc['vector'] = dfc.loc[:,selected_columns].apply(lambda row: row.values,axis=1)
    res_df = dfc[['user','vector']]
    res_df.selected_columns = ",".join(selected_columns)
    return res_df

def extract_distance(row):
    """
    If user doesn't exist in one of the compared DFs, return -1, otherwise return result of lancedb distance search
    """
    if len(row)>0:
        return row[0]['_distance']
    else:
        return -1
    
def pa2py_with_metadata(df):
    """
    Transforms Pandas DF into pyarrow DF and adds metadata about selected columns into pyarrow schema metadata
    """
    df_py = py.Table.from_pandas(df)
    new_meta = df_py.schema.metadata | {'vector_columns': df.selected_columns}
    df_py_meta = df_py.replace_schema_metadata(new_meta)
    return df_py_meta

def return_tmp_table(df,path,feature_f):
    """
    Creates temporary table build from provided Pandas DF
    """
    path = path + f"/tmp/"
    df_vct = vectorize_df(df,feature_f)
    df_py_meta = pa2py_with_metadata(df_vct)
    db_connection = lancedb.connect(path)
    lance_table = db_connection.create_table(str(uuid.uuid4()),df_py_meta)
    return lance_table

def df_from_vector(df, db_columns):
    """
    Reverses result of vectorize_df, ie creates DF from vector using provided column names
    """
    db_columns = db_columns
    def _df_from_vector(row):
        return {'user':row[0]}|{c:v for c,v in zip(db_columns,row[1])}
    dicts = df[['user','vector']].apply(lambda row: _df_from_vector(row),axis=1)
    df_db_dicts = pa.DataFrame(list(dicts.values))
    return df_db_dicts

def normalize_columns(lncdb_dataset,df_1day,id_column,feature_selection_f):
    """
    Reconciles columns from reference and current datasets. At the moment result set is intersection of columns existing in both DFs. 
    If any of DFs has more columns, they are limited to common columns. For ref dataset it creates temp table
    """
    ref_columns = set(lncdb_dataset.schema.metadata[b'vector_columns'].decode('utf-8').split(","))
    cur_columns = set(feature_selection_f(df_1day.columns))
    common_columns = list(ref_columns & cur_columns)+[id_column]
    if len(common_columns) < len(cur_columns)+1:
        print(f"Current DF must be changed, subcolumns selected: {common_columns}")
        df_1day = df_1day[common_columns]
    df_1day_v = vectorize_df(df_1day, feature_selection_f)
    
    db_df = lncdb_dataset.to_table().to_pandas()
    db_devect = df_from_vector(db_df,ref_columns)
    if len(common_columns) < len(ref_columns) + 1:
        print(f"Ref DF must be changed, subcolumns selected: {common_columns}")
        db_devect = db_devect[common_columns]
    lance_tbl= return_tmp_table(db_devect,lncdb_dataset.uri,feature_selection_f)
    return lance_tbl, df_1day_v

def matches_against_ref(ref_tbl,df_1day,id_column,metric="cosine"):
    """
    Searches for distance defined by 'metric' for every user in current DF. 
    It assumes that ref and current DFs are vectorized and vectors are of equal length
    """
    df_copy = df_1day.copy()
    df_copy['distance'] = df_copy.loc[:,[id_column,'vector']] \
        .apply(lambda row: ref_tbl.search(row.values[1]).metric(metric).where(f"{id_column}=\'{row.values[0]}\'", prefilter=True).to_list(),axis=1) \
        .apply(lambda row: extract_distance(row))
    return df_copy.sort_values(by=['distance'],ascending=False)[[id_column,'distance']]

def find_similar_users(ref_tbl,df_1day,id_column,id_value,metric="cosine",limit=10):
    vct = list(df_1day[df_1day[id_column] == id_value]['vector'].to_list()[0])
    similar_users = ref_tbl.search(vct).metric(metric).limit(limit).to_list()
    df_similar_users = pa.DataFrame.from_dict(similar_users)[[id_column,'_distance']]
    print(f"Similar objects to {id_column}: {id_value}")
    print(df_similar_users.head(limit))

if __name__=="__main__":
    
    metric = "l2"
    N=10
    
    df_1day = pa.read_parquet(path_1day)
    df_7day = pa.read_parquet(path_7day)

    print(f'Loading ref data')

    lance_dataset = get_lancedb_dataset(home_dir,dataset_name,df_7day,prepare_feature_list_Fs)
    
    print(f'Loading current data')

    print(f"Finding matches with metric function {metric} for equal dataframes")
    lance_tbl,df_cur = normalize_columns(lance_dataset,df_1day,'user',prepare_feature_list_Fs)
    res = matches_against_ref(lance_tbl,df_cur,'user',metric=metric)
    print(f"Showing {N} best_matches:")
    print(res[:N])
    print("\n------------------------------------------------\n")

    print(f"Finding anomalies with metric function {metric} with missing feature")
    df_1day_smaller = df_1day.drop(columns = ['f15', 'f10'], axis = 1)
    lance_tbl,df_cur = normalize_columns(lance_dataset,df_1day_smaller,'user',prepare_feature_list_Fs)
    res = matches_against_ref(lance_tbl,df_cur,'user',metric=metric)
    print(f"Showing {N} best_matches:")
    print(res[:N])
    print("\n------------------------------------------------\n")

    print(f"Finding anomalies with metric function {metric} with extra feature")
    df_1day['f77'] = df_1day['f1'] * 3
    lance_tbl,df_cur = normalize_columns(lance_dataset,df_1day,'user',prepare_feature_list_Fs)
    res = matches_against_ref(lance_tbl,df_cur,'user',metric=metric)
    print(f"Showing {N} best_matches:")
    print(res[:N])

    delete_tmp_tables(home_dir,dataset_name)
