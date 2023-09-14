import requests
from milvus_init import milvus_wmap 
import pandas as pd 
import pathlib
#import numpy as np
#import json

def inference(data, endpoint):
    input_data = {'data':open(data, 'rb')}
    return pd.DataFrame.from_dict((requests.post(endpoint, files=input_data).json())).values[:,0]

def batch_insert(data_file:pd.DataFrame, prediction_endpoint:str, wmap_db:milvus_wmap):
    result_list = [inference(data, endpoint=prediction_endpoint) for data in data_file.data.tolist()]
    wmap_data = {'index':data_file.index.tolist(), 'embeddings':result_list}
    wmap_db.insert_data(wmap_data)
    wmap_db.create_index()
    return wmap_db

def create_dataset(data_file:str):
    p = pathlib.Path(data_file)
    files = list(p.glob('**/*.jpg'))
    index = range(len(files))
    return pd.DataFrame(data = {'data':files, 'index':index})

def search(index:int, wmap_db:milvus_wmap):
    wmap_db.search_wmap(inference(data_path,endpoint=pre)


if __name__ == '__main__':
    data = create_dataset(data_file='/Users/yu-shengkuo/cloudML/waferMap/pytorch/structured/python_package/inference/prediction_data')
    print(data)
    batch_insert(data_file=data, prediction_endpoint='http://localhost:8080/predictions/model',wmap_db=milvus_wmap(drop_all=True))

