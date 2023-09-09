import numpy as np
import pickle
import pathlib
import json
from typing import Union, Sequence
from google.cloud import aiplatform, aiplatform_v1
from google.cloud import storage
import io
import pandas as pd
from PIL import Image as im

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


def list_blobs(bucket_name, prefix, output_file):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)

    # Note: The call returns a response only when the iterator is consumed.
    with open(output_file, 'w') as f:
        f.writelines(f"gs://{blob.bucket.name}/{blob.name}\n"for blob in blobs)

def ndarray_to_list(x:np.ndarray):
  return x.tolist()

def create_gcp_instance(file_path: str, index: list, output_file:str, local=False):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    a = pd.DataFrame()
    a.to_json
    data = data.query('failureType!="none"').reset_index().iloc[index]
    if local:
        data.waferMap.head(1).to_json(output_file, orient='records',)
    else:
       data.waferMap.head(1).to_json(output_file, orient='records', lines=True)
    #    with open(output_file) as f:
    #     for i in result:
    #         data = json.dumps(i)
    #         f.write(data)
    #     f.close()
   
import matplotlib.pyplot as plt
def save_np_pil(arr, path, grayscaling=64):
   arr*=grayscaling
   data = im.fromarray(arr)
   data.save(path)

def create_gcp_image_instance(file_path:str, index:list, output_dir:str, file_prefix='wmap', local=False):
   file = open(file_path,'rb')
   data = pickle.load(file)
   data = data.query('failureType!="none"').reset_index().iloc[index]
   p = pathlib.Path(output_dir)
   p.mkdir(parents=True, exist_ok=True)
   print(p)
   wmap = data.waferMap.tolist()
   if local:
    _ = [save_np_pil(arr,p/(f"{file_prefix}_{i}.jpg") ) for i, arr in enumerate(wmap)]
   else:
    _ = [upload_blob(arr,(f"{output_dir}/{file_prefix}_{i}.jpg") ) for i, arr in enumerate(wmap)]



def create_batch_prediction_job_dedicated_resources_sample(
    project: str,
    location: str,
    model_resource_name: str,
    job_display_name: str,
    gcs_source: Union[str, Sequence[str]],
    gcs_destination: str,
    instances_format: str ="jsonl",
    machine_type: str = "n1-standard-2",
    #accelerator_count: int = None,
    #accelerator_type: Union[str, aiplatform_v1.AcceleratorType] = None,
    starting_replica_count: int = 1,
    max_replica_count: int = 1,
    sync: bool = True,
):
    aiplatform.init(project=project, location=location)

    my_model = aiplatform.Model(model_resource_name)

    batch_prediction_job = my_model.batch_predict(
        job_display_name=job_display_name,
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination,
        instances_format=instances_format,
        machine_type=machine_type,
        #accelerator_count=accelerator_count,
        #accelerator_type=accelerator_type,
        starting_replica_count=starting_replica_count,
        max_replica_count=max_replica_count,
        sync=sync,
    )

    batch_prediction_job.wait()

    print(batch_prediction_job.display_name)
    print(batch_prediction_job.resource_name)
    print(batch_prediction_job.state)
    return batch_prediction_job

def upload_prediction_to_gcs(bucket:str, dest:str, file:str):
   bucket=storage.Client().bucket(bucket_name=bucket)
   blob = bucket.blob(dest)
   blob.upload_from_filename(file)

if __name__ == "__main__":
   local = False
   #list_blobs(bucket_name='wmap-811k', prefix='prediction_data/', output_file='prediction_file.txt')
   #create_gcp_image_instance(file_path='WM811K.pkl',index = range(0,100), output_dir='prediction_data')
#   create_gcp_instance(file_path='WM811K.pkl',index = range(0,1), output_file='wm811k.json',local=local)
#    upload_prediction_to_gcs(bucket='wmap-811k',dest='prediction.json',file='wm811k.jsonl')
   create_batch_prediction_job_dedicated_resources_sample(
      project='ykwafer-retreival',
      location='us-west1', 
      model_resource_name='6064176063091572736', 
      job_display_name='test',
      gcs_source='gs://wmap-811k/prediction_file.txt',
      gcs_destination='gs://wmap-811k/prediction_results',
      instances_format='file-list'
   )
   
   

