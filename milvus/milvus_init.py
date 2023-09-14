import numpy as np
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

fmt = "\n==={:30}===\n"
class milvus_wmap:
    def __init__(
        self,
        embedding_dim:int = 1024,
        drop_all:bool = False,
    ):
        print(fmt.format("start connecting to milvus"))
        self.connection = connections.connect("default", host="localhost", port="19530")
        has = utility.has_collection("wafermap")
        print(f"does collection wafermap exist in milvus:{has}")
        utility.drop_collection('waferMap')


        self.fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100,
            ),
            FieldSchema(name="index", dtype=DataType.INT64),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]

        self.schema = CollectionSchema(
            fields=self.fields, description="testing milvus for wafer map retrieval"
        )
        self.wmap = Collection(
            name="waferMap", schema=self.schema, 
        )

    def insert_data(self, vector_list: dict):
        # insert data
        print(fmt.format("Start inserting entities"))
        entities = [
            # [str(i) for i in range(num_entities)],
            [i for i in vector_list['index']],
            [i for i in vector_list['embeddings']],
        ]
        entities = [[[i], [v]] for i, v in zip(vector_list['index'], vector_list['embeddings'])]
        for i in entities:
            self.wmap.insert(i)
        self.wmap.flush()

    def create_index(
        self,
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
    ):
        self.wmap.create_index(field_name="embeddings", index_params=index_params)

    def search_wmap(
        self,
        vector_list: list,
        search_params={
            "metric_type": "L2",
            "params": {"nprobe": 10},
        },
        n_lims=3,
        output_fields=["index"],
    ):
        for v in vector_list:
            self.wmap.search(
                data=vector_list,
                anns_field="embeddings",
                params=search_params,
                limit=n_lims,
                output_fields=output_fields,
            )


# # init the database
# search_latency_fmt = "search latency = {:.4f}s"
# num_entities, dim = 3000, 1024

# print(fmt.format("start connecting to milvus"))
# connections.connect("default", host="localhost", port="19350")

# has = utility.has_collection("wafermap")
# print(f"does collection wafermap exist in milvus:{has}")

# fields = [
#     FieldSchema(
#         name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
#     ),
#     FieldSchema(name="index", dtype=DataType.DOUBLE),
#     FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
# ]

# schema = CollectionSchema(
#     fields=fields, description="testing milvus for wafer map retrieval"
# )
# wmap = Collection(name="waferMap", schema=schema, consistency_level="strong")

# # insert data
# print(fmt.format("Start inserting entities"))
# entities = [
#     [str(i) for i in range(num_entities)],
#     [i for i in range(num_entities)],
#     [i for i in results],
# ]

# insert_results = wmap.insert(entities)

# wmap.flush()
# print(f"number of entities in wmap:{wmap.num_entities}")

# # create index
# index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}

# wmap.create_index(field_name="embeddings", index_params=index_params)

# ########################################################################
# print(fmt.format("start loading"))
# wmap.load()


# def milvus_search(
#     vector_to_search, return_field: str, metric_type="L2", params={"nprobe": 10}
# ):
#     search_params = dict(metric_type=metric_type, params=params)
#     start_time = time.time()
#     result = wmap.search(
#         vector_to_search,
#         anns_field="embeddings",
#         params=search_params,
#         limit=10,
#         output_fields=return_field,
#     )
#     end_time = time.time()
#     search_time = search_latency_fmt, format(end_time - start_time)
#     return {"search_time": search_time, "item": result}
