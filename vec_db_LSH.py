from typing import Dict, List, Annotated
import numpy as np
import os
import struct
from sklearn.cluster import KMeans
from utils import *
import tqdm
import heapq
import shutil
import sys
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.file_path="./hash"
        self.no_centroids=0
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def _vectorized_cal_score(self, vec1, vec2):
        vec2_broadcasted = np.broadcast_to(vec2, vec1.shape)

        # Calculate the dot product between each vector in vec1 and the broadcasted vec2
        dot_product = np.sum(vec1 * vec2_broadcasted, axis=1)
        # Calculate the dot product between each vector in vec1 and vec2
        # dot_product = np.dot(vec1, vec2.T)

        # Calculate the norm of each vector in vec1
        norm_vec1 = np.linalg.norm(vec1, axis=1)

        # Calculate the norm of vec2
        norm_vec2 = np.linalg.norm(vec2)

        # Calculate the cosine similarity for each pair of vectors
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

        return cosine_similarity.squeeze()


    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    def print_size(self,name, obj):
                print(f"{name}: {sys.getsizeof(obj)} bytes")

    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        # scores = []
        # num_records = self._get_num_records()
        # # here we assume that the row number is the ID of each vector
        # for row_num in range(num_records):
        #     vector = self.get_one_row(row_num)
        #     score = self._cal_score(query, vector)
        #     scores.append((score, row_num))
        # # here we assume that if two rows have the same score, return the lowest ID
        # scores = sorted(scores, reverse=True)[:top_k]
        # return [s[1] for s in scores]
            norm_input_query = query / np.linalg.norm(query)
            projections = np.dot(self.norm_random_vectors, norm_input_query.squeeze())
            hash_value = ''.join(['1' if p > 0 else '0' for p in projections])
            vectors=read_file_records_mmap(self.file_path + "/" + str(hash_value) + ".bin")
            data = np.array([v[0] for v in vectors])  # Extract vector data
            ids = np.array([v[1] for v in vectors])  # Extract vector IDs

            dot_products = np.dot(data, query.squeeze())  # Vectorized dot product

            norms_data = np.linalg.norm(data, axis=1)  # Norms of the data vectors
            scores = dot_products / (norms_data * np.linalg.norm(query))  # Cosine similarity for all
            results=[]
            # Append scores and IDs to a list
            results.extend((score, vector_id) for score, vector_id in zip(scores, ids))

            # Print sizes of key objects
            # self.print_size("vectors", vectors)
            # self.print_size("data", data)
            # self.print_size("ids", ids)
            # self.print_size("dot_products", dot_products)
            # self.print_size("norms_data", norms_data)
            # self.print_size("scores", scores)
            # self.print_size("results", results)

            # Convert results to a list, sort, and slice
            results = list(results)
            results.sort(reverse=True, key=lambda x: x[0])
            results=results[0:top_k]
            top_k_ids = [result[1] for result in results]

            return top_k_ids





           
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
       
        num_random_vectors = 8
        vector_size = 70
        if os.path.exists(self.file_path):
            shutil.rmtree(self.file_path)
        os.makedirs(self.file_path, exist_ok=True)
       
        # Step 1: Generate 8 random vectors of size 70
        rng = np.random.default_rng(DB_SEED_NUMBER)  # Use optional seed for reproducibility
        random_vectors = rng.random((num_random_vectors, vector_size), dtype=np.float32) * 2 - 1
        self.norm_random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=0)

        # Step 2: Compute hash values for each data point
        all_rows=self.get_all_rows()
        for i, point in enumerate(all_rows):
            # Compute dot products with random vectors
            norm_input_vector = point / np.linalg.norm(point)
            projections = np.dot(self.norm_random_vectors, norm_input_vector.squeeze())
            # Convert to binary hash: 1 if dot product > 0, else 0
            hash_value = ''.join(['1' if p > 0 else '0' for p in projections])
            print(hash_value)
            write_file_records(self.file_path + "/" + str(hash_value) + ".bin", (point, i))

