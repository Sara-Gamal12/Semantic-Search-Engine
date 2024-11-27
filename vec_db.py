from typing import Dict, List, Annotated
import numpy as np
import os
import struct
from sklearn.cluster import KMeans
from utils import *
import tqdm
import heapq
import shutil

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.file_path="./clusters"
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

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
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
        k=self.no_centroids//20
        top_centroids=self._get_top_centroids(query, k)
        #make priority queue with size top-k
        heap=[]

        for centroid in top_centroids:
            vectors=read_file_records(self.file_path + "/" + str(centroid[1])+".bin")
            for vector in (vectors) :
                score=self._cal_score(query,vector[0])
                if len(heap)<top_k:
                  heapq.heappush(heap,(score,vector[1]))
                else:
                    heapq.heappushpop(heap,(score,vector[1]))
        
        heap.sort(reverse=True)
        top_k_ids = [id for _, id in heap]
    
        return top_k_ids
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
      
        self.no_centroids = int(np.sqrt(self._get_num_records()))
        if self._get_num_records()==10**2:
            chuck_size=10
        if self._get_num_records()==10**3:
            chuck_size=100
        elif self._get_num_records()==10**6:
            chuck_size=10**5

        chuck_size = self._get_num_records()//self.no_centroids
        training_data=self.get_all_rows()[0:chuck_size]   

        kmeans=KMeans(n_clusters=self.no_centroids, random_state=0).fit(training_data)
        labels=kmeans.predict(self.get_all_rows())
        centroids= kmeans.cluster_centers_
        #save centroids in file
        if os.path.exists(self.file_path):
            shutil.rmtree(self.file_path)
        os.makedirs(self.file_path, exist_ok=True)
       
        write_file_centroids(self.file_path+"/centroids.bin",centroids)
    
        
        for i,label in tqdm.tqdm(enumerate(labels)):
                data=(self.get_one_row(i),i)
                write_file_records(self.file_path + "/" + str(label)+".bin",data)

            

            
            
    def _get_top_centroids(self, query, k):
            # find the nearest centroids to the query
            centroids = read_file_centroids(self.file_path+"/centroids.bin")
            heap = []
            for i, centroid in enumerate(centroids):
                score = self._cal_score(query, centroid)
                heapq.heappush(heap, (score, i))
            top_centroids = heapq.nlargest(k, heap)
            return top_centroids



