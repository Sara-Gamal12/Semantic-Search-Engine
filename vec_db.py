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
from joblib import parallel_backend

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.no_centroids=0
        self.file_path="./clusters"

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        if self._get_num_records()==10**6:
           self.file_path="./clusters1"
        if self._get_num_records()==10*10**6:
           self.file_path="./clusters2"
        if self._get_num_records()==15*10**6:
           self.file_path="./clusters3"
        if self._get_num_records()==20*10**6:
           self.file_path="./clusters4"

    
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
            mmap_vector = np.memmap( self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"
    def get_rows(self, off,size) -> np.ndarray:
        # This function is only load one row in memory
        try:
            # off=r* DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(size, DIMENSION), offset=off)
           
            return np.array(mmap_vector)
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

            k =10
            top_centroids = self._get_top_centroids(query, k)
            # Initialize a list to store results
            results = []
            results1 = []
            for centroid in top_centroids:
                  ids = read_file_records_mmap(self.file_path + "/" + str(centroid[1]) + ".bin")
                  # ids=range(id,id+centroid[3])
                  data = np.array(self.get_rows(centroid[2],centroid[3]))
                  

                  # Compute cosine similarity for all vectors in the file
                  dot_products = np.dot(data, query.squeeze())  # Vectorized dot product
                  norms_data = np.linalg.norm(data, axis=1)  # Norms of the data vectors
                  norm_query = np.linalg.norm(query)  # Norm of the query vector
                  scores = dot_products / (norms_data * norm_query)  # Cosine similarity for all
                  # Append scores and IDs to a list

                  results.extend(zip(scores, ids))
                  results.sort( key=lambda x:- x[0])
                  results = results[:top_k]



                 
            top_k_ids = [result[1] for result in results]
            return top_k_ids

    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
      
        self.no_centroids = int(np.sqrt(self._get_num_records()))
       
        chuck_size = min(10**6,self._get_num_records())
        training_data=self.get_all_rows()[0:chuck_size]   
        kmeans = MiniBatchKMeans(n_clusters=self.no_centroids, random_state=0, batch_size=10**4)

        # Fit the model
        kmeans.fit(training_data)
        
        
        labels=kmeans.predict(self.get_all_rows())
        centroids= kmeans.cluster_centers_
        #save centroids in file
        if os.path.exists(self.file_path):
            shutil.rmtree(self.file_path)
        os.makedirs(self.file_path, exist_ok=True)
       
        all_rows=self.get_all_rows()
        unique_labels = np.unique(labels)
        offsets={}
        sizes={}
         #delete _temp.dat if exists
       
        sorted=[]
        offsets={}
        sizes={}
        for label in kmeans.labels_:
            offsets[label]=0    
            sizes[label]=0
        for label in tqdm.tqdm(unique_labels):
            flag=False
            indices = np.where(labels == label)[0]

            # data = all_rows[indices]
            size=0
           
            data=all_rows[indices]
            for index in (indices):
                size+=1
                write_file_records(self.file_path + "/" + str(label) + ".bin",  index)
                offset=DIMENSION*ELEMENT_SIZE*len(sorted)

                sorted.append(all_rows[index])
                if not flag:
                  offsets[label]=offset
                  flag=True
            
            sizes[label]=size
        

        write_file_centroids(self.file_path+"/centroids.bin",centroids,offsets,sizes)
        self._write_vectors_to_file(np.array(sorted))
          


            
    def _get_top_centroids(self, query, k):
          # Find the nearest centroids to the query
          centroids_data = read_file_centroids(self.file_path + "/centroids.bin")
          # Initialize a heap to store the centroids and their scores
          heap = []
          
          # Iterate over the centroids data, which contains (offset, size, centroid)
          for i,(offset, size, centroid) in enumerate(centroids_data):
              # Calculate the score for the current centroid
              score = self._cal_score(query, centroid)
              
              # Push the score, index, and the corresponding centroid metadata onto the heap
              heapq.heappush(heap, (score,i, offset, size ))
          
          # Get the top k centroids by score (largest score first)
          top_centroids = heapq.nlargest(k, heap)
          
          return top_centroids