import struct
import numpy as np
import os
import struct
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

      
def write_file_centroids(file_path,centroids):
  
        with open(file_path, "ab") as fout:
            # Loop over each centroid and write its data along with the offset
            for  centroid in (centroids):
                  binary_data =struct.pack(f"{DIMENSION}f" ,*centroid.flatten())
                  fout.write(binary_data)
            
       
def read_file_centroids_with_memap(file_path):
        
            # Memory-map the file
            mmap_data = np.memmap(file_path, dtype=np.uint8, mode='r')
    

            # Precompute the struct formats
           

            # Calculate the number of centroids
            num_centroids = len(mmap_data) // (DIMENSION * ELEMENT_SIZE)
            
            # Reshape the mmap_data to a 2D array of centroids
            centroids = np.frombuffer(mmap_data, dtype=np.float32, count=num_centroids * DIMENSION).reshape(num_centroids, DIMENSION)
            return centroids
    

def write_file_records(file_path, data):
    
    binary_data = struct.pack("i", data)
    
    # Write the packed data to the file in append mode
    with open(file_path, "ab") as fout:
        fout.write(binary_data)
  


def read_file_records_mmap(file_path):
       # Memory-map the file
    mmap_data = np.memmap(file_path, dtype=np.uint8, mode='r')


    records = np.frombuffer(mmap_data, dtype=np.int32)
    return records