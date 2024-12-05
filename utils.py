import struct
import numpy as np
import os
import struct
ELEMENT_SIZE = np.dtype(np.float32).itemsize

      
def write_file_centroids(file_path,centroids):
  
        with open(file_path, "ab") as fout:
            # Loop over each centroid and write its data along with the offset
            for  centroid in (centroids):
                  binary_data =struct.pack(f"{70}f" ,*centroid.flatten())
                  fout.write(binary_data)
            
         
def read_file_centroids(file_path):
  
        with open(file_path, "rb") as f:
            binary_data = f.read()
        
        centroids = []
        idx = 0
        while idx < len(binary_data):
            centroid = struct.unpack_from(f"{70}f", binary_data, idx )  # Read 280 bytes for centroid
            centroids.append((centroid))
            idx += 280  # Move the index forward by the size of one entry

        return centroids
  
       
def read_file_centroids_with_memap(file_path):
        
            # Memory-map the file
            mmap_data = np.memmap(file_path, dtype=np.uint8, mode='r')
    
            # Unpack the binary data: offset (int), size (int), centroid (70 floats)
            centroids = []
            idx = 0

            # Precompute the struct formats
           

            while idx < len(mmap_data):
                centroid = struct.unpack_from(f"{70}f", mmap_data, idx )  # Read 280 bytes for centroid
                centroids.append((centroid))
                idx += 280  # Move the index forward by the size of one entry
            return centroids
    

def write_file_records(file_path, data):
    
    binary_data = struct.pack("i", data)
    
    # Write the packed data to the file in append mode
    with open(file_path, "ab") as fout:
        fout.write(binary_data)
  


def read_file_records_mmap(file_path):
   
    record_size =  4 
    # Memory-map the file
    mmap_data = np.memmap(file_path, dtype=np.uint8, mode='r')

    # Calculate the number of records in the file
    num_records = len(mmap_data) // record_size

    records = []
    for i in range(num_records):
        # Extract the binary chunk corresponding to one record
        record = mmap_data[i * record_size:(i + 1) * record_size]

        index = struct.unpack("i", record)[0]  # 1 integer
        
        records.append( index)
    return records