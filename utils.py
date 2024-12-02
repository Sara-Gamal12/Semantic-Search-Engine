import struct
import numpy as np
def write_file_centroids(file_path,data):
        with open(file_path, "ab") as fout:
            # Pack the entire array into binary data
            binary_data = struct.pack(len(data)*f"{70}f", *data.flatten())
            fout.write(binary_data)
import struct

          
def read_file_centroids(file_path):
    # Calculate the size of each centroid (70 floats, each 4 bytes)
    dtype = np.float32
    centroid_size = 70

    # Create a memory-mapped NumPy array
    data = np.memmap(file_path, dtype=dtype, mode='r')

    # Reshape the array to have rows of 70 floats (each row represents a centroid)
    num_centroids = data.size // centroid_size
    centroids = data[:num_centroids * centroid_size].reshape(num_centroids, centroid_size)

    return centroids


def write_file_records(file_path, data):
    
    binary_data = struct.pack("i", data)
    
    # Write the packed data to the file in append mode
    with open(file_path, "ab") as fout:
        fout.write(binary_data)

def read_file_records(file_path):
    with open(file_path, "rb") as f:
        binary_data = f.read()

    # Calculate the number of records (each record is 70 floats + 1 integer index)
    record_size =  4 
    num_records = len(binary_data) // record_size

    # Unpack the data
    records = []
    for i in range(num_records):
        # Extract the binary chunk corresponding to one record (70 floats + 1 integer)
        record = binary_data[i*record_size:(i+1)*record_size]
        
        # Unpack the 70 floats and the integer index
        index = struct.unpack("i", record)[0]  # 1 integer
        
        records.append( index)

    return records

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