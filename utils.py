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
    # Assume data is a tuple (array_of_floats, index)
    array, index = data
    
    # Flatten the array and combine it with the index
    # Pack the 70 floats from the array and the integer index into binary
    binary_data = struct.pack(f"{70}f", *array) + struct.pack("i", index)
    
    # Write the packed data to the file in append mode
    with open(file_path, "ab") as fout:
        fout.write(binary_data)

def read_file_records(file_path):
    with open(file_path, "rb") as f:
        binary_data = f.read()

    # Calculate the number of records (each record is 70 floats + 1 integer index)
    record_size = 70 * 4 + 4  # 70 floats (4 bytes each) + 1 integer (4 bytes)
    num_records = len(binary_data) // record_size

    # Unpack the data
    records = []
    for i in range(num_records):
        # Extract the binary chunk corresponding to one record (70 floats + 1 integer)
        record = binary_data[i*record_size:(i+1)*record_size]
        
        # Unpack the 70 floats and the integer index
        array = struct.unpack(f"{70}f", record[:70*4])  # 70 floats
        index = struct.unpack("i", record[70*4:])[0]  # 1 integer
        
        records.append((array, index))

    return records

def read_file_records_mmap(file_path):
    # Use np.memmap to memory-map the file (assuming float32 for the vector and int32 for the index)
    # 70 floats (4 bytes each) + 1 integer (4 bytes) per record
    record_size = 70 * 4 + 4  # 70 floats + 1 integer

    # Memory-map the file
    mmap_data = np.memmap(file_path, dtype=np.uint8, mode='r')

    # Calculate the number of records in the file
    num_records = len(mmap_data) // record_size

    records = []
    for i in range(num_records):
        # Extract the binary chunk corresponding to one record
        record = mmap_data[i * record_size:(i + 1) * record_size]

        # Unpack the 70 floats (each of size 4 bytes) and the integer (4 bytes)
        array = np.frombuffer(record[:70 * 4], dtype=np.float32)  # 70 floats
        index = struct.unpack("i", record[70 * 4:])[0]  # 1 integer
        
        records.append((array, index))

    return records