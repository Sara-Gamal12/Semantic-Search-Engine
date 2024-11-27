import struct
import numpy as np
def write_file_centroids(file_path,data):
        with open(file_path, "ab") as fout:
            # Pack the entire array into binary data
            binary_data = struct.pack(len(data)*f"{70}f", *data.flatten())
            fout.write(binary_data)
import struct

def read_file_centroids(file_path):
    with open(file_path, "rb") as f:
        binary_data = f.read()
        
    # Calculate the number of centroids based on the size of the file
    # Assuming each centroid is represented by 70 floats (4 bytes each)
    num_floats = len(binary_data) // (70 * 4)
    
    # Unpack the binary data
    data = struct.unpack(f"{num_floats*70}f", binary_data)
    
    # Reshape the unpacked data into a 2D array of centroids (num_floats x 70)
    centroids = [data[i:i+70] for i in range(0, len(data), 70)]
    
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