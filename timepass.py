import os
import numpy as np

def check_image_dimensions(data_directory):
    for filename in os.listdir(data_directory):
        if filename.endswith('.bin'):
            data = np.fromfile(os.path.join(data_directory, filename), dtype=np.float32)
            print(f"{filename}: Total elements = {data.size}")

check_image_dimensions("E:\git\Imaginari\.quickdrawcache")