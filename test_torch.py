import torch
import numpy as np

arr= np.arange(100)
index = 5
b_sz = 3
nodes_batch = arr[index*b_sz:(index+1)*b_sz]        #[15, 18]
print(nodes_batch)