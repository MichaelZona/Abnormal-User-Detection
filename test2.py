import os
import sys
import time
import torch
import random
import argparse
import numpy as np

a= np.arange(9).reshape(3,3)
print(a)
a = np.array([[0, 1, 2]
            ,[3, 0, 0]
            ,[0, 7, 0]])

b = a[:,1].nonzero()[0]
print(a[:,1].nonzero()[0])
print(a[b])