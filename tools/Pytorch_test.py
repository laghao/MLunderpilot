from __future__ import print_function
import torch

# Tensors

#x = torch.tensor([5.5, 3])
#x = x.new_ones(5, 3, dtype=torch.double) 
#print(x)
#x = torch.randn_like(x, dtype=torch.float)    # override dtype!
#print(x)                                      # result has the same size
#print(x.size())


# Operations

#y = torch.rand(5, 3)
#result = torch.empty(5, 3)
#torch.add(x, y, out=result)
#print(result)
#print(x[:, 2])
#x = torch.randn(4, 4)
#y = x.view(16)
#z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
#print(x.size(), y.size(), z.size())
#x = torch.randn(1)
#print(x)
#print(x.item())


# NumPy Bridge

#a = torch.ones(5)
#print(a)
#
#b = a.numpy()
#print(b)
#
#a.add_(1)
#print(a)
#print(b)


# Converting NumPy Array to Torch Tensor

#import numpy as np
#a = np.ones(5)
#b = torch.from_numpy(a)
#np.add(a, 1, out=a)
#print(a)
#print(b)

# CUDA Tensors
if torch.cuda.is_available():
   device = torch.device("cuda")
   y = torch.ones_like(x, device=device)
   x = x.to(device)
   z = x + y
   print(z)
   print(z.to("cpu", torch.double))
