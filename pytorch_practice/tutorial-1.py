from __future__ import print_function
import torch

# a 2D tensor of values = 0.
# 5 rows, 3 columns
x = torch.empty(5, 3)

print(x)

# a 2D tensor of randomly drawn floats between 0 and 1
# 5 rows, 3 columns
x = torch.rand(5, 3)
print(x)

# a 2D tensor of long values = 0
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# new_* methods take in sizes
x = x.new_ones(5, 3, dtype=torch.double) 
print(x)

# override dtype, with a tensor having values based on a 
# Gaussian distribution with mean=0, and std=1.0
x = torch.randn_like(x, dtype=torch.float) 
# result has the same size
print(x)
print(x.size())


y = x.new_ones(5, 3, dtype=torch.double)
print(x,'\n',y)

z = torch.add(x, y)
print(z)



# providing an output tensor as an argument
result = torch.empty(5, 3)
print(result)
torch.add(x, y, out=result)
print(result)

# torch tensors can be indexed like numpy arrays
print(y,x)
y.add_(x)
print(y)
print(y[0,:])
print(y[:,0])


# Resizing via torch.view()
x = torch.randn(4, 4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1,8)
print(z)
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

# Convert a torch tensor to a numpy array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# see how the numpy array changed in value
a.add_(1)
print(a)
print(b)


# Convert a numpy array to a torch tensor
import numpy as np
a = np.ones(5)
b1 = torch.from_numpy(a)
b2 = torch.from_numpy(np.copy(a))
np.add(a,1,out=a)
print(a)
print(b1)
print(b2)




