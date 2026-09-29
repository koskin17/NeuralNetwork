import torch


# print("First Tensor:", torch.tensor([1, 2, 3, 4, 5]))

# Vector 1D
v = torch.tensor([1, 2, 3, 4, 5])
print("Vector 1D: ", v)
print("Form: ", v.shape)

# Matrix 2D
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6,],
                  [7, 8, 9]])
print("Matrix 2d:\n", m)
print("Form: ", m.shape)

# Tensor 3D (as a little picture)
img = torch.tensor([[[1, 2], [3, 4],
                     [5, 6], [7, 8]]])
print("Tensor 3D:\n", img)
print("Form: ", img.shape)

# Tensor with null
zeros = torch.zeros(3, 4)
print("Zeros:\n", zeros)

#Tensor with 1
ones = torch.ones(2, 5)
print("Ones:\n", ones)

#Tensoк with random
random = torch.randn(3, 3)
print("Random1:\n", random)

#Operation with tensor
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
c = a + b
print(c)

x = torch.tensor([2,4,6])
y = x * 3
print(y)

A = torch.tensor([[1,2,3],
                 [4,5,6]])
B = torch.tensor([[7,8],
                 [9,10],
                 [11,12]])

C = A @ B
print(C)

#Change the form of tensor
x = torch.arange(12)
print("Original: ", x)
print("Matrix 3x4: ", x.reshape(3,4))
print("Matrix 2x6: ", x.reshape(2,6))
