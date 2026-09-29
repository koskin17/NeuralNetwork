import torch


m = torch.randn(4, 4)
print(m)

v = m.reshape(16)
print(v)

result = v + torch.ones(16)
print(result)

