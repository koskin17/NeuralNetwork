import torch
import torch.nn as nn


#Make linear layer
layer = nn.Linear(4, 3)

#Input: 1 example, 4 signs
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

#Direct input
y = layer(x)
print(y)