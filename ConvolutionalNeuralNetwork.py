#Сверточная нейросеть

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. Заггружаем данные (тоже самое, что в предыдущем примере нейросети)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 1000, shuffle = False)


# 2. Создаём сверточную нейросеть
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Первый блок: свёртка + пулинг
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1) #1 канал - 32 фильтра
        self.pool = nn.MaxPool2d(2, 2)  #уменьшаем размер в 2 раза
        
        # Второй блок: свёртка + пулинг
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)    # 32 канала - 64 фильтра
        
        # Полносвязная часть
        self.fc1 = nn.Linear(64 * 7 * 7, 128)   #64 * 7 * 7 = 3136
        self.fc2 = nn.Linear(128, 10)   #128 - 10 классов (цифры от 0 до 9)
        
    def forward(self, x):
        # Первый блок
        x = torch.relu(self.conv1(x))   #28*28*32
        x = self.pool(x)             #14*14*32
        
        # Второй блок
        x = torch.relu(self.conv2(x))   #14*14*64
        x = self.pool(x)             #7*7*64
        
        # Выпрямляем
        x = x.view(-1, 64 * 7 * 7)  #3136 чисел
        
        # Полносвязная часть
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
model = CNN()

# 3. Обучение (тоже самое, что и в предыдущем примере нейросети)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
    # Проверка
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f'Эпоха {epoch+1}: точность {100 * correct / total:.2f}%')