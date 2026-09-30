import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#Upload the MNIST dataset - examples of handwritten digits
transform = transforms.Compose([
    transforms.ToTensor(),  # Превращает картинку в тензор (список чисел)
    transforms.Normalize((0.1307,), (0.3081,))  #Делает числа удобными для обучения (центрирует около ноля)
])

train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)

#Making neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)    #784 to 256. Это первый слой. 784 входа, 128 выходов. 784 - это 28х28 пикселей, 256 - это скрытый слой (мы можем менять это число)
        self.fc2 = nn.Linear(256, 128)      #256 to 128
        self.fc3 = nn.Linear(128, 10)       #128 to 10
        
    def forward(self, x):
        x = x.view(-1, 28*28)   #Convert image to vector. Это второй слой. 128 входов, 10 выходов. 10 - это цифры от 0 до 9.
        x = torch.relu(self.fc1(x)) #ReLU - это функциб активации, которая делает нейросеть нелинейной, чтобы она могла распозновать сложные узоры. При этом ReLU(x) = max(0, x), т.е. отрицательные числа превращаются в 0, а положительные остаются как есть.
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
model = SimpleNN()

#Configuring education proccess
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#Training the model
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()   #Обнуление старых гралиентов, чтобы не смешивать их.
        output = model(images)  #Это прямой подход - нейросеть выдаёт предсказание.
        loss = criterion(output, labels)    #Вычисляем насколько мы ошиблись. 
        loss.backward() #Вычисляем градиенты, т.е. куда крутить веса.
        optimizer.step()    #Крутим все веса в правильную сторону.
        
    #Check the accuracy after each epoch
    # Точность = Правильных ответов / Всех ответов * 100%
    # Если сеть правильно распознаёт 9700 из 10000 числе, то точно равна 9700 / 10000 * 100% = 97%.
    # Для MNIST хорошая точноть >95%.
    # Отличная: >98%.
    # Человек: ~99.7% - редко ошибается в рукописных цифрах
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {100 * correct / total}%')