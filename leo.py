import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
import os

max_duration = 360000

# Function to extract breed name from label
def extract_breed_name(label):
    return label.split('-')[-1]

# Path to the dataset
dataset_path = 'C:\\Users\\codya\\Downloads\\testDataML\\dog_breeds'

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a function to display images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Load the dataset from folders
dataset = ImageFolder(root=dataset_path, transform=transform)

# Create a mapping from breed names to class indices based on the dataset's classes
classes = dataset.classes
breed_to_class = {breed: idx for idx, breed in enumerate(classes)}

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
batch_size = 8
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define your neural network architecture here
class Net(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # Adjusted for dynamic class number

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Load previous model weights if available
model_path = 'leotrain.pth'
try:
    net.load_state_dict(torch.load(model_path))
    print("Previous model weights loaded successfully!")
except FileNotFoundError:
    print("No previous model weights found. Starting from scratch.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

# Train the neural network
for epoch in range(80):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        if time.time() - start_time > max_duration:
            print("Training stopped due to reaching the maximum duration.")
            break
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        if time.time() - start_time > max_duration:
            break

print('Finished Training')

# Testing the network on the test data
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {len(testloader.dataset)} test images: {100 * correct / total} %')

# Adjust the output layer dimension and test some images (example from the first batch)
dataiter = iter(testloader)
images, labels = next(dataiter)
outputs = net(images)
_, predicted = torch.max(outputs, 1)
predicted_breeds = [classes[idx] for idx in predicted]  # Corrected generation of predicted_breeds

# Display images and predictions
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:', ' '.join(f'{classes[labels[j]]}' for j in range(len(labels))))
print('Predicted:', ' '.join(f'{predicted_breeds[j]}' for j in range(len(labels))))

# Save the model weights for future use
torch.save(net.state_dict(), 'leotrain.pth')