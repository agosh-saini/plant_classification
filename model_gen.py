'''
Name: Agosh Saini
Contact: Agosh.Saini@gmail.com
Website: agoshsaini.com
'''


# --- Imports --- #
import torch
import torchvision

# --- path to dataset --- #
path = 'jpeg-192x192'

# --- randomly transform images --- #
train_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(180),
    torchvision.transforms.RandomResizedCrop(150),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

test_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(180),
    torchvision.transforms.RandomResizedCrop(150),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

# --- creating splits for training and testing --- #
train_data = torchvision.datasets.ImageFolder(path + '/train', transform=train_data_transform)
test_data = torchvision.datasets.ImageFolder(path + '/test', transform=test_data_transform)

# --- dataloader information --- #
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# --- model parameters --- #
model = torchvision.models.resnet18()
features = model.fc.in_features
model.fc = torch.nn.Linear(features, len(train_data.classes))

# --- loss function and optimizer --- #
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# --- training the last layer --- #
epochs = 10

for epoch in range(epochs):
    running_loss = 0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_data)
    print('Epoch ' + str(epoch + 1) + ' of ' + str(epochs) + ' Loss: ' + str(epoch_loss))

# --- testing the model --- #
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

print("Accuracy: " + str(accuracy))

# --- saving the model --- #
torch.save(model.state_dict(), 'flower_classifier.pth')

