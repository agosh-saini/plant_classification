'''
Name: Agosh Saini
Contact: Agosh.Saini@gmail.com
Website: agoshsaini.com

Note - The code breakdown can be found in jupiter notebook (.ipynb) file
'''

# --- Imports --- #
import torch
import torchvision

# --- path to dataset --- #
path = 'jpeg-512x512'

# --- making it work with gpu --- #
if torch.cuda.is_available():
    device_name = torch.device("cuda")
else:
    device_name = torch.device('cpu')

print("Using - " + str(device_name))

# --- randomly transform images --- #
train_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(180),
    torchvision.transforms.RandomResizedCrop(300),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                     std=[0.2, 0.2, 0.2])
    ])

test_data_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(180),
    torchvision.transforms.RandomResizedCrop(300),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                     std=[0.2, 0.2, 0.2])
    ])

# --- creating splits for training and testing --- #
train_data = torchvision.datasets.ImageFolder(path + '/train',
                                              transform=train_data_transform)
test_data = torchvision.datasets.ImageFolder(path + '/val',
                                             transform=test_data_transform)
# --- dataloader information --- #
batch_size = 16

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

# --- send the data and labels to gpu if available --- #
for inputs, labels in train_loader:
    inputs, labels = inputs.to(device_name), labels.to(device_name)
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device_name), labels.to(device_name)

# --- model parameters --- #
model = torchvision.models.resnet18(pretrained=True)
features = model.fc.in_features
model.fc = torch.nn.Linear(features, len(train_data.classes))

# --- loss function and optimizer --- #
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# --- send model to gpu if available --- #
model.to(device_name)

# --- training the last layer --- #
epochs = 5

for epoch in range(epochs):
    running_loss = 0

    for i, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = model(inputs.cuda())
        loss = loss_func(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.cuda().size(0)

    epoch_loss = running_loss / len(train_data)
    print('Epoch ' + str(epoch + 1) + ' of ' + str(epochs) + ' Loss: ' + str(epoch_loss))


# --- saving the model --- #
torch.save(model.state_dict(), 'flower_classifier.pth')

# --- testing the model --- #
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.cuda().size(0)
        correct += (predicted == labels.cuda()).sum().item()

accuracy = correct / total

print("Accuracy: " + str(accuracy))

# --- end of script --- #
