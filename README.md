# Plant Classification
Name: Agosh Saini
Website: agoshsaini.com

# Objective
The objective of this project is to train a model that can be used for classification of plants from images. This project was used because I am really into foraging and this would be a good tool to verify plants before consumption.

Lets go into the breakdown of the code.

```
import torch
import torchvision
```

torch and torchvission are common libraries used for image classification. In these lines, we import the required modules.

```
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
```

The end user does not know the orientation and image size. The code block above creates a function that can be applied to all the photos to randomly transform the images. 

```
train_data = torchvision.datasets.ImageFolder(path + '/train', transform=train_data_transform)
test_data = torchvision.datasets.ImageFolder(path + '/test', transform=test_data_transform)
```

The training data is supervised and can be found under path/train. Same applies to the test data as well which is not labeled. These lines just allow the module to find the required data.

The 

# Citations
source for data: https://www.kaggle.com/datasets/msheriey/104-flowers-garden-of-eden?resource=download

note - OpenAI chatbot used for understanding and learning during this project
