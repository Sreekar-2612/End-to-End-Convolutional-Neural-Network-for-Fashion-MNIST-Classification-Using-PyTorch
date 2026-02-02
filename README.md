```markdown
# End-to-End Convolutional Neural Network for Fashion-MNIST Classification Using PyTorch

## Project Overview
This project demonstrates an end-to-end implementation of a Convolutional Neural Network (CNN) using PyTorch for classifying images from the Fashion-MNIST dataset. The Fashion-MNIST dataset consists of 60,000 training images and 10,000 test images of Zalando's fashion articles, each a 28x28 grayscale image, associated with a label from 10 classes.

## Table of Contents
1.  [Setup and Dependencies](#setup-and-dependencies)
2.  [Data Preprocessing](#data-preprocessing)
3.  [CNN Model Architecture](#cnn-model-architecture)
4.  [Model Initialization and Training](#model-initialization-and-training)
5.  [Model Evaluation (Accuracy & Score)](#model-evaluation-accuracy--score)
6.  [Saving and Loading the Model](#saving-and-loading-the-model)
7.  [Single Image Prediction](#single-image-prediction)
8.  [Visualization](#visualization)

## 1. Setup and Dependencies
This project uses PyTorch for building and training the CNN. The necessary libraries are imported at the beginning of the notebook.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

The code also checks for GPU availability and sets the device accordingly:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
```

## 2. Data Preprocessing

### Transformations
Images are transformed by converting them to tensors and normalizing their pixel values.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
```

### Dataset Loading
The Fashion-MNIST dataset is loaded for both training and testing. If not already present, the dataset will be downloaded.

```python
train_dataset = datasets.FashionMNIST(
    root="./data",
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.FashionMNIST(
    root = "./data",
    train  = False,
    download = True,
    transform = transform
)
```

### DataLoader
DataLoaders are created to efficiently batch and shuffle the training and testing data.

```python
train_loader = DataLoader(train_dataset,batch_size = 64,shuffle=True)
test_loader  = DataLoader(test_dataset,batch_size = 64,shuffle=True)
```

## 3. CNN Model Architecture
**Model Description:** The `FashionCNN` class defines a simple Convolutional Neural Network. This model is designed for image classification tasks and consists of two convolutional layers, each followed by a ReLU activation and max-pooling, to extract features from the input images. These layers are then followed by two fully connected (dense) layers to perform the final classification.

```python
class FashionCNN(nn.Module):
  def __init__(self):
    super(FashionCNN,self).__init__()

    self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding = 1)
    self.conv2 = nn.Conv2d(16,32,kernel_size = 3,padding = 1)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(32*7*7,128)
    self.fc2 = nn.Linear(128,10)

  def forward(self,x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = x.view(x.size(0),-1)
    x = torch.relu(self.fc1(x))
    return self.fc2(x)
```

## 4. Model Initialization and Training

**Model Setup:** The `FashionCNN` model is initialized and moved to the appropriate device (CPU or GPU). The loss function chosen is `CrossEntropyLoss`, which is suitable for multi-class classification problems. The `Adam` optimizer is used to update the model's parameters during training, with a learning rate of 0.001.

```python
model  = FashionCNN().to(device)

loss_fn = nn.CrossEntropyLoss()   #for multiclass classification
optimizer= optim.Adam(model.parameters(),lr = 0.001)
```

The training loop iterates for a specified number of epochs. In each epoch, the model processes batches of images, calculates the loss, performs backpropagation, and updates its weights.

```python
epochs = 10
for epoch in range(epochs):
  model.train()
  cur_loss = 0.0

  for images,labels in train_loader:
    images,labels = images.to(device),labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs,labels)
    loss.backward()
    optimizer.step()

    cur_loss += loss.item()

  print(f"Epoch {epoch+1}/{epochs}, Loss: {cur_loss/len(train_loader)}")
```

## 5. Model Evaluation (Accuracy & Score)
After training, the model's performance is evaluated on the unseen test dataset. The primary metric used here is **accuracy**, which represents the percentage of correctly classified images.

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for images,labels in test_loader:
    images,labels  = images.to(device),labels.to(device)
    outputs = model(images)
    _,predicted = torch.max(outputs,1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()
accuracy = 100 *correct / total
print(f"Test accuracy: {accuracy:2f}%")
```

**Achieved Accuracy (Score):** The model achieved a test accuracy of approximately **91.61%**. This score indicates that the model is performing well on the Fashion-MNIST classification task.

## 6. Saving and Loading the Model
The trained model's state dictionary can be saved to a file (`Fashion_cnn.pt`) and loaded later for inference or further training, avoiding the need to retrain the model from scratch.

### Saving
```python
torch.save(model.state_dict(),'Fashion_cnn.pt')
```

### Loading
```python
model = FashionCNN().to(device)
model.load_state_dict(torch.load("Fashion_cnn.pt"))
model.eval()
```

## 7. Single Image Prediction
This section demonstrates how to use the trained model to predict the class of a single image from the test set and compares it with the actual label.

```python
classes = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

image, label = test_dataset[0]
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    pred = output.argmax(dim=1)

print("Predicted:", classes[pred.item()])
print("Actual   :", classes[label])
```

## 8. Visualization
The predicted image can be visualized along with its predicted label for a clear understanding of the model's output.

```python
plt.imshow(image.cpu().squeeze(), cmap="gray")
plt.title(f"Predicted: {classes[pred.item()]}")
plt.axis("off")
plt.show()
```
```
