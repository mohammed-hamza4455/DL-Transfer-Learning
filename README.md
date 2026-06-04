# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
To classify chip images as defect or notdefect using a pretrained VGG19 model with transfer learning.


## DESIGN STEPS
Step 1:
Load the chip image dataset from Google Drive and organize it into train and test folders.

Step 2:
Preprocess the images by resizing, converting to tensor, and normalizing them for VGG19.

Step 3:
Load the pretrained VGG19 model and modify its final layer for binary classification.

Step 4:
Freeze selected layers and define the loss function and optimizer.

Step 5:
Train the model using the training dataset and validate it using the test dataset.

Step 6:
Evaluate and predict results using accuracy, confusion matrix, classification report, and sample image prediction.






## PROGRAM

### Name:MOHAMMED HAMZA M

### Register Number:212224230167

```python
# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# Step 1: Load and Preprocess Data

# Define image transformations
# Resize image to 224x224 because VGG19 requires this input size
# Normalize using ImageNet mean and standard deviation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Unzip dataset from Google Drive
# Change this path if your dataset is in another folder
!unzip -qq "/content/drive/MyDrive/DEEP LEARNING/chip_data.zip" -d data

# Dataset path
dataset_path = "./data/dataset/"

# Load training and testing datasets
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

# Print class details
print("Classes:", train_dataset.classes)
print("Class to Index:", train_dataset.class_to_idx)


# Function to display sample images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

    # Mean and standard deviation used for normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(num_images):
        image, label = dataset[i]

        # Convert tensor from C,H,W to H,W,C
        image = image.cpu().numpy().transpose((1, 2, 0))

        # Unnormalize image for correct display
        image = image * std + mean
        image = np.clip(image, 0, 1)

        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")

    plt.show()


# Show sample images
show_sample_images(train_dataset)

# Print dataset information
print(f"Total number of training samples: {len(train_dataset)}")

first_image, label = train_dataset[0]
print(f"Shape of the first training image: {first_image.shape}")

print(f"Total number of test samples: {len(test_dataset)}")

first_image1, label = test_dataset[0]
print(f"Shape of the first testing image: {first_image1.shape}")


# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Step 2: Load Pretrained Model and Modify for Transfer Learning

from torchvision.models import VGG19_Weights

# Load pretrained VGG19 model
model = models.vgg19(weights=VGG19_Weights.DEFAULT)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print model summary before modification
from torchsummary import summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer for binary classification
# Output is 1 because BCEWithLogitsLoss is used
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)

# Freeze feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False

# Move modified model to device
model = model.to(device)

# Print model summary after modification
summary(model, input_size=(3, 224, 224))

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Step 3: Train the Model

def train_model(model, train_loader, test_loader, num_epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels.unsqueeze(1).float())

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        # Average training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()

        # Average validation loss
        val_loss = val_loss / len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Plot training and validation loss
    print("Name:M MOHAMMED HAMZA ")
    print("Register Number: 212224230167")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


# Train the model
train_model(model, train_loader, test_loader, num_epochs=10)


# Step 4: Test the Model

def test_model(model, test_loader):
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get model output
            outputs = model(images)

            # Apply sigmoid to convert output into probability
            probs = torch.sigmoid(outputs)

            # Convert probability to class prediction
            preds = (probs > 0.5).int().squeeze()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("Name:M MOHAMMED HAMZA ")
    print("Register Number: 212224230167")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Name: MOHAMMED HAMZA M")
    print("Register Number: 212224230167")

    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=train_dataset.classes,
            zero_division=0
        )
    )


# Evaluate the model
test_model(model, test_loader)


# Step 5: Predict on a Single Image and Display It

def predict_image(model, image_index, dataset):
    model.eval()

    # Get image and actual label
    image, label = dataset[image_index]

    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Apply sigmoid to get probability
        prob = torch.sigmoid(output)

        # Threshold at 0.5
        predicted = (prob > 0.5).int().squeeze().item()

    class_names = dataset.classes

    # Convert image tensor from C,H,W to H,W,C
    image_to_display = image.cpu().numpy().transpose((1, 2, 0))

    # Unnormalize image for correct display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_to_display = image_to_display * std + mean
    image_to_display = np.clip(image_to_display, 0, 1)

    # Display image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f"Actual: {class_names[label]}\nPredicted: {class_names[predicted]}")
    plt.axis("off")
    plt.show()

    print(f"Actual: {class_names[label]}, Predicted: {class_names[predicted]}")


# Example Predictions
print("New Sample Data Prediction")

predict_image(model, image_index=55, dataset=test_dataset)
predict_image(model, image_index=25, dataset=test_dataset)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="2272" height="1888" alt="11" src="https://github.com/user-attachments/assets/c6f1c8c3-8afb-489e-8c89-725cbef664c2" />



## Confusion Matrix
<img width="1167" height="1086" alt="image" src="https://github.com/user-attachments/assets/0cffad42-b1e8-49af-b6cd-3c2cdd221af0" />



### New Sample Data Prediction
<img width="520" height="535" alt="image" src="https://github.com/user-attachments/assets/e04f755c-2a72-425b-b4c1-91631f0b29a3" />
<img width="518" height="506" alt="image" src="https://github.com/user-attachments/assets/3c0b0674-e6ad-4674-b01b-61a9c4970e0b" />



## RESULT
Thus the program to developing a Neural Network Classification Model using Transfer Learning was executed successfully.
