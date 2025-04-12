import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torchvision
import time
import copy
from tqdm import tqdm  

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

data_dir = '/content/fer2013/train' 
print("Path to dataset files:", data_dir)

num_classes = 7
batch_size = 64
num_epochs = 25
learning_rate = 0.001
image_size = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

print("Loading dataset...")
try:
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    class_names = dataset.classes
    dataset_size = len(dataset)

    print(f"Dataset loaded successfully. Found {dataset_size} images in {len(class_names)} classes:")
    print(class_names)

    if num_classes != len(class_names):
        print(f"Warning: num_classes ({num_classes}) does not match detected folders ({len(class_names)}). Using detected number.")
        num_classes = len(class_names)
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

print("Initializing model...")
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
print("Model initialized.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model():
    print("Starting training...")
    start_time = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print(f'Epoch Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("  -> New best training accuracy!")

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Training Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    model_save_path = '/content/fer2013_resnet18_best.pth'  
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model weights saved to {model_save_path}")

    print("\n" + "="*40)
    print("NOTE: This code trains ResNet18 for FER. CARD diffusion model not included.")
    print("="*40)

train_model()
