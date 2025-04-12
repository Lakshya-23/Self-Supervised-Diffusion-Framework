import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

test_dir = '/content/fer2013/test' 
print("Path to test dataset:", test_dir)

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((224, 224)),                
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])          
])

print("Loading test dataset...")
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

model.load_state_dict(torch.load('/content/fer2013_resnet18_best.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  

def evaluate_model():
    model.eval()  
    running_corrects = 0
    total = 0
    with torch.no_grad():  
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    accuracy = running_corrects.double() / total
    print(f"Test Accuracy: {accuracy:.4f}")


evaluate_model()
