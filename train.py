import argparse
import json
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 
parser = argparse.ArgumentParser(
    prog='TrainModel', 
    description='Trains an image classifier model.'
)
parser.add_argument('data_dir')
parser.add_argument('--save_dir', default='.')
parser.add_argument('--arch', default='vgg11')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--hidden_units', type=int, default=4096)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

# Extract arguments
data_dir = args.data_dir
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

# Set up transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_transforms = {
    'train': 
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    'test':
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
}

# Load the datasets with ImageFolder
image_datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': ImageFolder(valid_dir, transform=data_transforms['test']),
}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    key: DataLoader(dataset, batch_size=32, shuffle=(key=='train'))
    for key, dataset in image_datasets.items()
}

# Make the model
model_constructor = getattr(models, arch, None)
if model_constructor is None:
    raise ValueError(f'Invalid model architecture: {arch}')
model = model_constructor(pretrained=True)

# Turn off gradients for the model
for param in model.parameters():
    param.requires_grad = False

# Define the classifier
# num_features = 25088
classifier = nn.Sequential(
    # Infer the input size from the model
    nn.Linear(model.classifier[0].in_features, hidden_units),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(hidden_units, hidden_units),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(hidden_units, len(image_datasets['train'].classes)),
)

# Add the classifier to the model
model.classifier = classifier

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=lr)

if args.gpu and not torch.cuda.is_available():
    raise ValueError('GPU requested but not available')

device = torch.device("cuda" if args.gpu else "cpu")
model = model.to(device)

# Train the model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for images, labels in dataloaders['train']:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Get the features
        features = model.features(images)
        features = features.view(features.size(0), -1)

        # Run classifier
        outputs = model.classifier(features)

        # Backpropagation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    train_loss = running_loss / len(image_datasets['train'])
    train_accuracy = correct_predictions / len(image_datasets['train'])

    # Validation
    model.eval()
    val_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in dataloaders['valid']:
            images = images.to(device)
            labels = labels.to(device)

            features = model.features(images)
            features = features.view(features.size(0), -1)
            outputs = model.classifier(features)

            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

    val_loss = val_loss / len(image_datasets['valid'])
    val_accuracy = correct_predictions / len(image_datasets['valid'])
    emoji = '😖' if val_accuracy < 0.5 else \
            '😕' if val_accuracy < 0.75 else \
            '🙂' if val_accuracy < 0.9 else \
            '😊' if val_accuracy < 0.91 else \
            '😁' if val_accuracy < 0.92 else \
            '🤩' if val_accuracy < 0.93 else \
            '🤯' if val_accuracy < 0.94 else \
            '✨🧠🤯🧠✨'

    print(f"🌈 Epoch {epoch + 1} of {epochs} 💅 - "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f} "
          f"{emoji}")

model.eval()
val_loss = 0.0
correct_predictions = 0

with torch.no_grad():
    for images, labels in dataloaders['test']:
        images = images.to(device)
        labels = labels.to(device)

        features = model.features(images)
        features = features.view(features.size(0), -1)
        outputs = model.classifier(features)

        val_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

test_loss = val_loss / len(image_datasets['test'])
test_accuracy = correct_predictions / len(image_datasets['test'])

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint_filename = os.path.join(save_dir, 'trained_model_checkpoint.pt')
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'train_accuracy': train_accuracy,
    'class_to_idx': image_datasets['train'].class_to_idx,
    'arch': arch,
    'hidden_units': hidden_units,
    'learning_rate': lr,
}
torch.save(checkpoint, checkpoint_filename)
