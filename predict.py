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
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 
parser = argparse.ArgumentParser(
    prog='PredictModel', 
    description='Given a model and an image, predict the class or classes of the image using a trained deep learning model.'
)
parser.add_argument('image_path')
parser.add_argument('checkpoint_path')
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--category_names')
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

# Extract arguments
image_path = args.image_path
checkpoint_path = args.checkpoint_path
top_k = args.top_k
category_names_path = args.category_names
gpu = args.gpu
if args.gpu and not torch.cuda.is_available():
    raise ValueError('GPU requested but not available')
device = torch.device("cuda" if args.gpu else "cpu")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Load checkpoint
loaded_checkpoint = torch.load(checkpoint_path)
hidden_units = loaded_checkpoint['hidden_units']

# Load model
model = getattr(models, loaded_checkpoint['arch'])(pretrained=True)
model.class_to_idx = loaded_checkpoint['class_to_idx']
for param in model.parameters():
    param.requires_grad = False

# Load classifier
classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, hidden_units),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(hidden_units, hidden_units),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(hidden_units, len(model.class_to_idx)),
)

# Add the classifier to the model
model.classifier = classifier
model.load_state_dict(loaded_checkpoint['model_state_dict'])
model = model.to(device)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    shortest_side = 256
    width, height = image.size
    ratio = width / height

    if width < height:
        new_width = shortest_side
        new_height = int(new_width / ratio)
    else:
        new_height = shortest_side
        new_width = int(new_height * ratio)

    image = image.resize((new_width, new_height), Image.BICUBIC)
    
    # Chop out the center 224x224 square
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2

    image = image.crop((left, top, right, bottom))
    
    img_array = np.array(image)
    img_array = img_array / 255.0
    
    means = np.array(mean)
    stds = np.array(std)
    img_array = (img_array - means) / stds
    
    # Transpose the image array to match PyTorch tensor format (C, H, W)
    img_array = img_array.transpose((2, 0, 1))

    return img_array

def predict(image_path, model, topk):
    img_raw = Image.open(image_path)
    processed_img = process_image(img_raw)
    
    # Get tensor into the correct format and load on to the GPU
    img_tensor=torch.from_numpy(processed_img).float().to(device)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
    
    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    topk_probabilities, topk_indices = torch.topk(probabilities, k=topk)

    # Convert back
    topk_probabilities = topk_probabilities.squeeze()
    topk_probabilities = [topk_probabilities.item()] if topk_probabilities.numel() == 1 else topk_probabilities.tolist()
    topk_indices = topk_indices.squeeze()
    topk_indices = [topk_indices.item()] if topk_indices.numel() == 1 else topk_indices.tolist()
    
    label_map = {v: k for k, v in model.class_to_idx.items()}
    return topk_probabilities, [label_map[idx] for idx in topk_indices]

# Load category names
if category_names_path:
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = {k: k for k, v in model.class_to_idx.items()}

# Predict the class (or classes) of an image using a trained deep learning model.
topk_probabilities, topk_indices = predict(image_path, model, top_k)
topk_labels = [cat_to_name[idx] for idx in topk_indices]
max_label_len = max([len(label) for label in topk_labels])

print(f"The image {image_path} is most likely a {topk_labels[0]} with a probability of {topk_probabilities[0] * 100:.2f}%")
if top_k > 1:
    print(f"Top {top_k} predictions for {image_path}:")
    for i in range(top_k):
        num_blocks = round(topk_probabilities[i] * 50)
        print(
            f"{topk_labels[i].rjust(max_label_len)}: "
            f"{'█' * num_blocks if num_blocks > 0 else '░'} "
            f"{topk_probabilities[i] * 100:.2f}%"
        )