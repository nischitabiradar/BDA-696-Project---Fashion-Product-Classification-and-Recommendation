
# Import necessary libraries
#Importing modules
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import time
import copy
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path to CSV file
csv_file_path = '/content/myntradataset/styles.csv'

try:
    # Attempt to read the CSV file with initial row count
    initial_row_count = sum(1 for _ in open(csv_file_path)) - 1  # excluding header
    df = pd.read_csv(csv_file_path, on_bad_lines='skip')

    # Counting the number of rows read
    final_row_count = df.shape[0]
    skipped_rows = initial_row_count - final_row_count
    print(f"Number of rows skipped due to errors: {skipped_rows}")

    # Efficiently creating the 'image' column
    df['image'] = df['id'].astype(str) + ".jpg"
    df = df.reset_index(drop=True)

except Exception as e:
    print(f"An error occurred: {e}")

df.head(5)

class_counts = df['articleType'].value_counts()

max_class = class_counts.idxmax()
max_count = class_counts.max()

min_class = class_counts.idxmin()
min_count = class_counts.min()


print(f"Class with the most images: {max_class}, Count: {max_count}")
print(f"Class with the least images: {min_class}, Count: {min_count}")

N_Pictures = 275
# Calculate the number of images per class
images_per_class = df.articleType.value_counts()

# Filter classes with more than N_Pictures images
sufficient_images = images_per_class[images_per_class > N_Pictures]

# Number of classes with sufficient images to train on
N_Classes = len(sufficient_images)

# Display the count of classes and the classes with their image counts
print(f"Number of classes with more than {N_Pictures} images: {N_Classes}")
print("Classes and their image counts:")
print(sufficient_images)

#Saving item types(labels) with their counts
items_count = sufficient_images.values
items_label = sufficient_images.index.tolist()

#Creating new dataframes for training/validation
df_train = pd.DataFrame(columns=['articleType','image'])
df_val   = pd.DataFrame(columns=['articleType','image'])


for ii in range(0,N_Classes):

    #print(items_label[ii])

    temp = df[df.articleType==items_label[ii]].sample(N_Pictures)

    df_train = pd.concat([df_train, temp[ :int(N_Pictures*0.6) ][['articleType','image']] ]            , sort=False)
    df_val   = pd.concat([df_val,   temp[  int(N_Pictures*0.6): N_Pictures ][['articleType','image']] ], sort=False)

df_train.reset_index(drop=True)
df_val.reset_index(drop=True)

import os
import logging
from PIL import Image

# Set the path to the dataset
DATASET_PATH = '/content/myntradataset'  # Adjust this path to where your dataset is located

# Function to save images in organized directories
def save_images(data, dataset_path, target_root='data'):
    # Create the necessary directories for the dataset if they do not exist
    for phase in data:
        os.makedirs(os.path.join(target_root, phase), exist_ok=True)

    # Process and save images
    for phase in data:
        logging.info(f"Processing {phase} data")
        # Make sure to select only the 'label' and 'filename' columns
        for label, filename in data[phase][['articleType', 'image']].values:
            src_file_path = os.path.join(dataset_path, 'images', filename)
            target_dir_path = os.path.join(target_root, phase, label)
            target_file_path = os.path.join(target_dir_path, filename)

            # Ensure target directory exists
            os.makedirs(target_dir_path, exist_ok=True)

            try:
                with Image.open(src_file_path) as img:
                    img.save(target_file_path)
            except FileNotFoundError:
                logging.warning(f"File not found: {src_file_path}")
            except Exception as e:
                logging.error(f"Error saving file {target_file_path}: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Assuming df_train and df_val are your DataFrames with 'articleType' and 'image' columns
data = {'train': df_train, 'val': df_val}

# Call the function
save_images(data, DATASET_PATH)

# Commented out IPython magic to ensure Python compatibility.
#Inspect if all the folders have been created
# %ls data/train/

data_transforms = {

    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
}

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features

#Changing the number of outputs in the last layer to the number of different item types
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=11)

# Commented out IPython magic to ensure Python compatibility.
#Saving our model's weights:

# %mkdir model
torch.save(model_ft.state_dict(), 'model/model_fine_tuned.pt')
# %ls

#Download the model weights and save them locally
from IPython.display import FileLink
FileLink(r'model/model_fine_tuned.pt')

visualize_model(model_ft)

def classify_image_with_input(model, device, data_transforms, class_names):
    # Prompt for image URL
    url = input("Please enter the URL of the image: ")

    # Download the image
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    del response

    # Transform the image
    img_t = data_transforms['val'](img).unsqueeze(0)
    img_t = img_t.to(device)

    # Predict the class
    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[int(preds.cpu().numpy())]

    return predicted_class

# Example usage:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming data_transforms and class_names are defined and model_ft is loaded
predicted_class = classify_image_with_input(model_ft, device, data_transforms, class_names)
print("Predicted class:", predicted_class)
