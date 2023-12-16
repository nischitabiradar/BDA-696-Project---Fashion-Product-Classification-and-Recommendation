
! pip install flask flask_ngrok

from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

# Step 1: Writing the requirements to a file
with open('requirements.txt', 'w') as file:
    file.write('swifter\n')
    file.write('torchvision\n')
    file.write('opencv-python\n')

# Step 2: Installing the libraries using the requirements.txt file
!pip install -r requirements.txt

import os
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from pandas.core.common import flatten
import torchvision.transforms as transforms
from torch.autograd import Variable
import warnings
import torchvision.models as models

plt.style.use('ggplot')
warnings.filterwarnings("ignore")

from pathlib import Path

# Defining the dataset path
DATASET_PATH = Path('/content/myntradataset')

# Listing the contents of the directory
print([item.name for item in DATASET_PATH.iterdir()])

from pathlib import Path
import pandas as pd

# Defining the dataset path
DATASET_PATH = Path('/content/myntradataset')

# Check if the styles.csv file exists in the directory
csv_file = DATASET_PATH / 'styles.csv'
if csv_file.is_file():
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file, nrows=5000)
    except pd.errors.ParserError as e:
        print(f"Error while reading the CSV file: {e}")
else:
    print(f"The file {csv_file} does not exist.")

df.head(5)

import pandas as pd

# Assuming df is your DataFrame and DATASET_PATH is defined
# Add a new column 'image' to the DataFrame with the constructed image filenames
df['image'] = df['id'].astype(str) + '.jpg'

# Display the first few rows of the DataFrame
df.head(5)

import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Assuming DATASET_PATH is defined as before
DATASET_PATH = Path('/content/myntradataset/')

def get_image_path(filename):
    """Generate the full path for an image file."""
    return DATASET_PATH / 'images' / filename

def load_image(filename):
    """Load an image from a given filename."""
    path = get_image_path(filename)
    return cv2.imread(str(path))

def show_images_grid(images_dict, rows=2, cols=3, grid_size=(8, 8)):
    """Display a grid of images."""
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=grid_size)
    for index, (title, img) in enumerate(images_dict.items()):
        if img is not None:
            axes.ravel()[index].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes.ravel()[index].set_title(title)
        axes.ravel()[index].axis('off')
    plt.tight_layout()

# Generation of a dictionary of (title, image)
images_info = {f'Image {i}': load_image(row.image) for i, row in df.sample(3).iterrows()}

# Plot of the images in a figure, with 2 rows and 3 columns
show_images_grid(images_info, rows=1, cols=3)

# Desired dimensions for images
image_width, image_height = 224, 224

# Loading the pre-trained ResNet model
pretrained_resnet = models.resnet18(pretrained=True)

# Selecting the desired layer from the model
selected_layer = pretrained_resnet._modules.get('avgpool')

#evaluate the model
pretrained_resnet.eval()

from torchvision import transforms

# scaling, normalizing, and tensor conversion
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Scaling the data
    transforms.ToTensor(),          # Converting to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalizing
                         std=[0.229, 0.224, 0.225])
])

from PIL import Image
from torch.autograd import Variable

# Global variables for transformations and missing images list
global image_transforms
global missing_img

# Function to get embeddings
def modified_vector_extraction(pretrained_resnet, image_id):
    try:
        # Load the image with Pillow library
        img = Image.open(get_image_path(image_id)).convert('RGB')

        # Apply the predefined transformations
        t_img = Variable(image_transforms(img).unsqueeze(0))

        # Create a vector of zeros that will hold our feature vector
        # The 'avgpool' layer has an output size of 512
        embeddings = torch.zeros(512)

        # Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            embeddings.copy_(o.data.reshape(o.data.size(1)))

        # Attach that function to our selected layer
        hlayer = pretrained_resnet.avgpool.register_forward_hook(copy_data)

        # Run the model on our transformed image
        pretrained_resnet(t_img)

        # Detach our copy function from the layer
        hlayer.remove()

        return embeddings

    # If file not found
    except FileNotFoundError:
        print(f"File not found for image ID: {image_id}")
        missing_img.append(image_id)
        return None

from torchvision import models
from PIL import Image
from torch.autograd import Variable

# Function to modify the ResNet model
def modify_resnet_model(model):
    # Remove the last fully connected layer (fc) to get features from the avgpool layer
    model = nn.Sequential(*list(model.children())[:-1])
    return model

# Function to extract features
def extract_features_modified_model(modified_model, transformations, image_id):
    try:
        # Load the image with Pillow library
        img = Image.open(get_image_path(image_id)).convert('RGB')

        # Apply transformations
        transformed_img = Variable(transformations(img).unsqueeze(0))

        # Forward pass through the modified model
        modified_model.eval()
        with torch.no_grad():
            # Get the output directly from the modified model
            features = modified_model(transformed_img)

            # Flatten the features to a 1D vector
            features_vector = features.view(features.size(0), -1)

        return features_vector.numpy().flatten()

    except FileNotFoundError:
        print(f"File not found for image ID: {image_id}")
        return None

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import swifter
# 
# # Applying embeddings on subset of this huge dataset
# df_embeddings = df  # We can apply on the entire df, like: df_embeddings = df
# 
# # Define the modified ResNet model
# modified_resnet = modify_resnet_model(models.resnet18(pretrained=True))
# 
# # Looping through images to get embeddings
# map_embeddings = df_embeddings['image'].swifter.apply(lambda img: extract_features_modified_model(modified_resnet, image_transforms, img))
# 
# # Convert to a DataFrame
# df_embs = pd.DataFrame(map_embeddings.tolist())
# 
# # Print the shape and the first few rows of the DataFrame
# print(df_embs.shape)
# df_embs.head()
#

from pathlib import Path

# Specify the directory path
directory_path = Path('/content/working')

# Create the directory
directory_path.mkdir(parents=True, exist_ok=True)

# List the contents of the current working directory to verify
print(list(Path('/content').iterdir()))

# Specify the directory path
directory_path = Path('/content/working')

# Create the directory if it doesn't exist
directory_path.mkdir(parents=True, exist_ok=True)

# Specify the CSV file path
csv_file_path = directory_path / 'df_embs.csv'

# Export the embeddings to the CSV file
df_embs.to_csv(csv_file_path, index=False)

# Specify the CSV file path
csv_file_path = directory_path / 'df_embs.csv'

# Check if the CSV file exists
if csv_file_path.exists():
    # Read the embeddings from the CSV file
    df_embs = pd.read_csv(csv_file_path)
    df_embs.dropna(inplace=True)
    df_embs.reset_index(drop=True, inplace=True)
else:
    print(f"CSV file {csv_file_path} does not exist.")

import pickle


# Specify the directory path
directory_path = Path('/content/working')

# Create the directory if it doesn't exist
directory_path.mkdir(parents=True, exist_ok=True)

# Specify the pickle file path
pkl_file_path = directory_path / 'df_embs.pkl'

# Export the DataFrame as a pickle file
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(df_embs, pkl_file, protocol=4)

# Specify the pickle file path
pkl_file_path = directory_path / 'df_embs.pkl'

# Check if the pickle file exists
if pkl_file_path.exists():
    # Load the DataFrame from the pickle file
    with open(pkl_file_path, 'rb') as pkl_file:
        df_embs = pickle.load(pkl_file)
else:
    print(f"Pickle file {pkl_file_path} does not exist.")

# Calculating similarity between images ( using embedding values )
cosine_sim = cosine_similarity(df_embs)

# Previewing first 4 rows and 4 columns similarity just to check the structure of cosine_sim
cosine_sim[:4, :4]

from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix for all embeddings
cosine_sim = cosine_similarity(df_embs)

def recommend_images(image_index, df, top_n=6):
    # Get the cosine similarities of the specified image index with all other images
    sim_scores = list(enumerate(cosine_sim[image_index]))

    # Sort the similarities in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Extract the top n similar image indices (excluding the specified image itself)
    top_similar_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Extract the similarity scores
    similarity_scores = [i[1] for i in sim_scores[1:top_n+1]]

    # Map the similar indices to the original DataFrame index
    similar_image_indices = df.index[top_similar_indices]

    return similar_image_indices, similarity_scores

# Sample usage
recommend_images(3820, df, top_n=5)

def Rec_viz_image(input_image_id):
    # Getting recommendations
    similar_indices, similarity_scores = recommend_images(input_image_id, df, top_n=6)

    # Printing the similarity scores
    print(similarity_scores)

    # Plotting the image of the item requested by the user
    input_image = load_image(df.iloc[input_image_id].image)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

    # Generating a dictionary of {index, image} for similar images
    figures = {'im' + str(i): load_image(df.iloc[i].image) for i in similar_indices}

    # Plotting the similar images in a figure, with 2 rows and 3 columns
    show_images_grid(figures, 2, 3)

# Sample usage
Rec_viz_image(3820)

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip -o ngrok-stable-linux-amd64.zip

! /content/ngrok authtoken 2ZTic5usQOzcSIWeBwWH3eeh0dy_6GDzyLRgpaQJRn2ybpFsi

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

import requests
import io

def recm_user_input(url):
    try:
        # Download the image from the URL
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')

        # Apply the image transformations
        t_img = Variable(image_transforms(img).unsqueeze(0))

        # Rest of the code
        embeddings = torch.zeros(512)

        def select_d(m, i, o):
            embeddings.copy_(o.data.reshape(o.data.size(1)))

        hlayer = selected_layer.register_forward_hook(select_d)
        pretrained_resnet(t_img)
        hlayer.remove()
        emb = embeddings

        cs = cosine_similarity(emb.unsqueeze(0), df_embs)
        cs_list = list(flatten(cs))
        cs_df = pd.DataFrame(cs_list, columns=['Score'])
        cs_df = cs_df.sort_values(by=['Score'], ascending=False)

        print(cs_df['Score'][:5])

        top5 = cs_df[:5].index
        top5 = list(flatten(top5))
        images_list = []

        for i in top5:
            image_id = df[df.index == i]['image']
            print(image_id)
            images_list.append(image_id)

        images_list = list(flatten(images_list))

        figures = {'im' + str(i): Image.open('/content/myntradataset/images/' + i) for i in images_list}
        fig, axes = plt.subplots(1, 5, figsize=(8, 8))

        for index, name in enumerate(figures):
            axes.ravel()[index].imshow(figures[name])
            axes.ravel()[index].set_title(name)
            axes.ravel()[index].set_axis_off()

        plt.tight_layout()
        return(images_list)

    except Exception as e:
        print("Error:", str(e))

# Example usage with a URL
input_url = input("Enter the URL: ")
recm_user_input(input_url)

!mkdir templates

# Create 'index.html' template
with open('templates/index.html', 'w') as file:
    file.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fashion Recommendation</title>
</head>
<body class="bg-light">
    <div class="container mt-5">
        <h1 class="text-center">Welcome to the Fashion Recommender!</h1>
        <form action="/recommendation" method="post" class="mt-4">
        <label for="image_url">Enter Image URL:</label>
        <input type="text" id="image_url" name="image_url" required>
        <button type="submit">Get Recommendation</button>
    </form>
</body>
</html>
''')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for displaying recommendations
@app.route('/recommendation', methods=['POST'])
def show_recommendation():
    try:
        url = request.form['image_url']
        images_list = recm_user_input(url)
        return render_template('result.html', images_list=images_list)
    except Exception as e:
        error_message = "Error: " + str(e)
        return render_template('error.html', error_message=error_message)

# Create 'error.html' template
with open('templates/error.html', 'w') as file:
    file.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error</title>
</head>
<body>
    <h1>Error</h1>
    <p>{{ error_message }}</p>
</body>
</html>
''')

# Create 'result.html' template with Bootstrap styling
with open('templates/result.html', 'w') as file:
    file.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recommendation Result</title>
    <style>
        body {
            background-color: #f8f9fa;
        }

        .container {
            max-width: 800px;
            margin: 20px auto;
        }

        h1 {
            color: #007bff;
            text-align: center;
        }

        .result-card {
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.1);
        }

        .result-card img {
            width: 100%;
            height: auto;
            border-radius: 8px 8px 0 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Top 5 Recommendations</h1>
        <div class="row">
            {% for image_id in images_list %}
                <div class="col-md-6">
                    <div class="result-card">
                        <img src="{{ url_for('static', filename='/content/images/' + image_id) }}" alt="{{ image_id }}">
                    </div>
                </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
''')

if __name__ == '__main__':
    app.run()
