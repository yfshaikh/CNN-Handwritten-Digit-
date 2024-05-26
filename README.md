# Introduction

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The CNN is built using PyTorch and includes features such as data normalization, training with GPU support, and visualization of training progress.


## Installation
To install and set up the project, follow these steps:

1. Clone the repository:


`git clone https://github.com/yourusername/cnn-mnist-classifier.git` 

`cd cnn-mnist-classifier`

2. Create and activate a virtual environment (optional but recommended):


`python -m venv venv`

`source venv/bin/activate  # On Windows use venv\Scripts\activate`

3. Install the required packages:


`pip install -r requirements.txt`

## Usage

1. Training the Model:

To train the CNN on the MNIST dataset, run:

`python train.py`

This will download the MNIST dataset, train the CNN, and display training and testing loss and accuracy.

2. Predicting an Image:

To predict the class of a new image, use the predict function. Ensure the image is preprocessed to match the input requirements of the model:

`from predict import predict`

`prediction = predict('path_to_image.png', model)`

`print(f"Predicted class: {prediction}")`

## Features

**Data Normalization:** The input images are normalized using the mean and standard deviation of the MNIST dataset.

**Training with GPU Support:** The training script automatically uses GPU if available.

**Visualization:** Training and testing loss and accuracy are plotted for each epoch.

**Prediction:** Functionality to predict the class of new images using the trained model.

## Dependencies

Python 3.x

torch

torchvision

matplotlib

numpy

opencv-python

pillow

These dependencies can be installed via:


`pip install torch torchvision matplotlib numpy opencv-python pillow`

## Configuration

**Batch Size:** The batch size for training and testing can be configured in the train.py script.

**Learning Rate:** The learning rate for the optimizer can also be configured in the train.py script.

**Epochs:** The number of training epochs is set to 10 but can be modified as needed.

## Documentation

**Dataset:**

MNIST dataset is used for training and testing.

The dataset is automatically downloaded and stored in the ./temp directory.

**Transforms:**

The images are resized, converted to tensors, and normalized using specified mean and standard deviation.

**Model Architecture:**

The CNN consists of two convolutional layers followed by batch normalization, ReLU activation, and max-pooling.

The output from the convolutional layers is flattened and passed through two fully connected layers with a dropout in between.

## Examples

**Training Example:**

`python train.py`

**Prediction Example:**

`from predict import predict`

`prediction = predict('path_to_image.png', model)`

`print(f"Predicted class: {prediction}")`

## Troubleshooting

**Common Issues:**

Ensure all dependencies are installed correctly.

Verify the availability of GPU if CUDA is enabled.

Check the image preprocessing steps in the predict function to match the input dimensions and normalization.

## Contributors

Yusuf

## License
This project is licensed under the MIT License - see the LICENSE file for details.
