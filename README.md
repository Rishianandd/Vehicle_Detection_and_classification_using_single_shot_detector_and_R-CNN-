Vehicle Detection using SSD and PyTorch

Overview

This project implements a vehicle detection system using a pre-trained Single Shot MultiBox Detector (SSD) with a VGG16 backbone. The model is trained on a dataset of vehicle images with annotated bounding boxes and is optimized for accuracy and performance using PyTorch and CUDA-enabled GPUs.

Features

Vehicle Detection: Identifies vehicles in images using object detection.

Data Augmentation: Enhances training with Albumentations for better model generalization.

Custom PyTorch Dataset: Loads and processes data dynamically.

GPU Acceleration: Utilizes CUDA for faster training.

Model Optimization: Implements Adam optimizer and Non-Maximum Suppression (NMS).

Inference & Visualization: Draws bounding boxes on detected vehicles using OpenCV and Matplotlib.

Technologies Used

Python

PyTorch

Torchvision

Albumentations

OpenCV

Matplotlib

Pandas

NumPy

Installation

Clone the repository:

git clone https://github.com/yourusername/vehicle-detection.git
cd vehicle-detection

Install dependencies:

pip install -r requirements.txt

Ensure you have a CUDA-compatible GPU (optional for faster training):

python -c "import torch; print(torch.cuda.is_available())"

Dataset

The dataset consists of vehicle images with labeled bounding boxes.

Bounding box data is stored in a CSV file.

Example format:

image,xmin,ymin,xmax,ymax,label
vid_1.jpg,50,60,200,220,1
vid_2.jpg,30,40,150,170,1

Training the Model

Run the training script:

python train.py

Training progress will be displayed, showing loss reduction over epochs.

The best model is saved as model_ssd.pt.

Testing the Model

Load the trained model and run inference:

python test.py --image test_image.jpg

The output image with detected vehicles will be displayed.

Project Workflow

Data Preprocessing: Load dataset, apply augmentations.

Model Selection: Use SSD300 with VGG16 backbone.

Training: Train using PyTorch, optimize with Adam.

Inference: Predict vehicles in unseen images.

Visualization: Display results with bounding boxes.

Results

319 training images with 491 bounding boxes used for training.

Best model checkpoint saved at lowest loss.

GPU acceleration improves training efficiency.

Future Improvements

Train on a larger dataset for improved accuracy.

Experiment with YOLO or Faster R-CNN for comparison.

Deploy as a real-time application using Flask or FastAPI.

Author

Rishi Anand

LinkedIn: Your Profile






You can Download the Dataset from the given link :https://www.kaggle.com/datasets/sshikamaru/car-object-detection
After successful execution of the program the Output will look like this :  <img width="278" alt="ssd output" src="https://github.com/user-attachments/assets/be0aa55c-7b26-4ad0-a1f6-168ae400e626">
