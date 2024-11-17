
# Gaymo NuScenes Dataset for Autonomous Vehicle Project using IANVS

## Overview

This repository provides an extensive framework for processing and analyzing the **Gaymo NuScenes** multimodal dataset used in autonomous vehicle research. The dataset contains data from multiple sensors such as **LiDAR**, **RGB Images**, **Video**, **Radar**, and **GPS/IMU**. The goal of this project is to leverage **early fusion** techniques to integrate these sensor data sources effectively, improving vehicle perception and decision-making. The project uses **IANVS** (KubeEdge-Ianvs), a distributed cloud-edge computing platform, to manage data preprocessing, model training, and deployment.

The main objectives of this project are:
1. **Multimodal Data Fusion**: Integrating LiDAR, image, video, and other sensor data.
2. **Data Preprocessing**: Proper formatting and handling of different data types for machine learning.
3. **Early Fusion Approach**: Merging raw data or feature-level representations from multiple sensors before feeding them into machine learning models.
4. **Autonomous Vehicle Perception**: Using fused data to detect and understand the environment around the vehicle.

---

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [System Architecture](#system-architecture)
3. [Installation Instructions](#installation-instructions)
4. [Project Setup](#project-setup)
5. [Data Preprocessing](#data-preprocessing)
   - [LiDAR Data Preprocessing](#lidar-data-preprocessing)
   - [Image and Video Data Preprocessing](#image-and-video-data-preprocessing)
   - [Early Fusion of Multimodal Data](#early-fusion-of-multimodal-data)
6. [Model Training and Inference](#model-training-and-inference)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Deployment in IANVS](#deployment-in-ianvs)
9. [Project Structure](#project-structure)
10. [Contributing](#contributing)
11. [License](#license)

---

## Dataset Description

The **Gaymo NuScenes** dataset is a multimodal dataset designed for autonomous vehicle research, offering a rich set of sensor data. The dataset consists of the following modalities:

- **LiDAR Point Clouds (3D)**: Captures spatial data, helping detect objects and surfaces in the vehicle’s surroundings.
- **RGB Images**: Camera images provide visual input for object detection and scene understanding.
- **Video Frames**: Time-sequenced video frames are used for motion tracking and decision-making.
- **Radar Data**: Detects the speed, direction, and distance of objects surrounding the vehicle.
- **GPS/IMU**: Provides geolocation information and the vehicle's orientation and movement data.

These modalities are synchronized with timestamps, allowing for precise alignment for multimodal fusion.

---

## System Architecture

This project leverages **IANVS** for distributed cloud-edge processing. Here's a breakdown of the architecture:

1. **Cloud Side**:
   - **Data Collection & Preprocessing**: Raw data from sensors (LiDAR, images, video, etc.) is collected and preprocessed. This task is handled by cloud nodes.
   - **Model Training**: Machine learning models are trained using the processed multimodal data.

2. **Edge Side**:
   - **Inference**: The trained models are deployed to edge nodes for real-time inference.
   - **Data Aggregation**: Edge nodes aggregate data from various sensors and forward it to the cloud for further processing.

---

## Installation Instructions

### Requirements

Make sure you have the following prerequisites in place:

- **Python 3.8+**
- **Pip** (or **Conda** for environment management)
- **Docker** (for containerized deployment, optional but recommended)
- **Kubernetes** (for cloud-edge computing via IANVS)
- **TensorFlow / PyTorch** (for model training and evaluation)
- **OpenCV** (for image and video processing)
- **PCL** (Point Cloud Library for LiDAR data)
- **Scikit-learn** (for evaluation metrics)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/gaymo-nuscense-ianvs.git
cd gaymo-nuscense-ianvs

# Create a Python virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install the required Python dependencies
pip install -r requirements.txt
```

### Docker Setup (Optional)

To containerize the environment for consistency and scalability, you can use Docker:

```bash
# Build Docker image
docker build -t gaymo-nuscense-ianvs .

# Run the Docker container
docker run -it gaymo-nuscense-ianvs
```

### Kubernetes Setup (For IANVS)

Follow the [KubeEdge documentation](https://kubeedge.io/en/docs/) to set up **KubeEdge** for cloud-edge processing. Afterward, deploy services that handle preprocessing and fusion tasks.

```bash
# Deploy the services on cloud and edge nodes
kubectl apply -f k8s-deployment.yaml
```

---

## Project Setup

### Configure Dataset Paths

Set the dataset paths in `config.yaml`.

```yaml
dataset:
  lidar_data: /path/to/lidar/data
  image_data: /path/to/image/data
  video_data: /path/to/video/data
  radar_data: /path/to/radar/data
  gps_imu_data: /path/to/gps_imu/data
```

### Set Up the Environment

Ensure all configurations are set correctly in `config.yaml`, particularly for paths and preprocessing parameters (e.g., LiDAR voxel size, image resizing).

---

## Data Preprocessing

### LiDAR Data Preprocessing

LiDAR data is 3D point cloud data that can be processed with the following steps:

1. **Voxelization**: Converts the point cloud to a 3D grid (voxels) for efficient computation.
2. **Bird's Eye View (BEV) Projection**: Projects the 3D data onto a 2D plane for fusion with image data.

Example:

```python
import pcl  # Point Cloud Library

def preprocess_lidar_data(point_cloud):
    # Voxelization
    voxel_grid = pcl.VoxelGrid()
    voxel_grid.set_leaf_size(0.2, 0.2, 0.2)
    voxel_grid.set_input_cloud(point_cloud)
    filtered_point_cloud = voxel_grid.filter()

    # BEV Projection
    bev_projection = project_to_bev(filtered_point_cloud)
    
    return bev_projection
```

### Image and Video Data Preprocessing

For image and video processing, we apply standard preprocessing steps like resizing and normalization.

```python
import cv2

def preprocess_image(image):
    # Resize
    image_resized = cv2.resize(image, (224, 224))

    # Normalize
    image_normalized = image_resized / 255.0

    return image_normalized

def preprocess_video(video_frames):
    frames = [preprocess_image(frame) for frame in video_frames]
    return frames
```

### Early Fusion of Multimodal Data

In **early fusion**, data from different modalities (LiDAR, images, radar) are combined before being fed into the model. This is done by concatenating their feature representations or raw data:

```python
import numpy as np

def early_fusion(lidar_data, image_data, radar_data):
    # Concatenate data from different sensors
    fused_data = np.concatenate([lidar_data, image_data, radar_data], axis=-1)
    return fused_data
```

---

## Model Training and Inference

### Model Training

After preprocessing and fusing the data, the next step is model training. You can use deep learning models such as **3D CNNs** or **Multimodal Transformers**.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

def create_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Change as per task (e.g., classification)
    ])
    return model

# Example training code
model = create_model((224, 224, 3))  # Example input shape
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(fused_data, labels, epochs=10, batch_size=32)
```

### Inference

After training, the model is deployed to edge nodes for real-time inference:

```python
def infer(model, input_data):
    predictions = model.predict(input_data)
    return predictions
```

---

## Testing and Evaluation

To evaluate the model's performance, metrics like **accuracy**, **precision**, and **recall** can be used.

```python
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

---

## Deployment in IANVS

Finally, the trained model is deployed in the **IANVS** environment for real-time inference across cloud and

 edge nodes. Configuration files like `k8s-deployment.yaml` help manage this process in **Kubernetes**.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: edge-inference
  template:
    metadata:
      labels:
        app: edge-inference
    spec:
      containers:
      - name: inference-container
        image: inference-image:latest
        ports:
        - containerPort: 8080
```

---

## Project Structure

```
.
├── config.yaml              # Configuration file for dataset paths and parameters
├── data/                    # Raw and preprocessed dataset
├── src/                     # Source code for preprocessing, training, and inference
│   ├── preprocessing.py     # Functions for data preprocessing
│   ├── model.py             # Functions for model architecture and training
│   └── inference.py         # Functions for running inference
├── k8s-deployment.yaml      # Kubernetes deployment configuration
└── README.md                # This file
```

---

## Contributing

We welcome contributions to this project! Please fork the repository, create a new branch, and submit a pull request.

---

## License

This project is licensed under the MIT License.

---


