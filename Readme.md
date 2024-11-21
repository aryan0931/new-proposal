
# NuScenes Dataset for Autonomous Vehicle Project using IANVS

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

Testing and Evaluation

Evaluation is a critical step to ensure that the trained models meet performance expectations and are robust across diverse scenarios. This project includes several testing and evaluation techniques:

Key Evaluation Metrics
Accuracy: Measures the percentage of correctly predicted labels.
Formula:
Accuracy
=
Number of Correct Predictions
Total Number of Predictions
Accuracy= 
Total Number of Predictions
Number of Correct Predictions
​	
 
Precision and Recall:
Precision: Fraction of correctly predicted positive instances out of all predicted positives.
Precision
=
True Positives
True Positives
+
False Positives
Precision= 
True Positives+False Positives
True Positives
​	
 
Recall: Fraction of correctly predicted positive instances out of all actual positives.
Recall
=
True Positives
True Positives
+
False Negatives
Recall= 
True Positives+False Negatives
True Positives
​	
 
F1 Score: The harmonic mean of Precision and Recall, balancing both.
Formula:
F1 Score
=
2
×
Precision
×
Recall
Precision
+
Recall
F1 Score=2× 
Precision+Recall
Precision×Recall
​	
 
Intersection over Union (IoU): Measures overlap between predicted and ground truth bounding boxes (useful for object detection).
Formula:
IoU
=
Area of Overlap
Area of Union
IoU= 
Area of Union
Area of Overlap
​	
 
Mean Average Precision (mAP): Evaluates the average precision across different classes and IoU thresholds.
Latency: Measures the time taken for the model to process and infer on edge devices.
Testing Procedure
Dataset Splitting:
Training Set: 70% of the dataset for model training.
Validation Set: 15% for hyperparameter tuning.
Test Set: 15% to evaluate model performance.
Cross-Validation:
K-fold cross-validation ensures robustness by training and testing the model on multiple dataset partitions.
Stress Testing:
Evaluate model performance under challenging conditions such as:
Poor lighting (e.g., night-time scenes).
Occlusions (e.g., partially visible objects).
Adverse weather conditions (e.g., rain, fog).
Edge Deployment Testing:
Evaluate model inference on edge devices for latency and throughput.
Monitor performance under real-world conditions.
Example Evaluation Code
Accuracy, Precision, Recall, and F1 Score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predictions and ground truth
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
IoU Calculation for Object Detection

def calculate_iou(box1, box2):
    # Box format: [x_min, y_min, x_max, y_max]
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # IoU calculation
    return intersection / union if union > 0 else 0

box_a = [50, 50, 150, 150]
box_b = [60, 60, 170, 170]

iou = calculate_iou(box_a, box_b)
print(f"IoU: {iou:.2f}")
Evaluation Scenarios
Multimodal Testing: Evaluate how well the model fuses and utilizes data from multiple modalities like LiDAR and camera inputs.
Scenario-Specific Testing: Test performance under specific scenarios, such as highway driving, urban traffic, or low-visibility conditions.
Energy Efficiency: Measure the model’s energy consumption on edge devices to ensure sustainability.
Test Results and Reporting
Visualizations:
Generate confusion matrices for classification tasks.
Visualize IoU overlaps for object detection.
Performance Comparison:
Compare results against baseline models and industry benchmarks.
Reporting: Save evaluation results in JSON or CSV format for further analysis.
import json

results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "iou": iou
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)
By integrating these testing and evaluation practices, the project ensures comprehensive analysis, robustness, and adaptability of the trained models across various scenarios and condi



## Contributing

We welcome contributions to this project! Please fork the repository, create a new branch, and submit a pull request.

---

## License

This project is licensed under the MIT License.

---


