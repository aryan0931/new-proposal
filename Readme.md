

# NuScenes Dataset for Autonomous Vehicle Project using IANVS

## Overview

This repository provides an extensive framework for processing and analyzing the **NuScenes** multimodal dataset used in autonomous vehicle research. The dataset contains data from multiple sensors such as **LiDAR**, **RGB Images**, **Video**, **Radar**, and **GPS/IMU**. The goal of this project is to leverage **early fusion** techniques to integrate these sensor data sources effectively, improving vehicle perception and decision-making. The project uses **IANVS** (KubeEdge-Ianvs), a distributed cloud-edge computing platform, to manage data preprocessing, model training, and deployment.

The main objectives of this project are:

1. **Multimodal Data Fusion**: Integrating LiDAR, image, video, and other sensor data.
2. **Data Preprocessing**: Proper formatting and handling of different data types for machine learning.
3. **Early Fusion Approach**: Merging raw data or feature-level representations from multiple sensors before feeding them into machine learning models.
4. **Autonomous Vehicle Perception**: Using fused data to detect and understand the environment around the vehicle.

## System Architecture

This project leverages **IANVS** for distributed cloud-edge processing. Here's a breakdown of the architecture:

### Cloud Side
* **Data Collection & Preprocessing**: Raw data from sensors (LiDAR, images, video, etc.) is collected and preprocessed. This task is handled by cloud nodes.
* **Model Training**: Machine learning models are trained using the processed multimodal data.

### Edge Side
* **Inference**: The trained models are deployed to edge nodes for real-time inference.
* **Data Aggregation**: Edge nodes aggregate data from various sensors and forward it to the cloud for further processing.

## Installation Instructions

### Requirements
* Python 3.8+
* Pip (or Conda for environment management)
* Docker (for containerized deployment, optional but recommended)
* Kubernetes (for cloud-edge computing via IANVS)
* TensorFlow / PyTorch (for model training and evaluation)
* OpenCV (for image and video processing)
* PCL (Point Cloud Library for LiDAR data)
* Scikit-learn (for evaluation metrics)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/gaymo-nuscense-ianvs.git
   cd gaymo-nuscense-ianvs
   ```
2. **Create a Python virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. **Install the required Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Docker Setup (Optional)
To containerize the environment for consistency and scalability, you can use Docker:
1. **Build Docker image:**
   ```bash
   docker build -t gaymo-nuscense-ianvs .
   ```
2. **Run the Docker container:**
   ```bash
   docker run -it gaymo-nuscense-ianvs
   ```



## Data Preprocessing

### LiDAR Data Preprocessing
LiDAR data is 3D point cloud data that can be processed with the following steps:
1. **Voxelization**: Converts the point cloud to a 3D grid (voxels) for efficient computation.
2. **Bird's Eye View (BEV) Projection**: Projects the 3D data onto a 2D plane for fusion with image data.

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
In early fusion, data from different modalities (LiDAR, images, radar) are combined before being fed into the model. This is done by concatenating their feature representations or raw data:

```python
import numpy as np

def early_fusion(lidar_data, image_data, radar_data):
    # Concatenate data from different sensors
    fused_data = np.concatenate([lidar_data, image_data, radar_data], axis=-1)
    return fused_data
```

## Model Training and Inference

### Model Training
After preprocessing and fusing the data, the next step is model training. You can use deep learning models such as 3D CNNs or Multimodal Transformers.

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

## Testing and Evaluation

**Key Evaluation Metrics:**
* **Accuracy:** Measures the percentage of correctly predicted labels.
* **Precision and Recall:** Evaluate the model's ability to correctly identify positive and negative instances.
* **F1 Score:** The harmonic mean of Precision and Recall, balancing both.
* **Intersection over Union (IoU):** Measures overlap between predicted and ground truth bounding boxes (useful for object detection).
* **Mean Average Precision (mAP):** Evaluates the average precision across different classes and IoU thresholds.
* **Latency:** Measures the time taken for the model to process and infer on edge devices.

**Testing Procedure:**
* **Dataset Splitting:**
  * Training Set: 70% of the dataset for model training.
  * Validation Set: 15% for hyperparameter tuning.
  * Test Set: 15% to evaluate model performance.
* **Cross-Validation:** K-fold cross-validation ensures robustness by training and testing the model on multiple dataset partitions.
* **Stress Testing:** Evaluate model performance under challenging conditions (e.g., poor lighting, occlusions, adverse weather).
* **Edge Deployment Testing:** Evaluate model inference on edge devices for latency and throughput.

**Example Evaluation Code:**

```python
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
```

## Contributing
We welcome contributions to this project! Please fork the repository, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License.
