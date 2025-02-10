# 📌 Image Classification Project: Vegetable

## 📖 Project Description

This project focuses on classifying vegetable images using deep learning models based on TensorFlow and Keras. The dataset used is from Kaggle: [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset).

## ⚙️ Package Installation

Before running this project, install the required packages by executing:

```bash
pip install tensorflow tensorflowjs kagglehub matplotlib numpy pandas scikit-learn
```

## 📂 Dataset Structure

The dataset is divided into three main parts:

-   **Train**: Training data
-   **Test**: Testing data
-   **Validation**: Validation data

## 🛠️ Project Workflow

1. **Import Libraries**: Load required packages like TensorFlow, Keras, Matplotlib, and NumPy.
2. **Data Preparation**:
    - Load dataset
    - Label data
    - Preprocess images (resizing, normalization)
    - Split dataset into training, validation, and test sets
3. **Model Training**:
    - CNN architecture with Conv2D, MaxPooling, and Dense layers
    - Adam optimizer with a learning rate of 0.0001
    - Early stopping for optimal training
4. **Model Evaluation**:
    - Generate Confusion Matrix and Classification Report
    - Visualize accuracy and loss trends
5. **Model Conversion**:
    - Save the model in `.keras`, TensorFlow.js, and TFLite formats

## 🎯 Running Inference

The trained model can be used to classify new images as follows:

### 🔹 Using TensorFlow Model:

```python
from tensorflow import keras
import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

model = keras.models.load_model('submission/saved_model.keras')
predicted_label, confidence = predict_image('path/to/image.jpg', model, labels)
print(f'Predicted: {predicted_label} ({confidence:.2f})')
```

### 🔹 Using TFLite Model:

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='submission/tflite/model.tflite')
interpreter.allocate_tensors()
```

## 📊 Results

The model achieved approximately **96%** accuracy on the validation dataset with a high f1-score for all classes.

```
              precision    recall  f1-score   support

     pumpkin       0.95      0.96      0.96       280
      papaya       0.99      0.97      0.98       280
      potato       0.97      0.99      0.98       280
     cabbage       0.91      0.96      0.94       280
     brinjal       0.90      0.97      0.93       280
bitter_gourd       0.89      0.98      0.93       280
      tomato       0.99      0.99      0.99       280
    broccoli       0.98      0.99      0.99       280
      carrot       0.97      0.94      0.95       280
        bean       0.98      0.94      0.96       280
    cucumber       0.97      0.96      0.97       280
 cauliflower       0.99      0.97      0.98       280
    capsicum       0.98      0.91      0.94       280
bottle_gourd       0.98      0.99      0.99       280
      radish       0.98      0.91      0.94       280

    accuracy                           0.96      4200
   macro avg       0.96      0.96      0.96      4200
weighted avg       0.96      0.96      0.96      4200
```

## 📜 License

This project is intended for educational and research purposes. You are free to use it with proper attribution.

---

> _🚀 Created by Joshua Palti Sinaga, 2025_
