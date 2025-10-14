


#  **Deep Learning Projects — CNN, RNN & Transfer Learning**

This module covers the **core fundamentals of Deep Learning** using **TensorFlow/Keras**, focusing on three essential architectures:
- **Convolutional Neural Networks (CNN)** for Image Classification  
- **Recurrent Neural Networks (RNN)** for Text/Sentiment Analysis  
- **Transfer Learning** using MobileNetV2 for Custom Dataset Fine-Tuning  

Each subproject demonstrates a practical workflow — from data preprocessing to model training, evaluation, and saving models for deployment.

---

##  Folder Structure

```

deep-learning-(cnn-rnn-transfer-learning)/
│
├── models/
│   │   ├──  movie_sentiment_rnn.h5
│   │   ├── digit_recognizer_cnn.h5
│   │   └── cat_vs_dog_transfer_learning.h5
│   │
│   │
│   │ 
├── notebooks/
│   │   ├── Cat_vs_Dog_Transfer_Learning_(MobileNetV2).ipynb
│   │   ├── Digit_Recognizer_CNN.ipynb
│   │   └── Movie_Review_Sentiment_RNN.ipynb
│   │ 
│   │   
├── README.md

```

---

##  **Project 1 — CNN: Digit Recognizer**

### **Objective**
Train a **Convolutional Neural Network (CNN)** to classify handwritten digits using the **MNIST dataset** (or CIFAR-10 for extended tasks).

### **Dataset Overview**
- **Source:** Keras Datasets (`tensorflow.keras.datasets.mnist`)
- **Shape:** 60,000 training, 10,000 test images (28×28 grayscale)
- **Classes:** 10 (digits 0–9)
- **Preprocessing:** Normalization (pixel values scaled 0–1)

### **Model Overview**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
````

### **Training Summary**

* Optimizer: `Adam`
* Loss: `categorical_crossentropy`
* Epochs: 10–15
* Achieved Accuracy: ~99% (MNIST) or ~85–90% (CIFAR-10)

### **Visual Placeholders**

* 🖼 *Insert here*: `training_curves.png` — Loss/Accuracy over epochs
* 🖼 *Insert here*: `confusion_matrix.png` — True vs Predicted labels
* 🖼 *Insert here*: `sample_predictions.png` — Random test predictions

### **Key Insights**

* Deeper CNNs improve feature extraction but risk overfitting on small datasets.
* Pooling reduces spatial dimension and computation cost.
* Batch normalization can stabilize learning and boost performance.

---

##  **Project 2 — RNN: Movie Review Sentiment Classifier**

### **Objective**

Build a **Recurrent Neural Network (RNN)** using **LSTM** layers to classify **IMDb movie reviews** as positive or negative.

### **Dataset Overview**

* **Source:** Keras Datasets (`tensorflow.keras.datasets.imdb`)
* **Size:** 50,000 reviews (25k train, 25k test)
* **Encoding:** Tokenized and integer-sequenced (word index vocabulary size = 10,000)
* **Padding:** Fixed length (e.g., 200 words per review)

### **Model Overview**

````
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

````

### **Training Summary**

* Optimizer: `Adam`
* Loss: `binary_crossentropy`
* Epochs: 5–8
* Validation Accuracy: ~86–88%


### **Key Insights**

* LSTM handles sequential dependencies well (unlike standard RNNs prone to vanishing gradients).
* Embedding layers learn semantic relationships between words.
* Dropout helps avoid overfitting in text-based models.

---

##  **Project 3 — Transfer Learning: Cats vs Dogs Classifier**

### **Objective**

Fine-tune **MobileNetV2** on a small custom dataset (e.g., Cats vs Dogs) to demonstrate the efficiency of **Transfer Learning**.

### **Dataset Overview**

* **Source:** Kaggle Cats vs Dogs Dataset (or local dataset)
* **Classes:** 2 (Cat, Dog)
* **Size:** ~2,000–5,000 images (resized to 160×160)
* **Preprocessing:** Data augmentation, normalization

### **Model Overview**

````
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160,160,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # freeze base layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
````

### **Training Summary**

* Optimizer: `Adam`
* Loss: `binary_crossentropy`
* Epochs: 10–15 (fine-tune top layers)
* Accuracy: ~92–96%


### **Key Insights**

* Transfer Learning drastically reduces training time and improves accuracy on small datasets.
* Fine-tuning upper layers allows adapting pretrained features to new domains.
* Data augmentation enhances generalization and prevents overfitting.

---

##  Model Saving & Deployment

Each trained model is saved in **HDF5 (.h5)** format for reuse or integration:

````
model.save('model_name.h5')
````

Optionally, export supporting artifacts (`LabelEncoders`, scalers) via `.joblib` for consistent preprocessing pipelines.

---

##  Summary Insights

| Model Type        | Dataset          | Accuracy  | Key Strength                              |
| ----------------- | ---------------- | --------- | ----------------------------------------- |
| CNN               | MNIST / CIFAR-10 | 99% / 88% | Excellent for spatial pattern recognition |
| RNN (LSTM)        | IMDb Reviews     | 87%       | Captures sequential dependencies          |
| Transfer Learning | Cats vs Dogs     | 94%       | Leverages pretrained features efficiently |

---

##  Future Improvements

* Experiment with **BatchNorm + Dropout** combinations for better generalization.
* Visualize **Grad-CAM** heatmaps for CNN/Transfer Learning interpretability.
* Deploy as **REST API** using FastAPI or Streamlit for live inference demos.
* Log experiments with **MLflow** or **Weights & Biases** for MLOps tracking.

---

## 🧾 Deliverables

| File                         | Description                               |
| ---------------------------- | ----------------------------------------- |
| `Digit_Recognizer_CNN.ipynb` | Handwritten digit classification notebook |
| `Sentiment_RNN.ipynb`        | IMDb text sentiment analysis notebook     |
| `Transfer_MobileNetV2.ipynb` | Fine-tuned transfer learning notebook     |
| `.h5 models`                 | Trained model weights for deployment      |
| `outputs/*.png`              | All visual results & metrics              |



---

* **Author:** Yash Desai
* **Email:** desaisyash1000@gmail.com 
* **GitHub:** yashdesai023

