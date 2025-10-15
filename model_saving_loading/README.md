
# Model Saving, Loading & Reuse — Iris Classification

This project demonstrates how to **train, save, load, and reuse Machine Learning models** using the **Iris Dataset**.  
It covers the practical workflow of persisting models with both `pickle` and `joblib`—a foundational step before model deployment.

---

##  Folder Structure

```

model_saving_loading/
│
├── models/
│   ├── iris_model.joblib         # Model saved using Joblib
│   └── iris_model.pkl            # Model saved using Pickle
│
├── notebooks/
│   └── Save_Load_Demo.ipynb      # Jupyter notebook for full demonstration
│
├── predict.py                    # Standalone script for loading and predicting
│
└── README.md                     # Project documentation

````

---

##  Project Overview

The **goal** is to train a simple classification model on the Iris dataset, then:
- Save it in two formats (`.pkl` and `.joblib`)
- Load both versions
- Verify they produce identical predictions
- Use a standalone script (`predict.py`) for inference

This prepares you for **model deployment workflows** in Flask, FastAPI, or Streamlit.

---

##  Dataset — Iris Flower Classification

**Source:** Scikit-learn’s built-in `load_iris()`  
**Samples:** 150 total  
**Classes:**
1. Setosa  
2. Versicolor  
3. Virginica  

**Features:**
| Feature | Description |
|----------|--------------|
| Sepal Length | Length of the sepal in cm |
| Sepal Width | Width of the sepal in cm |
| Petal Length | Length of the petal in cm |
| Petal Width | Width of the petal in cm |

**Objective:** Predict the flower species based on its measurements.

---

##  Workflow Summary

### 1. Load and Split the Data

````python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


* Dataset loaded directly from `sklearn`
* Split 80% training, 20% testing
````
---

### 2. Train the Model

````python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
````

> **Result:** Model achieved approximately **96–98% accuracy** on test data.

---

### 3. Save the Model

#### Using Pickle

````python
import pickle

with open('models/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)
````

#### Using Joblib

````python
import joblib
joblib.dump(model, 'models/iris_model.joblib')
````

**Insight:**

* `pickle` can serialize any Python object.
* `joblib` is optimized for NumPy-heavy models (faster and lighter).

---

### 4. Load and Test Saved Models

````python
# Load with Pickle
with open('models/iris_model.pkl', 'rb') as f:
    loaded_pickle = pickle.load(f)

# Load with Joblib
loaded_joblib = joblib.load('models/iris_model.joblib')

# Compare predictions
print("Pickle Model Accuracy:", accuracy_score(y_test, loaded_pickle.predict(X_test)))
print("Joblib Model Accuracy:", accuracy_score(y_test, loaded_joblib.predict(X_test)))
````

✅ **Result:** Both models reproduce identical accuracy and outputs.
No data loss or change after serialization.

---

### 5. Reusable Prediction Script (`predict.py`)

This script allows **standalone model prediction** — ideal for deployment or integration with APIs.

````python
import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/iris_model.joblib')

# Example input: [sepal_length, sepal_width, petal_length, petal_width]
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)

class_names = ['setosa', 'versicolor', 'virginica']
print("Predicted class:", class_names[prediction[0]])
````

**Run via Terminal:**

````bash
python predict.py
````


---

##  Key Learnings

* Both `pickle` and `joblib` effectively preserve trained model states.
* Re-loaded models maintain **identical accuracy** and **behavior**.
* `joblib` is preferred for large datasets or deep learning feature arrays.
* The same model can now be plugged into **deployment frameworks**.

---

##  Next Steps

* Wrap this `predict.py` inside a Flask/FastAPI endpoint.
* Create a Streamlit interface for real-time input prediction.
* Experiment with model versioning tools like **MLflow** or **DVC**.

---

##  Files Overview

| File                   | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `Save_Load_Demo.ipynb` | Jupyter notebook for training, saving, loading, and validation |
| `predict.py`           | Script for standalone model inference                          |
| `iris_model.pkl`       | Model serialized with Pickle                                   |
| `iris_model.joblib`    | Model serialized with Joblib                                   |
| `README.md`            | Documentation for project overview and usage                   |

---

* **Author:** Yash Desai
* **Email:** desaisyash1000@gmail.com 
* **GitHub:** yashdesai023

---
