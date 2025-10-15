# predict.py
import joblib
import numpy as np

# Load trained model
model = joblib.load('models/iris_model.joblib')

# Example input (replace with dynamic input later)
sample = np.array([[7.2, 3.0, 5.8, 1.6]])  # sepal_length, sepal_width, petal_length, petal_width
prediction = model.predict(sample)

class_names = ['setosa', 'versicolor', 'virginica']
print("Predicted class:", class_names[prediction[0]])
