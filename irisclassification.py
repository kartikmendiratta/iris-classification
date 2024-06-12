
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_iris

# Load the dataset
iriskar = load_iris()
x, y = iriskar.data, iriskar.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Initialize the model
model = Sequential()

# neural network layers , activation function
model.add(Dense(64, input_dim=4, activation='relu'))  # Input layer , 4 features
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Output layer , 3 classes (iris species)


def precision(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    precision = tf.keras.metrics.Precision()
    precision.update_state(y_true, y_pred)
    return precision.result()

def recall(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    recall = tf.keras.metrics.Recall()
    recall.update_state(y_true, y_pred)
    return recall.result()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
model.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Evaluate the model
loss, accuracy, precision, recall = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")

from tensorflow.keras.models import load_model
model.save('iriskar.h5')
# Load the model from the file
model = load_model('iriskar.h5')
x=input("enter a number :")
y=input("enter a number :")
z=input("enter a number :")
n=input("enter a number :")
new_data = np.array([[x, y, z, n]])

# Normalize the new data point
new_data_normalized = scaler.transform(new_data)
# Predict the class of the new data
prediction = model.predict(new_data_normalized)
predicted_class = np.argmax(prediction,axis=1)
print(prediction)

# Map the predicted class index to the class label
class_labels = ['setosa', 'versicolor', 'virginica']  # Replace with actual class labels
print(f"Predicted class: {class_labels[predicted_class[0]]}")