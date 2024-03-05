from tensorflow.keras.models import load_model  # Importing load_model function
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load the saved Keras model
model = load_model('image_model.keras')

# Load and preprocess the image
img = cv.imread('Image_Classification/horse.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))  # Resize the image to match the input size of the model
img = img / 255.0  # Normalize pixel values to [0, 1]

plt.imshow(img, cmap=plt.cm.binary)
#plt.show()

# Perform prediction
prediction = model.predict(np.array([img]))
index = np.argmax(prediction)
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
print(f"Prediction is {class_names[index]}")