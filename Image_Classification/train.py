import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 dataset
(traning_images, traning_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
traning_images, testing_images = traning_images / 255 , testing_images / 255

# Resize images to 64x64
traning_images = np.array([cv.resize(img, (64, 64)) for img in traning_images])
testing_images = np.array([cv.resize(img, (64, 64)) for img in testing_images])

# Define class names
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Display sample images with labels
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(traning_images[i])
    plt.xlabel(class_names[traning_labels[i][0]])
plt.show()

# Reduce dataset size for faster training (optional)
'''traning_images = traning_images[:200000]
traning_labels = traning_labels[:200000]
testing_images = testing_images[:40000]
testing_labels = testing_labels[:40000]
'''

# Define the convolutional neural network model
model = models.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10,activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(traning_images, traning_labels, epochs=20, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_model64.keras')
