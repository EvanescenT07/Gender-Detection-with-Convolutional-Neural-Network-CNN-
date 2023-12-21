import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from glob import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Function to read and resize images
def read_images(data):
    lst_images = [cv2.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), (299, 299)) for img in data]
    return lst_images

# Paths to the dataset
data_female = glob("D:/Documents/President University/Semester 4/Deep Learning/Face Detection/Training/female/*.jpg")
data_male = glob("D:/Documents/President University/Semester 4/Deep Learning/Face Detection/Training/male/*.jpg")

# Read and resize images
lst_imgs_male = read_images(data_male)
lst_imgs_female = read_images(data_female)

# Display sample images
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.ravel()

for i in range(3):
    male_index = i
    female_index = i
    
    # Check if the index is within the range of lst_imgs_male
    if male_index < len(lst_imgs_male):
        axes[male_index].imshow(lst_imgs_male[male_index], cmap='gray', interpolation='none')
        axes[male_index].set_title("Gender: Male")
        axes[male_index].axis('off')

    # Check if the index is within the range of lst_imgs_female
    if female_index < len(lst_imgs_female):
        axes[female_index + 3].imshow(lst_imgs_female[female_index], cmap='gray', interpolation='none')
        axes[female_index + 3].set_title("Gender: Female")
        axes[female_index + 3].axis('off')

plt.tight_layout()
plt.show()

# Labels for the images
labels_male = np.ones(len(lst_imgs_male))
labels_female = np.zeros(len(lst_imgs_female))

# Combine data and labels
Y = np.concatenate([labels_male, labels_female])
X = np.concatenate([lst_imgs_male, lst_imgs_female])

# Convert to NumPy arrays
Y = Y.astype("float32")
X = X.astype("float32") / 255.0

# Remove duplicate data
unique_indices = np.unique(X, axis=0, return_index=True)[1]
X = X[unique_indices]
Y = Y[unique_indices]

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=100)
print("X_train: ", X_train.shape, "  X_test: ", X_test.shape)

# Build a simple model for gender detection
model = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=1, padding="same", activation="relu", input_shape=(299, 299, 3)),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Display the model summary
print(model.summary())

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
model.evaluate(X_test, y_test)
