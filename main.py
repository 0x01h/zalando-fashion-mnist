import tensorflow
import keras
import helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import randint
from sklearn.model_selection import train_test_split


label_names = helpers.label_names
num_labels = len(label_names)

# Data Scraping
# train_labels -> train_data
# test_labels -> test_data
print('Training data are loading...')
train_data = pd.read_csv('fashion-mnist_train.csv', nrows=60000)
print('Training data loaded!')
print('Test data are loading...')
test_data = pd.read_csv('fashion-mnist_test.csv', nrows=10000)
print('Test data loaded!')

# Data Slicing
train_labels = train_data.iloc[:, 0]  # All rows, include only first "label" column.
test_labels = test_data.iloc[:, 0]
train_data = train_data.iloc[:, 1:]  # All rows, exclude only first "label" column.
test_data = test_data.iloc[:, 1:]
train_data = np.array(train_data)  # Convert data frame to numpy array.
test_data = np.array(test_data)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Slice 0.1 of training dataset to create validation dataset.
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=randint(0, 100))

# Data Preprocessing
# Reshape all data from 1x784 -> 28x28x1 to get valid images.
train_data = np.array(list(map(lambda x: x.reshape(28, 28, 1), train_data)))
val_data = np.array(list(map(lambda x: x.reshape(28, 28, 1), val_data)))
test_data = np.array(list(map(lambda x: x.reshape(28, 28, 1), test_data)))

# Data Preprocessing
# Reshape all data from 28x28x1 to 28x28 to plot them successfully.
plot_test_data = np.array(list(map(lambda x: x.reshape(28, 28), test_data)))

# Get sample image.
sample_img = plot_test_data[0]
sample_img_label = label_names[test_labels[0]]

# Display sample image in colorbar.
plt.figure()
plt.imshow(sample_img)
plt.colorbar()
plt.grid(False)
plt.title(sample_img_label)
plt.show()

print('Training data shape:', train_data.shape)
print('Validation data shape:', val_data.shape)
print('Test data shape:', test_data.shape)
print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

# Data Preprocessing
# Normalization
# We want values between 0-1, not 0-255.
train_data = train_data / 255.0
val_data = val_data / 255.0
test_data = test_data / 255.0

# Display sample images with their labels to verify preprocessing phase.
# cmap='gray makes images black and white.
helpers.display_sample_multi(plot_test_data, test_labels)

# Label Encoding
# One hot encoding for categorical classification (categorical_crossentropy).
train_labels = keras.utils.to_categorical(train_labels, num_classes=num_labels)
val_labels = keras.utils.to_categorical(val_labels, num_classes=num_labels)
test_labels = keras.utils.to_categorical(test_labels, num_classes=num_labels)

# CNNs are best for image processing.
keras.model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal', # he_normal using in ImageNet challenge.
                 input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(num_labels, activation='softmax')
])

keras.model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fit_history = keras.model.fit(
    train_data, train_labels, epochs=25,
    batch_size=256, validation_data=(val_data, val_labels))

test_loss, test_acc = keras.model.evaluate(test_data, test_labels)
predictions = keras.model.predict(test_data)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Display performance of model comparing validations.
helpers.display_model_performance(fit_history)

print('Sample one prediction array:', predictions[1])  # Prediction results vary between 0 and 1 for every class.
print('Strongest prediction:', np.argmax(predictions[1]), '-', label_names[np.argmax(predictions[1])])  # Get the biggest prediction result's class (index of prediction array).
print('Test strongest prediction:', test_labels[1], '-', label_names[np.argmax(test_labels[1])])  # Verify by checking real test label.

# Display multiple image predictions and results.
helpers.display_multi(predictions, test_labels, plot_test_data)

# Display single image prediction and result.
helpers.display_single(predictions, test_labels, plot_test_data)