import matplotlib.pyplot as plt
import numpy as np


label_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress',
    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
    'Ankle boot'
    ]

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(img, cmap='gray')

    true_label = np.argmax(true_label)
    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
  
    plt.xlabel("{} {:2.0f}% ({})".format(label_names[predicted_label],
                                100*np.max(predictions_array),
                                label_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    true_label = np.argmax(true_label)
 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def display_sample_multi(plot_test_data, test_labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(plot_test_data[i], cmap='gray')
        plt.xlabel(label_names[test_labels[i]])
    plt.show()

def display_multi(predictions, test_labels, plot_test_data):
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, plot_test_data)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)

    plt.show()

def display_single(predictions, test_labels, plot_test_data):
    i = 0
    plt.figure(figsize=(6,5))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, plot_test_data)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  test_labels)
    _ = plt.xticks(range(10), label_names, rotation=75)
    _ = plt.tick_params(axis='x', labelsize=6)  
    plt.show()

def display_model_performance(fit_history):
    accuracy = fit_history.history['acc']
    val_accuracy = fit_history.history['val_acc']
    loss = fit_history.history['loss']
    val_loss = fit_history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()