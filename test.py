import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Assuming that NeuralNetwork and load_and_preprocess_images are imported
from neural_network import NeuralNetwork
from image_processing import load_and_preprocess_images

def main():
    data_dict = load_and_preprocess_images('./data/SOCOFing/Real')

    # Assuming the keys of the dictionary are IDs and the values are image arrays
    x_test = np.array([img.flatten() for img in data_dict.values()])# Convert dictionary values (images) to numpy array
    
    y_test = np.array([0] * len(x_test))  


    neural_network = NeuralNetwork(input_size=128*128, num_classes=10, hidden_units=100, learning_rate=0.01)


    # Making predictions
    predictions = neural_network.predict(x_test)

    # Calculate accuracy (modify if actual labels are used)
    # Here dummy labels mean accuracy will not be meaningful
    test_accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Confusion Matrix and Classification Report (modify if actual labels are used)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Visualization of predictions (if applicable)
    plot_sample_predictions(x_test, y_test, predictions)

def plot_sample_predictions(x, true_labels, predicted_labels, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 2))
    for i, ax in enumerate(axes):
        ax.imshow(x[i].reshape(128, 128), cmap='gray')  # Adjust this based on your actual image shape
        ax.set_title(f"True: {true_labels[i]}, Pred: {predicted_labels[i]}")
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    main()