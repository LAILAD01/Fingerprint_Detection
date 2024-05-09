import numpy as np
from image_processing import load_and_preprocess_images
from conv_layer import ConvNet
from max_pooling import MaxPooling
from flattening import Flatten
from neural_network import NeuralNetwork
from utils import batch_process

# Load data
directory = './data/SOCOFing/Real'
loaded_data = load_and_preprocess_images(directory)
print(type(loaded_data), loaded_data)

# Unpack the tuple returned by the function
images_with_their_labels, labels, additional_data = loaded_data

# Assuming each key in the dictionary is an image ID and the value is the image data
images = np.array(list(images_with_their_labels.values()))

# Create instances
convNet = ConvNet(num_filters=8, filter_h=3, filter_w=3, stride=1)
max_pooling = MaxPooling(pool_h=2, pool_w=2, stride=2)
flatten = Flatten()
neural_network = NeuralNetwork(input_size=128 * 128 * 8 // 4, num_classes=10, hidden_units=100, learning_rate=0.01)

# Train or predict
for epoch in range(10):  # Number of epochs
    for batch_images, batch_labels in batch_process(images, labels, batch_size=100):
        feature_map = convNet.conv_forward(batch_images, 'same')
        pooled_feature_map = max_pooling.forward(feature_map)
        flattened_feature_map = flatten.forward(pooled_feature_map)
        predictions = neural_network.forward(flattened_feature_map)
        loss = neural_network.compute_loss(predictions, batch_labels)
        neural_network.backward(batch_labels)
        print(f"Epoch {epoch}, Loss: {loss}")

    # Save model weights after training
    neural_network.save_model('path_to_save_model_weights.npz')
