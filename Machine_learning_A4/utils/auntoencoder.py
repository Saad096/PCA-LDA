import numpy as np


# Define a simple autoencoder class
class Autoencoder:
    def __init__(self, input_size, hidden_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.weights = {
            "encoder": np.random.randn(input_size, hidden_size),
            "decoder": np.random.randn(hidden_size, input_size),
        }

        self.biases = {
            "encoder": np.zeros(hidden_size),
            "decoder": np.zeros(input_size),
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, data, epochs=50):
        for _ in range(epochs):
            # Forward pass
            encoder_output = self.sigmoid(
                np.dot(data, self.weights["encoder"]) + self.biases["encoder"]
            )
            decoder_output = self.sigmoid(
                np.dot(encoder_output, self.weights["decoder"]) + self.biases["decoder"]
            )

            # Backpropagation
            error = data - decoder_output
            d_output = error * self.sigmoid_derivative(decoder_output)

            error_hidden = d_output.dot(self.weights["decoder"].T)
            d_hidden = error_hidden * self.sigmoid_derivative(encoder_output)

            # Update weights and biases
            self.weights["encoder"] += data.T.dot(d_hidden) * self.learning_rate
            self.biases["encoder"] += np.sum(d_hidden, axis=0) * self.learning_rate
            self.weights["decoder"] += (
                encoder_output.T.dot(d_output) * self.learning_rate
            )
            self.biases["decoder"] += np.sum(d_output, axis=0) * self.learning_rate

    def predict(self, data):
        encoder_output = self.sigmoid(
            np.dot(data, self.weights["encoder"]) + self.biases["encoder"]
        )
        decoder_output = self.sigmoid(
            np.dot(encoder_output, self.weights["decoder"]) + self.biases["decoder"]
        )
        return decoder_output


# Train autoencoders for each class
def train_autoencoders(train_data, train_labels, hidden_size, epochs=50):
    class_autoencoders = {}
    unique_labels = np.unique(train_labels)

    for label in unique_labels:
        class_data = train_data[train_labels == label]
        #  print(class_data.shape)
        input_size = class_data.shape[1]

        autoencoder = Autoencoder(input_size, hidden_size)
        autoencoder.train(class_data, epochs)
        class_autoencoders[label] = autoencoder

    return class_autoencoders


# Visualize output of the autoencoder as an image
def visualize_autoencoder_output(autoencoder, input_data, output_path):
    output_data = autoencoder.predict(input_data)
    # Implement your own visualization code here to display the input and output images.
    # This may involve reshaping and displaying the images as needed.
    # For example, you can use matplotlib to plot the input and output images.

    import matplotlib.pyplot as plt

    num_samples = min(5, len(input_data))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(input_data[i].reshape(32, 32, 3) / 255.0)
        plt.title("Original")
        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(output_data[i].reshape(32, 32, 3))
        plt.title("Reconstructed")

    plt.show()


# Function to measure reconstruction error
def reconstruction_error(input_data, output_data):
    error = input_data - output_data
    mse = np.mean(np.square(error))
    return mse


# Function to find the label of a test image using autoencoder based on minimum error criterion
def predict_label(test_image, class_autoencoders):
    min_error = float("inf")
    predicted_label = None

    for label, autoencoder in class_autoencoders.items():
        reconstructed_image = autoencoder.predict(test_image)
        error = reconstruction_error(test_image, reconstructed_image)

        if error < min_error:
            min_error = error
            predicted_label = label

    return predicted_label


# Evaluate models' performance
def evaluate_performance(test_images, true_labels, class_autoencoders):
    predicted_labels = [
        predict_label(image, class_autoencoders) for image in test_images
    ]

    correct_predictions = np.sum(np.array(predicted_labels) == true_labels)
    accuracy = correct_predictions / len(true_labels)

    # Calculate precision, recall, and F1-score for each class
    unique_labels = np.unique(true_labels)
    precision_list = []
    recall_list = []
    f1_list = []

    for label in unique_labels:
        true_positives = np.sum(
            (true_labels == label) & (np.array(predicted_labels) == label)
        )
        false_positives = np.sum(
            (true_labels != label) & (np.array(predicted_labels) == label)
        )
        false_negatives = np.sum(
            (true_labels == label) & (np.array(predicted_labels) != label)
        )

        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "Accuracy": accuracy,
        "Precision": precision_list,
        "Recall": recall_list,
        "F1-Score": f1_list,
    }
