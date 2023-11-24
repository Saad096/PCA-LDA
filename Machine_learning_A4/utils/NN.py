import numpy as np


class NN:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        # Define your neural network architecture
        self.input_size = 9  # Input size
        self.hidden_size = 64  # Number of hidden neurons
        self.output_size = 10  # Number of output classes

        # Initialize weights and self.biases
        np.random.seed(0)
        self.weights = {
            "hidden": np.random.randn(self.input_size, self.hidden_size),
            "output": np.random.randn(self.hidden_size, self.output_size),
        }
        self.biases = {
            "hidden": np.zeros((1, self.hidden_size)),
            "output": np.zeros((1, self.output_size)),
        }

    # Define the sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Define the softmax function for multi-class classification
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Define the cross-entropy loss function
    def cross_entropy_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = -np.log(y_pred[range(m), y_true]).mean()
        return loss

    # Training parameters

    def train(self):
        learning_rate = 0.01
        epochs = 100
        batch_size = 32

        # Training loop
        for epoch in range(epochs):
            for i in range(0, len(self.x_train), batch_size):
                # Mini-batch data
                batch_x = self.x_train[i : i + batch_size]
                batch_y = self.y_train[i : i + batch_size]

                # Forward propagation
                hidden_input = (
                    np.dot(batch_x, self.weights["hidden"]) + self.biases["hidden"]
                )
                hidden_output = self.sigmoid(hidden_input)

                output_input = (
                    np.dot(hidden_output, self.weights["output"])
                    + self.biases["output"]
                )
                output_output = self.softmax(output_input)

                # Compute loss
                loss = self.cross_entropy_loss(batch_y, output_output)

                # Backpropagation
                d_output = output_output
                d_output[range(len(batch_y)), batch_y] -= 1
                d_output /= len(batch_y)

                d_hidden = np.dot(d_output, self.weights["output"].T)
                d_hidden = d_hidden * hidden_output * (1 - hidden_output)

                # Update weights and self.biases
                self.weights["output"] -= learning_rate * np.dot(
                    hidden_output.T, d_output
                )
                self.biases["output"] -= learning_rate * np.sum(
                    d_output, axis=0, keepdims=True
                )

                self.weights["hidden"] -= learning_rate * np.dot(batch_x.T, d_hidden)
                self.biases["hidden"] -= learning_rate * np.sum(
                    d_hidden, axis=0, keepdims=True
                )

            # Print loss for this epoch
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    # Evaluation
    def predict(self, x):
        hidden_input = np.dot(x, self.weights["hidden"]) + self.biases["hidden"]
        hidden_output = self.sigmoid(hidden_input)

        output_input = (
            np.dot(hidden_output, self.weights["output"]) + self.biases["output"]
        )
        output_output = self.softmax(output_input)

        return np.argmax(output_output, axis=1)

    # Calculate precision, recall, and F1-score
    def calculate_metrics(self, y_true, y_pred, num_classes):
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)

        for c in range(num_classes):
            true_positives = np.sum((y_true == c) & (y_pred == c))
            false_positives = np.sum((y_true != c) & (y_pred == c))
            false_negatives = np.sum((y_true == c) & (y_pred != c))

            if true_positives == 0:
                precision[c] = 0.0
                recall[c] = 0.0
                f1[c] = 0.0
            else:
                precision[c] = true_positives / (true_positives + false_positives)
                recall[c] = true_positives / (true_positives + false_negatives)
                f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])

        return precision, recall, f1

    def print_metrics(self):
        self.y_pred = self.predict(self.x_test)
        precision, recall, f1 = self.calculate_metrics(
            self.y_test, self.y_pred, self.output_size
        )
        macro_precision = np.nanmean(precision)
        macro_recall = np.nanmean(recall)
        macro_f1 = np.nanmean(f1)

        # Calculate accuracy
        accuracy = np.mean(self.y_pred == self.y_test)

        print("Test Precision:", macro_precision)
        print("Test Recall:", macro_recall)
        print("Test F1-Score:", macro_f1)
        print("Test Accuracy:", accuracy)
