import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def lda_basis(X_train, y_train, n_components=9):
    n_classes = 10
    class_means = np.array(
        [X_train[y_train == i].mean(axis=0) for i in range(n_classes)]
    )

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train, y_train)
    return lda, class_means


def project_data_lda(lda, X, y_train, class_means):
    X_lda = np.dot(X - class_means[y_train], lda.scalings_)
    return X_lda


def build_nn(input_dim, output_dim, hidden_units=(64, 32)):
    # Initialize weights and biases
    np.random.seed(0)
    weights = [np.random.randn(input_dim, hidden_units[0])]
    biases = [np.zeros((1, hidden_units[0]))]

    for i in range(1, len(hidden_units)):
        weights.append(np.random.randn(hidden_units[i - 1], hidden_units[i]))
        biases.append(np.zeros((1, hidden_units[i])))

    weights.append(np.random.randn(hidden_units[-1], output_dim))
    biases.append(np.zeros((1, output_dim)))

    return weights, biases


def forward_pass(X, weights, biases):
    activations = [X]
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        activation = 1 / (1 + np.exp(-z))  # Sigmoid activation
        activations.append(activation)
    return activations


def train_nn(
    X_train, y_train, weights, biases, learning_rate=0.01, epochs=10, batch_size=8
):
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            # Forward pass
            activations = forward_pass(X_batch, weights, biases)
            output = activations[-1]

            print(output.shape, y_batch.shape)
            # Backpropagation
            d_output = output - y_batch
            for j in range(len(weights) - 1, -1, -1):
                d_activation = np.dot(d_output, weights[j].T)
                d_z = d_activation * (activations[j + 1] * (1 - activations[j + 1]))
                d_weights = np.dot(activations[j].T, d_z)
                d_biases = np.sum(d_z, axis=0)
                d_output = np.dot(d_z, weights[j].T)
                weights[j] -= learning_rate * d_weights
                biases[j] -= learning_rate * d_biases


def evaluate_lda_model(X_test, y_test, weights, biases):
    # Forward pass on test data
    activations = forward_pass(X_test, weights, biases)
    y_pred = activations[-1]

    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class, average="macro")
    recall = recall_score(y_test, y_pred_class, average="macro")
    f1 = f1_score(y_test, y_pred_class, average="macro")

    return accuracy, precision, recall, f1


# Helper function to standardize the data
def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test
