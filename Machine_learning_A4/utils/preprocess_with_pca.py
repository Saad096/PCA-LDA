import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def perform_pca(X_train, y_train, num_components=10):
    pca_models = []
    for class_label in range(10):
        class_indices = np.where(y_train == class_label)
        class_data = X_train[class_indices]

        # Standardize the data
        scaler = StandardScaler()
        class_data = scaler.fit_transform(class_data)

        # Perform PCA
        pca = PCA(n_components=num_components)
        pca.fit(class_data)
        pca_models.append((pca, scaler))

    return pca_models


def project_data(pca_models, data):
    projected_data = []
    for class_label in range(10):
        pca, scaler = pca_models[class_label]
        class_data = scaler.transform(data)
        projected_data.append(pca.transform(class_data))

    return np.concatenate(projected_data, axis=0)


def backproject_data(pca_models, projected_data):
    backprojected_data = []
    start_idx = 0
    for class_label in range(10):
        pca, scaler = pca_models[class_label]
        end_idx = start_idx + projected_data.shape[0] // 10
        class_projected_data = projected_data[start_idx:end_idx]
        backprojected_class_data = pca.inverse_transform(class_projected_data)
        backprojected_data.append(scaler.inverse_transform(backprojected_class_data))
        start_idx = end_idx

    return np.concatenate(backprojected_data, axis=0)


def compute_error(original_data, backprojected_data):
    return np.mean(np.square(original_data.reshape(1, -1) - backprojected_data))


def classify_with_pca_models(pca_models, projected_data, X_test):
    y_pred = []
    for sample in X_test:
        class_errors = []
        for class_label in range(10):
            pca, scaler = pca_models[class_label]
            backprojected_data = pca.inverse_transform(
                projected_data[class_label : class_label + 1]
            )
            backprojected_data = scaler.inverse_transform(backprojected_data)
            error = compute_error(sample, backprojected_data)
            class_errors.append(error)
        y_pred.append(np.argmin(class_errors))
    return np.array(y_pred)


def evaluate_pca_model(y_test, y_pred):
    # print(y_test.shape, y_pred.shape)
    print("Saad Alam")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=1
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=1
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", labels=np.unique(y_pred), zero_division=1
    )
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
