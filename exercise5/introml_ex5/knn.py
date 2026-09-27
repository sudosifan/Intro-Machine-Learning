
import numpy as np

from visualization import plot_knn_neighbors

class KNNClassifier:
    def __init__(self, n_neighbors=3, metric="euclidean", plot_neighbors=False, image_shape=None):
        # Number of nearest neighbors to use for prediction
        self.n_neighbors = n_neighbors
        # Distance metric: 'euclidean' or 'cosine'
        self.metric = metric
        self.plot_neighbors = plot_neighbors  # Flag to control neighbor visualization
        self.image_shape = image_shape  # Store image shape for reshaping

    def fit(self, X, y):
        # Store the training data and labels as numpy arrays
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        

    def predict(self, X):
        # Convert input data to numpy array
        X = np.array(X)
        predictions = []
        # Iterate over each test sample to predict its label
        for x in X:
            if self.metric == "euclidean":
                # Compute Euclidean distances from x to all training samples
                distances = []
                for train_sample in self.X_train:
                    distance = np.sqrt(np.sum((x - train_sample) ** 2))
                    distances.append(distance)
                distances = np.array(distances)

            elif self.metric == "cosine":
                # Compute the cosine similarity between x and each training sample:
                #   sim = (train · test) / (||train|| * ||test||)
                # Then convert similarity to distance: cosine_distance = 1 - cosine_similarity
                # Hint: Avoid division by zero while computing cosine similarity
                distances = []
                for train_sample in self.X_train:
                    dot_product = np.dot(x, train_sample)
                    norm_x = np.linalg.norm(x)
                    norm_train = np.linalg.norm(train_sample)

                    # Avoid division by zero
                    if norm_x == 0 or norm_train == 0:

                        cosine_distance = 1.0
                    else:
                        cosine_similarity = dot_product / (norm_x * norm_train)
                        cosine_distance = 1 - cosine_similarity
                    distances.append(cosine_distance)
                distances = np.array(distances)
                
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
            # Find the indices of the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]

            # Get the labels of the nearest neighbors
            nearest_labels = self.y_train[nearest_indices]
            
            # Find the most common label (majority vote)
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]

            # Append the predicted label
            predictions.append(predicted_label)

            # Optionally visualize the neighbors
            if self.plot_neighbors and self.image_shape is not None:
                test_image = x.reshape(self.image_shape)
                neighbor_images = [self.X_train[idx].reshape(self.image_shape) for idx in nearest_indices]
                neighbor_labels = [self.y_train[idx] for idx in nearest_indices]
                plot_knn_neighbors(
                    test_image=test_image,
                    neighbor_images=neighbor_images,
                    neighbor_labels=neighbor_labels,
                    test_label=None,
                    pred_label=predicted_label
                )
        # Return predictions as a numpy array
        return np.array(predictions)