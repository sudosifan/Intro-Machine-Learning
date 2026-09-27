import numpy as np
def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix using numpy only.
    Args:
        y_true: array-like of shape (n_samples,)
        y_pred: array-like of shape (n_samples,)
    Returns:
        cm: ndarray of shape (num_classes, num_classes)
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If infer num_classes from the data, use the assumption that classes are labeled from 0 to num_classes-1
    all_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = int(np.max(all_classes)) + 1

    # Initialize the confusion matrix with zeros
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Fill the confusion matrix by counting occurrences
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    return cm

