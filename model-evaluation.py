from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from azureml.core import Model

def evaluate_model(y_true, y_pred, y_prob):
    """Evaluate the model using various metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def plot_roc_curve(y_true, y_prob):
    """Plot the ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def get_feature_importance(model, feature_names):
    """Get feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't provide feature importances")
        return None
    
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return feature_importance

def plot_feature_importance(feature_importance, top_n=10):
    """Plot top N feature importances."""
    top_features = feature_importance[:top_n]
    features, importances = zip(*top_features)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

