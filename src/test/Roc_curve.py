import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Sample true labels and predicted probabilities (replace with your data)
true_labels_model1 = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
predicted_probs_model1 = np.array([0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.4, 0.85, 0.25, 0.95])

true_labels_model2 = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
predicted_probs_model2 = np.array([0.2, 0.7, 0.6, 0.3, 0.85, 0.4, 0.35, 0.9, 0.3, 0.88])

# Calculate ROC curves and AUCs for each model
fpr_model1, tpr_model1, thresholds_model1 = roc_curve(true_labels_model1, predicted_probs_model1)
roc_auc_model1 = auc(fpr_model1, tpr_model1)

fpr_model2, tpr_model2, thresholds_model2 = roc_curve(true_labels_model2, predicted_probs_model2)
roc_auc_model2 = auc(fpr_model2, tpr_model2)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_model1, tpr_model1, color='darkorange', lw=2, label='Model 1 (AUC = {:.2f})'.format(roc_auc_model1))
plt.plot(fpr_model2, tpr_model2, color='green', lw=2, label='Model 2 (AUC = {:.2f})'.format(roc_auc_model2))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend
