from scipy.stats import f_oneway
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#from mobileNet2 import true_labels, prediction
#from res_net50_cnn2 import true_labels2, predictions2
from build_test_cnn import test_labels, predictions


print(predictions)
y_true = test_labels
pred_probs = np.array(predictions)







fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = pred_probs.shape[1]  # Number of classes
print("Number of classes:", n_classes)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true, pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



# Plot the ROC curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-Class')
plt.legend(loc="lower right")
plt.show()

   

'''fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)

roc_auc = auc(fpr, tpr)
print("ROC-AUC Score:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
'''




