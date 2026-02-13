import pandas as pd
import matplotlib.pyplot as plt

# Sample accuracy scores 
models = ['MobileNet', 'ResNet_50', 'Self_built_CNN']
accuracy_scores = [0.85,  0.80, 0.67]  

# Create a DataFrame for the accuracy scores
d_frame = pd.DataFrame({'Model': models, 'Accuracy': accuracy_scores})

# Display the accuracy scores as a table
print(d_frame)

# Create a bar graph for visual representation
plt.figure(figsize=(8, 6))
plt.bar(d_frame['Model'], d_frame['Accuracy'], color=['green', 'orange', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison the Accuracy of CNN Models')
plt.ylim(0, 1)  
plt.xticks(rotation=15)  # Rotate x-axis labels for readability
plt.tight_layout()

plt.show()






models = ['MobileNet', 'ResNet_50', 'Self_built_CNN']
precision = [0.84, 0.77, 0.67]  

# Create a DataFrame for the precision scores
d_frame = pd.DataFrame({'Model': models, 'Precision': precision})

# Display the precision scores as a table
print(d_frame)

# Create a bar graph for visual representation
plt.figure(figsize=(8, 6))
plt.bar(d_frame['Model'], d_frame['Precision'], color=['green', 'orange', 'red'])
plt.xlabel('Model')
plt.ylabel('Precision')
plt.title('Comparison the Precision of CNN Models')
plt.ylim(0, 1)  
plt.xticks(rotation=15)  
plt.tight_layout()

plt.show()





models = ['MobileNet', 'ResNet_50', 'Self_built_CNN']
Recall = [0.93, 0.97, 1.00]  

# Create a DataFrame for the recall scores
d_frame = pd.DataFrame({'Model': models, 'Recall': Recall})

# Display the recall scores as a table
print(d_frame)

# Create a bar graph for visual representation
plt.figure(figsize=(8, 6))
plt.bar(d_frame['Model'], d_frame['Recall'], color=['green', 'orange', 'red'])
plt.xlabel('Model')
plt.ylabel('Precision')
plt.title('Comparison the Recall of CNN Models')
plt.ylim(0, 1)  
plt.xticks(rotation=15)  
plt.tight_layout()

# Show the bar graph
plt.show()





models = ['MobileNet', 'ResNet_50', 'Self_built_CNN']
f1score = [0.88, 0.86, 0.81]  

# Create a DataFrame for the f1 scores
d_frame = pd.DataFrame({'Model': models, 'f1score': f1score})

# Display the f1 scores as a table
print(d_frame)

# Create a bar graph for visual representation
plt.figure(figsize=(8, 6))
plt.bar(d_frame['Model'], d_frame['f1score'], color=['green', 'orange', 'red'])
plt.xlabel('Model')
plt.ylabel('Precision')
plt.title('Comparison the F1_score of CNN Models')
plt.ylim(0, 1)  
plt.xticks(rotation=15)  # Rotate x-axis labels for readability
plt.tight_layout()

plt.show()
