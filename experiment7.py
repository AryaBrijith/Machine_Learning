#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report

# Load winequality dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";")

# Split data into training and testing sets
X = df.drop(columns=['quality'])
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Print classification report for decision tree model
y_pred = dt.predict(X_test)
print("Decision Tree")
print(classification_report(y_test, y_pred))

# Train Bagging classifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)

# Print classification report for Bagging classifier
y_pred = bagging.predict(X_test)
print("Bagging")

print(classification_report(y_test, y_pred))

# Train Pasting classifier
pasting = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, bootstrap=False, random_state=42)
pasting.fit(X_train, y_train)

# Print classification report for Pasting classifier
y_pred = pasting.predict(X_test)
print("Pasting")
print(classification_report(y_test, y_pred))

# Evaluate Out-of-Bag score for Bagging classifier
bagging_oob = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, oob_score=True, random_state=42)
bagging_oob.fit(X_train, y_train)
oob_score = bagging_oob.oob_score_
print("Out-of-Bag Score")
print("OOB Score:", oob_score)


# In[3]:




import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()


# Plot histogram of predicted quality values for Bagging classifier
y_pred = bagging.predict(X_test)
plt.hist(y_pred, bins=range(3, 9), align='left')
plt.xticks(range(3, 9))
plt.xlabel("Predicted Quality")
plt.ylabel("Frequency")
plt.show()


# In[ ]:




