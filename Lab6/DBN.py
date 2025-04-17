import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

wine = load_wine()
X, y = wine['data'], wine['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rbm1 = BernoulliRBM(n_components=128, learning_rate=0.01, n_iter=20, verbose=1)
rbm2 = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=20, verbose=1)

logistic = LogisticRegression(max_iter=1000)

print("Pre-training RBM1...")
rbm1.fit(X_train_scaled)

X_train_rbm1 = rbm1.transform(X_train_scaled)

print("Pre-training RBM2...")
rbm2.fit(X_train_rbm1)

X_train_rbm2 = rbm2.transform(X_train_rbm1)

print("Fine-tuning with Logistic Regression...")
logistic.fit(X_train_rbm2, y_train)

X_test_rbm1 = rbm1.transform(X_test_scaled)
X_test_rbm2 = rbm2.transform(X_test_rbm1)

y_pred = logistic.predict(X_test_rbm2)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"DBN Classification Performance on the Wine Dataset:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
