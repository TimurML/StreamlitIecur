import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib as job

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

data_c = pd.read_csv('processed_data_cirrhosis_new.csv')


X = data_c.drop(columns=['Stage'])
y = data_c['Stage']

y = data_c['Stage'].astype(int) - 2

X = X.drop(columns=['Alk_Phos', 'Sex', 'SGOT', 'Unnamed: 0'])

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

job.dump(scaler, 'robust_scaler_cirr.pkl')

# SMOTE
from imblearn.over_sampling import SMOTENC
categorical_indices = [1, 2, 3, 4, 5]
smote = SMOTENC(random_state=42, categorical_features = categorical_indices)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

modelMLP = MLPClassifier(
    hidden_layer_sizes=(90,),
    max_iter=450,
    activation='logistic',
    random_state=42 )

model = modelMLP.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test_scaled)
y_pred_train = model.predict(X_resampled)

print("Accuracy train:", accuracy_score(y_resampled, y_pred_train), "\nAccuracy test:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=['1', '2', '3', '4'], yticklabels=['1', '2', '3', '4'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('GRID')
plt.show()

job.dump(model, 'mlp_model_cirr.pkl')