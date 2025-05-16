import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as job

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_hep = pd.read_csv('HepatitisCdata.csv')

data_h = data_hep.dropna()
data_h.drop(data_h[data_h['Category'] == '0s=suspect Blood Donor'].index, inplace=True)

filtered_df = data_h[data_h['Category'].isin(['1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'])]
healthy_samples = data_h[data_h['Category'] == '0=Blood Donor'].sample(n=56, random_state=42)
data = pd.concat([filtered_df, healthy_samples])
data = data.reset_index(drop=True)

data = data.drop(columns='Unnamed: 0')
data['Category'] = data['Category'].map({'0=Blood Donor': 1, '1=Hepatitis': 2, '2=Fibrosis': 3, '3=Cirrhosis': 4})
data['Sex'] = data['Sex'].map({'m': 1, 'f': 0})

X = data.drop(columns=['Category'])
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Обучаем scaler на тренировочных данных
X_test_scaled = scaler.transform(X_test) 
job.dump(scaler, 'robust_scaler.pkl')


#SVM
modelSVC = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'degree': [2, 3, 4]  # Только для полиномиального ядра
}

# Поиск по сетке
grid_searchSVC = GridSearchCV(modelSVC, param_grid, cv=5, scoring='accuracy')
grid_searchSVC.fit(X_train_scaled, y_train)
print(f"BEST: {grid_searchSVC.best_params_}")
best_model = grid_searchSVC.best_estimator_
y_predSVC_grid = best_model.predict(X_test_scaled)
print("SVC-Accuracy:", accuracy_score(y_test, y_predSVC_grid))
print("Classification Report:\n", classification_report(y_test, y_predSVC_grid))
sns.heatmap(confusion_matrix(y_test, y_predSVC_grid), annot=True, fmt='d', cmap='Blues', xticklabels=['1', '2', '3', '4'], yticklabels=['1', '2', '3', '4'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVM-GRID')
plt.show()

from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Точность на кросс-валидации: {scores}")
print(f"Средняя точность: {scores.mean()}")

# Дамп модели
job.dump(best_model, 'svm_model.pkl')