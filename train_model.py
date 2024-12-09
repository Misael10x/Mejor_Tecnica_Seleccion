import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Cargar el dataset reducido
df = pd.read_csv('reducido.csv')

# Separar las características (X) y la variable objetivo (y)
X = df.drop('calss', axis=1)  # Reemplaza 'target' con el nombre de la columna objetivo
y = df['calss']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los hiperparámetros para el GridSearch
param_grid = [
    {'n_estimators': [10, 5, 10], 'max_leaf_nodes': [1, 2, 3]},
    {'bootstrap': [False], 'n_estimators': [10, 50]},
]

rnd_clf = RandomForestClassifier(n_jobs=-1, random_state=42)

# Configurar el GridSearch
grid_search = GridSearchCV(rnd_clf, param_grid, cv=5, scoring='f1_weighted', return_train_score=True)
grid_search.fit(X_train, y_train)

# Guardar el GridSearch y el mejor modelo
joblib.dump(grid_search, 'grid_search.pkl')
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')

# Generar un reporte con los resultados en el conjunto de prueba
y_pred = grid_search.best_estimator_.predict(X_test)
report = classification_report(y_test, y_pred)

# Guardar el reporte en un archivo de texto
with open('classification_report.txt', 'w') as f:
    f.write(report)

print("Modelo entrenado y resultados guardados.")
