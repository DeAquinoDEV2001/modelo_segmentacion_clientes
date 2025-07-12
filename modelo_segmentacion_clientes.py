# modelo_segmentacion_clientes.py
# Segmentación, clasificación y regresión para campañas de marketing personalizadas

# ===============================
# 1. Importación de bibliotecas
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error

# ===============================
# 2. Simulación de datos de clientes
# ===============================
np.random.seed(42)
data = pd.DataFrame({
    'Edad': np.random.randint(18, 70, 200),
    'Ingresos': np.random.randint(20000, 150000, 200),
    'PuntuacionGasto': np.random.randint(1, 100, 200)
})

print("\nPrimeras filas del dataset simulado:")
print(data.head())

# ===============================
# 3. Escalamiento de variables
# ===============================
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(data)

# ===============================
# 4. Método del Codo para determinar K
# ===============================
inercia = []
K = range(1, 10)
for k in K:
    modelo = KMeans(n_clusters=k, n_init=10, random_state=42)
    modelo.fit(datos_escalados)
    inercia.append(modelo.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, inercia, marker='o')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.grid(True)
plt.tight_layout()
plt.savefig('metodo_del_codo.png')  # Guarda la gráfica como imagen
plt.show()

# ===============================
# 5. Aplicación de K-Means (suponiendo K=4)
# ===============================
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(datos_escalados)

print("\nDatos con Clúster asignado:")
print(data.head())

# ===============================
# 6. Visualización de los Clústeres
# ===============================
sns.pairplot(data, hue='Cluster', palette='Set2')
plt.suptitle("Visualización de Clústeres", y=1.02)
plt.savefig('visualizacion_clusters.png')
plt.show()

# ===============================
# 7. Modelo de Clasificación
# ===============================
X = data[['Edad', 'Ingresos', 'PuntuacionGasto']]
y = data['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_clasificacion = LogisticRegression(max_iter=1000)
modelo_clasificacion.fit(X_train, y_train)

y_pred = modelo_clasificacion.predict(X_test)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# ===============================
# 8. Modelo de Regresión
# ===============================
X_reg = data[['Edad', 'Ingresos']]
y_reg = data['PuntuacionGasto']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train_reg, y_train_reg)

y_pred_reg = modelo_regresion.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"\nError cuadrático medio de la regresión: {mse:.2f}")
