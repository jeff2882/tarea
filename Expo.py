# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# # Datos de temperatura y presión
# X = np.array([0, 10, 20, 30, 40, 50, 60]).reshape(-1, 1)
# y = np.array([101.3, 97.4, 93.5, 89.6, 85.7, 81.8, 77.9])

# # Ajuste del modelo de regresión polinómica de grado 2
# polynomial_features = PolynomialFeatures(degree=2)
# X_poly = polynomial_features.fit_transform(X)

# model = LinearRegression()
# model.fit(X_poly, y)

# # Predicciones con el modelo ajustado
# y_pred = model.predict(X_poly)

# # Graficar los datos reales y la regresión polinómica
# plt.scatter(X, y, color='blue', label='Datos reales')
# plt.plot(X, y_pred, color='red', label='Regresión polinómica (grado 2)')
# plt.title('Regresión Polinómica: Temperatura vs Presión')
# plt.xlabel('Temperatura (°C)')
# plt.ylabel('Presión (kPa)')
# plt.legend()
# plt.grid(True)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Datos proporcionados
# meses = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# gasto_publicidad = np.array([2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
# ventas = np.array([3.0, 4.5, 5.0, 7.0, 8.0, 8.5, 9.5, 11.0, 12.0, 14.0])

# # Calcular las medias de X y Y
# mean_X = np.mean(gasto_publicidad)
# mean_Y = np.mean(ventas)

# # Calcular las desviaciones y productos
# dev_X = gasto_publicidad - mean_X
# dev_Y = ventas - mean_Y
# dev_product = dev_X * dev_Y

# # Calcular la pendiente (β1) y la intersección (β0)
# beta1 = np.sum(dev_product) / np.sum(dev_X ** 2)
# beta0 = mean_Y - beta1 * mean_X

# Construir el modelo de regresión lineal
# ventas_pred = beta0 + beta1 * gasto_publicidad

# # Graficar los datos y la línea de regresión
# plt.figure(figsize=(10, 6))
# plt.scatter(gasto_publicidad, ventas, color='blue', label='Datos reales')
# plt.plot(gasto_publicidad, ventas_pred, color='red', label='Línea de regresión')
# plt.title('Regresión Lineal: Gasto en Publicidad vs Ventas')
# plt.xlabel('Gasto en Publicidad (miles de dólares)')
# plt.ylabel('Ventas (miles de dólares)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Mostrar coeficientes β0 y β1
# print(f"Coeficiente β0 (intersección): {beta0}")
# print(f"Coeficiente β1 (pendiente): {beta1}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Datos
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

# Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X, Y)
Y_pred_linear = linear_model.predict(X)

# Regresión Polinómica (grado 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, Y)
Y_pred_poly = poly_model.predict(X_poly)

# Gráfica
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Datos reales')
plt.plot(X, Y_pred_linear, color='red', label='Regresión Lineal')
plt.plot(X, Y_pred_poly, color='green', label='Regresión Polinómica (grado 2)')
plt.title('Regresión Lineal vs Regresión Polinómica')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

