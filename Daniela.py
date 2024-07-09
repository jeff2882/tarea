#DANIELA ARAUZ
#JEFFERSON ZAMBRANO
#LESLIE PAZMINO
#notas = [85, 90, 78, 92, 88]
#promedio_notas = sum(notas) / len(notas)
#print(f"el promedio es: {promedio_notas}")


#import pandas  as pd 
# ventas = [120, 135, 150, 145, 160, 155, 170, 175, 180, 190, 195, 200]
# df = pd.DataFrame(ventas, columns=["Ventas"])
# df['SMA_3'] = df['Ventas'].rolling(window=3).mean()
# print (df)


# import pandas  as pd 
# ventas = [100, 102, 101, 105, 110, 108, 107, 111, 115, 118, 117, 120]
# df = pd.DataFrame(ventas, columns=["Ventas"])
# df['SMA_3'] = df['Ventas'].rolling(window=5).mean()
# print (df)


#Promedio movil exponencial 
# import pandas  as pd 
# import matplotlib.pyplot as plt 
# temperaturas = [100, 102, 101, 105, 110, 108, 107, 111, 115, 118, 117, 120]
# df = pd.DataFrame(temperaturas, columns=["temperaturas"])
# df['SMA_4'] = df['temperaturas'].ewm(spam=4, adjust=False).mean()
# print (df)

# plt.figure(figsize=(10, 6))
# plt.plot(temperaturas.index, temperaturas['temperaturas'], label='temperaturas')
# plt.plot(temperaturas.index, temperaturas['EMA 4'], label='EMA(4 dias)',linestyle='--',marker='x')
# plt.xlabel('Dias')
# plt.xlabel('temperaturas')
# plt.ylabel('Promedio movil exponexial de temperaturas diarias')
# plt.legend()
# plt.grid()
# plt.show()


# REGRESION LINEAL SIMPLE 
# import numpy as np
# import matplotlib.pyplot as plt 
# from sklearn.linear_model import LinearRegression

# #Generar dato
# np.random.seed(0)
# X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]).reshape(-1, 1)
# y = np.array([45000, 50000, 60000, 650000, 70000, 80000, 85000, 90000, 950000, 100000])
# X = 2 * np.random.rand(100, 1) 
# y = 4 + 3 * X + np.random.randn(100, 1)

# #Ajustar el modelo
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)

# #Predicciones
# y_pred = lin_reg.predict(X)

# # Graficar
# plt.scatter(X, y, color ='blue', label ='Datos')
# plt.plot(X, y_pred, color='red', label = "Regresión Lineal")
# plt.xlabel("Cantidad Vendida")
# plt.ylabel("Precio")
# plt.title("Regresión Lineal Simple")
# plt.legend()
# plt.show()


# # REGRESION LINEAL MULTIPLE
import simpy
import random
import matplotlib.pyplot as plt

# Parámetros
RANDOM_SEED = 42
NUM_CLIENTES = 50 # Número de clientes
TIEMPO_LLEGADAS = 7 # Tiempo medio entre llegadas de clientes
TIEMPO_SERVICIO = 4 #Tienen sadio de servicio

# Almacenamiento de tiempos dk chara
wait_times = []

def customer(env, name, server):
#Un cliente llega, espera su turno y luego es atendido,
    arrival_time = env.now
    with server.request() as request:
         yield request
         wait_time = env.now - arrival_time
         wait_times.append(wait_time)
         service_time = random.expovariate (1.0 / TIEMPO_SERVICIO)
         yield env.timeout(service_time)

def setup(env, num_customers, interarrival_time, service_time):
    """Crea un servidor y genera clientes."""
    server = simpy.Resource(env, capacity=2)
    for i in range(num_customers):
        yield env.timeout(random.expovariate (1.0 / TIEMPO_LLEGADAS))
        env.process(customer(env, f'Cliente (i+1)', server))

#Configuración de la simulación
random.seed(RANDOM_SEED)
env = simpy.Environment()
env.process(setup(env, NUM_CLIENTES, TIEMPO_LLEGADAS, TIEMPO_SERVICIO))
env.run()

#Gráfico
plt.plot(wait_times, 'bo')
plt.xlabel('Número de Cliente')
plt.ylabel('Tiempo de Espera (min)')
plt.title('Tiempo de Espera de Clientes en la Tienda')
plt.show()
