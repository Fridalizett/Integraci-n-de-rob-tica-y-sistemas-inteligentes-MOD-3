import numpy as np
import matplotlib.pyplot as plt

def calcular_velocidad_angular(posicion_final_ideal, posicion_final_real, tiempo):
    # Calcula la diferencia de posición angular
    diferencia_posicion_angular = posicion_final_real - posicion_final_ideal

    # Calcula la velocidad angular como la diferencia de posición angular dividida por el tiempo
    velocidad_angular = diferencia_posicion_angular / tiempo
    
    return velocidad_angular

def calcular_error(kr, velocidad_angular_ideal, velocidad_angular_real):
    # Calcula la velocidad angular de cada rueda
    velocidad_angular_l = kr * velocidad_angular_ideal
    velocidad_angular_r = kr * velocidad_angular_ideal

    # Calcula el error cuadrático medio
    error = np.mean((velocidad_angular_l + velocidad_angular_r - velocidad_angular_real) ** 2)
    
    return error

def calcular_gradiente(kr, velocidad_angular_ideal, velocidad_angular_real, epsilon=1e-5):
    # Calcula el gradiente utilizando la definición de derivada
    grad = (calcular_error(kr + epsilon, velocidad_angular_ideal, velocidad_angular_real) - calcular_error(kr, velocidad_angular_ideal, velocidad_angular_real)) / epsilon
    
    return grad

def descenso_gradiente_kr(velocidad_angular_ideal, velocidad_angular_real, lr=0.01, num_iter=100):
    # Inicializa kr
    kr = 1.0

    # Listas para almacenar el historial de valores de kr y error
    kr_history = []
    error_history = []

    # Descenso de gradiente
    for i in range(num_iter):
        # Calcula el error y el gradiente
        error = calcular_error(kr, velocidad_angular_ideal, velocidad_angular_real)
        grad = calcular_gradiente(kr, velocidad_angular_ideal, velocidad_angular_real)

        # Actualiza kr usando el gradiente descendente
        kr -= lr * grad

        # Almacena el historial
        kr_history.append(kr)
        error_history.append(error)

    return kr, kr_history, error_history

# Datos de ejemplo
posicion_final_ideal = 2.312493295# Posición final ideal en radianes
posicion_final_real = np.array([2.167896476])  # Posición final real en radianes (para múltiples experimentos)
tiempo = 3.0  # Tiempo en segundos

# Calcula la velocidad angular promedio a partir de las posiciones finales
velocidad_angular_ideal = calcular_velocidad_angular(posicion_final_ideal, posicion_final_real, tiempo)

# Velocidad angular real (asumiendo que es la misma que la ideal)
velocidad_angular_real = velocidad_angular_ideal

# Descenso de gradiente para encontrar kr
kr_optimo, kr_history, error_history = descenso_gradiente_kr(velocidad_angular_ideal, velocidad_angular_real)

print("kr óptimo encontrado:", kr_optimo)

# Gráfico de la convergencia del error
plt.plot(error_history)
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.title('Convergencia del Error')
plt.show()


