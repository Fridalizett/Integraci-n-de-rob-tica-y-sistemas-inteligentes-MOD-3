import numpy as np
import matplotlib.pyplot as plt

def calcular_velocidad(posicion_final_ideal, posicion_final_real, tiempo):
    # Calcula la diferencia de posición
    diferencia_posicion = posicion_final_real - posicion_final_ideal

    # Calcula la velocidad como la diferencia de posición dividida por el tiempo
    velocidad = diferencia_posicion / tiempo
    
    return velocidad

def calcular_error(kl, velocidad_ideal, velocidad_real):
    # Calcula la velocidad de cada rueda
    velocidad_l = kl * velocidad_ideal
    velocidad_r = kl * velocidad_ideal

    # Calcula el error cuadrático
    error = np.mean((velocidad_l + velocidad_r - velocidad_real) ** 2)
    
    return error

def calcular_gradiente(kl, velocidad_ideal, velocidad_real, epsilon=1e-5):
    # Calcula el gradiente utilizando la definición de derivada
    grad = (calcular_error(kl + epsilon, velocidad_ideal, velocidad_real) - calcular_error(kl, velocidad_ideal, velocidad_real)) / epsilon
    
    return grad

def descenso_gradiente_kl(velocidad_ideal, velocidad_real, lr=0.01, num_iter=100):
    # Inicializa kl
    kl = 1.0

    # Listas para almacenar el historial de valores de kl y error
    kl_history = []
    error_history = []

    # Descenso de gradiente
    for i in range(num_iter):
        # Calcula el error y el gradiente
        error = calcular_error(kl, velocidad_ideal, velocidad_real)
        grad = calcular_gradiente(kl, velocidad_ideal, velocidad_real)

        # Actualiza kl usando el gradiente descendente
        kl -= lr * grad

        # Almacena el historial
        kl_history.append(kl)
        error_history.append(error)

    return kl, kl_history, error_history

# Datos de ejemplo
posicion_final_ideal = 2.445330811  # Posición final ideal
posicion_final_real = np.array([2.34674])  # Posición final real (para múltiples experimentos)
tiempo = 5.0  # Tiempo en segundos

# Calcula la velocidad a partir de las posiciones finales
velocidad_ideal = calcular_velocidad(posicion_final_ideal, posicion_final_real, tiempo)

# Velocidad real (asumiendo que es la misma que la ideal)
velocidad_real = velocidad_ideal

# Descenso de gradiente para encontrar kl
kl_optimo, kl_history, error_history = descenso_gradiente_kl(velocidad_ideal, velocidad_real)

print("kl óptimo encontrado:", kl_optimo)

# Gráfico de la convergencia del error
plt.plot(error_history)
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.title('Convergencia del Error')
plt.show()
