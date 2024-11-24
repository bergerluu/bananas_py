import pandas as pd

tabela = pd.read_excel("teste.xlsx")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Filtrar os dados para as categorias "Premium" e "Processing"
filtered_data = tabela[tabela['quality_category'].isin(['Premium', 'Processing'])]
filtered_data = filtered_data.sample(n=100, random_state=42)
length_cm = np.array(filtered_data['length_cm'])
weight_g = np.array(filtered_data['weight_g'])

# Mapear os rótulos: Premium = 1, Processing = -1
labels = np.where(filtered_data['quality_category'] == 'Premium', 1, 0)

# Normalizar os dados
X = np.vstack((length_cm / np.max(length_cm), weight_g / np.max(weight_g))).T
Y = labels.reshape(-1, 1)

# Arquitetura da rede neural
input_neurons = 2   # Número de entradas (length e weight)
hidden_neurons_1 = 4
hidden_neurons_2 = 4
hidden_neurons_3 = 4
output_neurons = 1

# Inicialização dos pesos e biases
np.random.seed(42)
Weight_1 = np.random.randn(input_neurons, hidden_neurons_1)
bias_1 = np.random.randn(1, hidden_neurons_1)
Weight_2 = np.random.randn(hidden_neurons_1, hidden_neurons_2)
bias_2 = np.random.randn(1, hidden_neurons_2)
Weight_3 = np.random.randn(hidden_neurons_2, hidden_neurons_3)
bias_3 = np.random.randn(1, hidden_neurons_3)
Weight_4 = np.random.randn(hidden_neurons_3, output_neurons)
bias_4 = np.random.randn(1, output_neurons)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Hiperparâmetros
learning_rate = 0.25
num_epochs = 30555

# Treinamento da rede neural
for epoch in range(num_epochs):
    # Forward pass
    Z1 = np.dot(X, Weight_1) + bias_1
    A1 = relu(Z1)
    Z2 = np.dot(A1, Weight_2) + bias_2
    A2 = relu(Z2)
    Z3 = np.dot(A2, Weight_3) + bias_3
    A3 = relu(Z3)
    Z4 = np.dot(A3, Weight_4) + bias_4
    A4 = sigmoid(Z4)

    # Z1 = np.dot(X, Weight_1) + bias_1
    # A1 = sigmoid(Z1)
    # Z2 = np.dot(A1, Weight_2) + bias_2
    # A2 = sigmoid(Z2)
    # Z3 = np.dot(A2, Weight_3) + bias_3
    # A3 = sigmoid(Z3)
    # Z4 = np.dot(A3, Weight_4) + bias_4
    # A4 = sigmoid(Z4)

    # Backpropagation
    error = Y - A4
    dA4 = error * sigmoid_derivative(A4)
    dW4 = np.dot(A3.T, dA4)
    db4 = np.sum(dA4, axis=0, keepdims=True)

    dA3 = np.dot(dA4, Weight_4.T) * relu_derivative(A3)
    dW3 = np.dot(A2.T, dA3)
    db3 = np.sum(dA3, axis=0, keepdims=True)

    dA2 = np.dot(dA3, Weight_3.T) * relu_derivative(A2)
    dW2 = np.dot(A1.T, dA2)
    db2 = np.sum(dA2, axis=0, keepdims=True)

    dA1 = np.dot(dA2, Weight_2.T) * relu_derivative(A1)
    dW1 = np.dot(X.T, dA1)
    db1 = np.sum(dA1, axis=0, keepdims=True)

    # error = Y - A4
    # dA4 = error * sigmoid_derivative(A4)
    # dW4 = np.dot(A3.T, dA4)
    # db4 = np.sum(dA4, axis=0, keepdims=True)

    # dA3 = np.dot(dA4, Weight_4.T) * sigmoid_derivative(A3)
    # dW3 = np.dot(A2.T, dA3)
    # db3 = np.sum(dA3, axis=0, keepdims=True)

    # dA2 = np.dot(dA3, Weight_3.T) * sigmoid_derivative(A2)
    # dW2 = np.dot(A1.T, dA2)
    # db2 = np.sum(dA2, axis=0, keepdims=True)

    # dA1 = np.dot(dA2, Weight_2.T) * sigmoid_derivative(A1)
    # dW1 = np.dot(X.T, dA1)
    # db1 = np.sum(dA1, axis=0, keepdims=True)

    # Atualização dos pesos
    Weight_4 += learning_rate * dW4
    bias_4 += learning_rate * db4
    Weight_3 += learning_rate * dW3
    bias_3 += learning_rate * db3
    Weight_2 += learning_rate * dW2
    bias_2 += learning_rate * db2
    Weight_1 += learning_rate * dW1
    bias_1 += learning_rate * db1

# Plotando os dados
plt.title("Gráfico de Comprimento vs Peso")
plt.xlabel("Comprimento (cm)")
plt.ylabel("Peso (cm equivalente)")
plt.scatter(length_cm, weight_g / 100, c=A4.flatten(), cmap='bwr', alpha=0.7)
plt.colorbar(label="Saída da Rede Neural (Probabilidade)")
plt.grid()
plt.show()

# Erro final
print("Erro final: ", np.mean(np.abs(error)))