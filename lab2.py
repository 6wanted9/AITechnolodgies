import numpy as np


class ArtificialNeuron:
    def __init__(self):
        # Складність нейрону = 5
        self.weights = np.random.randn(5)
        self.bias = np.random.randn()

    def activate(self, x):
        return np.tanh(np.dot(x, self.weights) + self.bias)

    def train(self, epochs, learning_rate, training_values, expected_values):
        for epoch in range(epochs):
            # Проходимо по всім прикладам у навчальному датасеті
            for training, expected in zip(training_values, expected_values):
                # Здійснюємо передачу сигналу через нейрон
                output = self.activate(training)
                # Обчислюємо помилку
                error = expected - output
                # Оновлюємо ваги та зсув
                self.weights += learning_rate * error * training
                self.bias += learning_rate * error


# Для демонстрації використаємо простий датасет операції "XOR"
training_values = np.array([
    [1, 0, 0, 0, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]])
expected_values = np.array([0, 0, 1, 0, 1])

neuron = ArtificialNeuron()
neuron.train(1000, 0.1, training_values, expected_values)

# Перевіряємо точність нейрону
correctPredictions = 0
for value, expected in zip(training_values, expected_values):
    predicted = neuron.activate(value)
    roundedPredicted = round(predicted)
    print("Надане значення:", value, ", Передбачення: ", roundedPredicted)
    if roundedPredicted == expected:
        correctPredictions += 1

accuracy = correctPredictions / len(training_values) * 100
print("Точність нейрону:", accuracy)
