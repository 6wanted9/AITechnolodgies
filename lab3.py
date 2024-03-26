import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.hidden_bias = np.random.randn(hidden_size)
        self.output_weights = np.random.randn(hidden_size, output_size)
        self.output_bias = np.random.randn(output_size)

    def activate(self, x):
        hidden_layer = np.tanh(np.dot(x, self.hidden_weights) + self.hidden_bias)
        output_layer = np.tanh(np.dot(hidden_layer, self.output_weights) + self.output_bias)
        return output_layer

    def train(self, epochs, learning_rate, training_values, expected_values):
        for epoch in range(epochs):
            # Проходимо по всім прикладам у навчальному датасеті
            for training, expected in zip(training_values, expected_values):
                # Здійснюємо передачу сигналу через нейрони
                hidden_layer_input = np.dot(training, self.hidden_weights) + self.hidden_bias
                hidden_layer_output = np.tanh(hidden_layer_input)
                output_layer_input = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
                output_layer_output = np.tanh(output_layer_input)

                # Обчислюємо помилки
                output_error = expected - output_layer_output
                output_delta = output_error * (1 - np.tanh(output_layer_input) ** 2)
                hidden_error = np.dot(output_delta, self.output_weights.T)
                hidden_delta = hidden_error * (1 - np.tanh(hidden_layer_input) ** 2)

                # Оновлюємо ваги та зсуви
                self.output_weights += learning_rate * np.outer(hidden_layer_output, output_delta)
                self.output_bias += learning_rate * output_delta
                self.hidden_weights += learning_rate * np.outer(training, hidden_delta)
                self.hidden_bias += learning_rate * hidden_delta


# Для демонстрації використаємо датасет із зображеннями цифр
digits = load_digits()
X, y = digits.data, digits.target

# Нормалізація даних
X /= np.max(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація перцептрону та тренування
mlp = MultiLayerPerceptron(input_size=X_train.shape[1], hidden_size=64, output_size=len(np.unique(y_train)))
mlp.train(epochs=1000, learning_rate=0.01, training_values=X_train, expected_values=np.eye(len(np.unique(y_train)))[y_train])

# Перевіряємо точність перцептрону
correct_predictions = 0
for x, y in zip(X_test, y_test):
    predicted = np.argmax(mlp.activate(x))
    if predicted == y:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test) * 100
print(f"Точність перцептрону: {round(accuracy, 2)}%")

# Візуалізація результатів
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Predicted: {np.argmax(mlp.activate(X_test[i]))}, Actual: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
