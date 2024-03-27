import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))

    def train(self, patterns):
        num_patterns = len(patterns)
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern) / num_patterns
        np.fill_diagonal(self.weights, 0)

    def update(self, pattern, max_iter=100):
        for _ in range(max_iter):
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):
                break
            pattern = new_pattern
        return pattern


# Для демонстрації використаємо датасет із зображеннями цифр
digits = load_digits()
numOfSamples = 3  # К-ть екземплярів, які ми обираємо з датасету
data = digits.data[:numOfSamples]

# Нормалізація даних
data[data < 8] = -1
data[data >= 8] = 1

# Ініціалізація мережі та тренування
hopfield_net = HopfieldNetwork(data.shape[1])
hopfield_net.train(data)

# Спотворення даних та тестування мережі
corrupted_data = np.copy(data)
for i in range(len(corrupted_data)):
    corrupted_data[i][15:25] *= -1  # Спотворюємо кожен екземпляр

recovered_data = [hopfield_net.update(pattern) for pattern in corrupted_data]

# Візуалізація результатів
plt.figure(figsize=(numOfSamples, 4))
for i in range(numOfSamples):
    plt.subplot(3, numOfSamples, i + 1)
    plt.imshow(data[i].reshape(8, 8), cmap='binary')
    plt.axis('off')
    plt.subplot(3, numOfSamples, i + numOfSamples + 1)
    plt.imshow(corrupted_data[i].reshape(8, 8), cmap='binary')
    plt.axis('off')
    plt.subplot(3, numOfSamples, i + numOfSamples * 2 + 1)
    plt.imshow(recovered_data[i].reshape(8, 8), cmap='binary')
    plt.axis('off')
plt.suptitle('Original (top) vs. Corrupted (middle) vs. Recovered (bottom)')
plt.show()
