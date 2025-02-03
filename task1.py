import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.random.randn(input_size + 1) * 0.01  # Including bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else -1

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        output = []
        for xi in X:
            pred = self.activation(np.dot(xi,self.weights))
            output.append(pred)
        return np.array(output)

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.activation(np.dot(xi, self.weights))
                error = target - output 
                self.weights += self.learning_rate * error * xi 

def generate_data(seed, samples, noise):
    """
    Generates two-class classification data with overlapping clusters.
    """
    np.random.seed(seed)
    X, y = make_blobs(n_samples=samples, centers=2, cluster_std=noise, random_state=seed)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for perceptron
    return X, y

def main():
    #19212579
    parser = argparse.ArgumentParser(description="Train a single-layer perceptron using NumPy.")
    parser.add_argument('--registration_number', type=int, required=True, help="Student's registration number (used as seed)")
    parser.add_argument('--samples', type=int, default=200, help="Number of data samples")
    parser.add_argument('--noise', type=float, default=1.5, help="Standard deviation of clusters")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Perceptron learning rate")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs")
    
    args = parser.parse_args()

    # Generate data
    X, y = generate_data(args.registration_number, args.samples, args.noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.registration_number)

    # Train the perceptron
    perceptron = Perceptron(input_size=2, learning_rate=args.learning_rate, epochs=args.epochs)
    perceptron.fit(X_train, y_train)

    # Evaluate the perceptron
    y_pred = perceptron.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Perceptron Training Completed.")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Visualizing decision boundary
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", alpha=0.6)
    
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid_points).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.title(f"Perceptron Decision Boundary (Accuracy: {accuracy:.4f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    main()