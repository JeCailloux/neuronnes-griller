import numpy as np

class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha

    def apply(self, z):
        z = np.array(z)
        if self.name == "heaviside":
            return np.where(z >= 0, 1.0, 0.0)
        elif self.name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        elif self.name == "tanh":
            return np.tanh(z)
        elif self.name == "relu":
            return np.maximum(0, z)
        elif self.name == "leaky_relu":
            return np.where(z >= 0, z, self.alpha * z)
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def derivative(self, z):
        z = np.array(z)
        if self.name == "heaviside":
            # Approximation: zero everywhere
            return np.zeros_like(z)
        elif self.name == "sigmoid":
            s = self.apply(z)
            return s * (1 - s)
        elif self.name == "tanh":
            t = np.tanh(z)
            return 1 - t**2
        elif self.name == "relu":
            return np.where(z > 0, 1.0, 0.0)
        elif self.name == "leaky_relu":
            return np.where(z > 0, 1.0, self.alpha)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.")
