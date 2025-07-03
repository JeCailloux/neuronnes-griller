import numpy as np
import matplotlib.pyplot as plt
from activation import ActivationFunction


def plot_activation(name, alpha=0.01):
    z = np.linspace(-10, 10, 400)
    act = ActivationFunction(name, alpha)
    g = act.apply(z)
    dg = act.derivative(z)

    plt.figure()
    plt.plot(z, g, label=name)
    plt.plot(z, dg, label=f"d{name}/dz")
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}.png")

if __name__ == "__main__":
    for name in ["heaviside", "sigmoid", "tanh", "relu", "leaky_relu"]:
        plot_activation(name)
