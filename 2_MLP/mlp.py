"""
Multi-Layer Perceptron (MLP) - Implementação do zero com NumPy
===============================================================
Rede neural com múltiplas camadas e algoritmo de backpropagation.

Arquitetura:
    Entrada -> [Camadas Ocultas com ReLU] -> Saída (Sigmoid/Softmax)

Algoritmo de treinamento:
    1. Forward pass: calcula as ativações camada por camada
    2. Backward pass: calcula os gradientes usando a regra da cadeia
    3. Atualização dos pesos via Gradiente Descendente Estocástico (SGD)
"""

import numpy as np
import matplotlib.pyplot as plt


class MLP:
    """Rede Neural Multi-Camadas (Multi-Layer Perceptron)."""

    def __init__(self, camadas, taxa_aprendizado=0.01, n_epocas=1000):
        """
        Args:
            camadas (list): Lista com número de neurônios por camada.
                            Ex: [2, 4, 1] = 2 entradas, 4 ocultos, 1 saída
            taxa_aprendizado (float): Taxa de aprendizado
            n_epocas (int): Número de épocas de treinamento
        """
        self.camadas = camadas
        self.taxa_aprendizado = taxa_aprendizado
        self.n_epocas = n_epocas
        self.pesos = []
        self.biases = []
        self.historico_perda = []
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        """Inicializa pesos com método de He (adequado para ReLU)."""
        np.random.seed(42)
        for i in range(len(self.camadas) - 1):
            fator = np.sqrt(2.0 / self.camadas[i])
            W = np.random.randn(self.camadas[i], self.camadas[i + 1]) * fator
            b = np.zeros((1, self.camadas[i + 1]))
            self.pesos.append(W)
            self.biases.append(b)

    # Funções de ativação
    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivada(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivada(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def _forward(self, X):
        """
        Passagem direta (forward pass).
        Retorna listas com os valores pré-ativação (Z) e pós-ativação (A).
        """
        ativacoes = [X]
        pre_ativacoes = []

        for i in range(len(self.pesos) - 1):
            z = np.dot(ativacoes[-1], self.pesos[i]) + self.biases[i]
            pre_ativacoes.append(z)
            ativacoes.append(self._relu(z))

        # Última camada usa sigmoid (classificação binária)
        z = np.dot(ativacoes[-1], self.pesos[-1]) + self.biases[-1]
        pre_ativacoes.append(z)
        ativacoes.append(self._sigmoid(z))

        return ativacoes, pre_ativacoes

    def _backward(self, X, y, ativacoes, pre_ativacoes):
        """
        Retropropagação (backpropagation).
        Calcula os gradientes de pesos e biases.
        """
        n = X.shape[0]
        grad_pesos = [np.zeros_like(w) for w in self.pesos]
        grad_biases = [np.zeros_like(b) for b in self.biases]

        # Erro na camada de saída (cross-entropy + sigmoid)
        delta = ativacoes[-1] - y.reshape(-1, 1)

        for i in reversed(range(len(self.pesos))):
            grad_pesos[i] = np.dot(ativacoes[i].T, delta) / n
            grad_biases[i] = np.mean(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, self.pesos[i].T) * self._relu_derivada(pre_ativacoes[i - 1])

        return grad_pesos, grad_biases

    def fit(self, X, y):
        """Treina a rede neural."""
        for epoca in range(self.n_epocas):
            ativacoes, pre_ativacoes = self._forward(X)
            perda = self._binary_cross_entropy(y, ativacoes[-1])
            self.historico_perda.append(perda)

            grad_pesos, grad_biases = self._backward(X, y, ativacoes, pre_ativacoes)

            for i in range(len(self.pesos)):
                self.pesos[i] -= self.taxa_aprendizado * grad_pesos[i]
                self.biases[i] -= self.taxa_aprendizado * grad_biases[i]

            if (epoca + 1) % 100 == 0:
                print(f"Época {epoca + 1}/{self.n_epocas} - Perda: {perda:.4f}")

        return self

    def predict_proba(self, X):
        """Retorna probabilidades de saída."""
        ativacoes, _ = self._forward(X)
        return ativacoes[-1]

    def predict(self, X, limiar=0.5):
        """Retorna predições binárias."""
        return (self.predict_proba(X) >= limiar).astype(int).flatten()

    def acuracia(self, X, y):
        """Calcula a acurácia."""
        return np.mean(self.predict(X) == y)

    def _binary_cross_entropy(self, y_real, y_pred):
        """Função de perda: Entropia Cruzada Binária."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred.flatten(), epsilon, 1 - epsilon)
        return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))


def plot_resultados(X, y, modelo, titulo="MLP - Fronteira de Decisão"):
    """Plota a fronteira de decisão e o histórico de perda."""
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05),
                            np.arange(x2_min, x2_max, 0.05))
    Z = modelo.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="red", marker="o", label="Classe 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", marker="s", label="Classe 1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(titulo)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(modelo.historico_perda)
    plt.xlabel("Época")
    plt.ylabel("Perda (Cross-Entropy)")
    plt.title("Curva de Aprendizado")

    plt.tight_layout()
    plt.savefig("mlp_resultado.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'mlp_resultado.png'")


def exemplo_xor():
    """Demonstra o MLP resolvendo o problema XOR (não-linearmente separável)."""
    print("=" * 50)
    print("Exemplo: Problema XOR")
    print("=" * 50)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)

    # Arquitetura: 2 entradas -> 4 neurônios ocultos -> 1 saída
    modelo = MLP(camadas=[2, 4, 1], taxa_aprendizado=0.5, n_epocas=2000)
    modelo.fit(X, y)

    print(f"\nAcurácia: {modelo.acuracia(X, y) * 100:.1f}%")
    print("\nTabela Verdade XOR:")
    print(f"{'Entrada':^12} | {'Saída Real':^10} | {'Predição':^10} | {'Probabilidade':^14}")
    print("-" * 55)
    probs = modelo.predict_proba(X).flatten()
    for xi, yi, y_pred, prob in zip(X, y, modelo.predict(X), probs):
        print(f"{str(xi.astype(int)):^12} | {int(yi):^10} | {y_pred:^10} | {prob:^14.4f}")


def exemplo_classificacao_2d():
    """Demonstra o MLP em dados com fronteira não-linear."""
    print("\n" + "=" * 50)
    print("Exemplo: Classificação 2D Não-Linear (Círculos)")
    print("=" * 50)
    np.random.seed(42)
    n = 200
    angulos = np.random.uniform(0, 2 * np.pi, n)

    # Classe 0: círculo interno
    r0 = np.random.uniform(0, 1, n // 2)
    X0 = np.column_stack([r0 * np.cos(angulos[:n // 2]), r0 * np.sin(angulos[:n // 2])])

    # Classe 1: anel externo
    r1 = np.random.uniform(1.5, 2.5, n // 2)
    X1 = np.column_stack([r1 * np.cos(angulos[n // 2:]), r1 * np.sin(angulos[n // 2:])])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)])

    # Arquitetura: 2 entradas -> 8 -> 4 -> 1 saída
    modelo = MLP(camadas=[2, 8, 4, 1], taxa_aprendizado=0.1, n_epocas=1000)
    modelo.fit(X, y)

    print(f"\nAcurácia final: {modelo.acuracia(X, y) * 100:.1f}%")
    plot_resultados(X, y, modelo, "MLP - Classificação de Círculos")


if __name__ == "__main__":
    exemplo_xor()
    exemplo_classificacao_2d()
