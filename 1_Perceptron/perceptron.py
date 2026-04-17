"""
Perceptron Simples - Implementação do zero com NumPy
======================================================
O Perceptron é o modelo mais básico de rede neural, proposto por Frank Rosenblatt em 1958.
Ele é capaz de classificar padrões linearmente separáveis.

Regra de atualização dos pesos:
    w = w + lr * (y_real - y_pred) * x
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """Perceptron simples para classificação binária."""

    def __init__(self, taxa_aprendizado=0.1, n_epocas=100):
        self.taxa_aprendizado = taxa_aprendizado
        self.n_epocas = n_epocas
        self.pesos = None
        self.bias = None
        self.erros_por_epoca = []

    def _funcao_ativacao(self, z):
        """Função degrau: retorna 1 se z >= 0, senão 0."""
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        """Treina o Perceptron com os dados de entrada X e rótulos y."""
        n_amostras, n_features = X.shape
        self.pesos = np.zeros(n_features)
        self.bias = 0

        for epoca in range(self.n_epocas):
            erros = 0
            for xi, yi in zip(X, y):
                z = np.dot(xi, self.pesos) + self.bias
                y_pred = self._funcao_ativacao(z)
                delta = self.taxa_aprendizado * (yi - y_pred)
                self.pesos += delta * xi
                self.bias += delta
                erros += int(delta != 0)
            self.erros_por_epoca.append(erros)
            if erros == 0:
                print(f"Convergiu na época {epoca + 1}!")
                break

        return self

    def predict(self, X):
        """Realiza predições para as amostras em X."""
        z = np.dot(X, self.pesos) + self.bias
        return self._funcao_ativacao(z)

    def acuracia(self, X, y):
        """Calcula a acurácia do modelo."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def plot_fronteira_decisao(X, y, modelo, titulo="Fronteira de Decisão do Perceptron"):
    """Plota a fronteira de decisão aprendida pelo Perceptron."""
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                            np.arange(x2_min, x2_max, 0.02))
    Z = modelo.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="o", label="Classe 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="s", label="Classe 1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(titulo)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(modelo.erros_por_epoca) + 1), modelo.erros_por_epoca, marker="o")
    plt.xlabel("Época")
    plt.ylabel("Erros de Classificação")
    plt.title("Erros por Época")

    plt.tight_layout()
    plt.savefig("perceptron_resultado.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'perceptron_resultado.png'")


def exemplo_porta_logica():
    """Demonstra o Perceptron aprendendo a porta lógica AND."""
    print("=" * 50)
    print("Exemplo: Porta Lógica AND")
    print("=" * 50)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND

    modelo = Perceptron(taxa_aprendizado=0.1, n_epocas=20)
    modelo.fit(X, y)

    print(f"\nPesos aprendidos: {modelo.pesos}")
    print(f"Bias aprendido: {modelo.bias}")
    print(f"Acurácia: {modelo.acuracia(X, y) * 100:.1f}%\n")
    print("Tabela Verdade AND:")
    print(f"{'Entrada':^12} | {'Saída Real':^10} | {'Predição':^10}")
    print("-" * 38)
    for xi, yi, y_pred in zip(X, y, modelo.predict(X)):
        print(f"{str(xi):^12} | {yi:^10} | {y_pred:^10}")


def exemplo_dados_lineares():
    """Demonstra o Perceptron em dados linearmente separáveis com 2D."""
    print("\n" + "=" * 50)
    print("Exemplo: Dados Linearmente Separáveis 2D")
    print("=" * 50)
    np.random.seed(42)
    X_classe0 = np.random.randn(50, 2) + np.array([-2, -2])
    X_classe1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([X_classe0, X_classe1])
    y = np.hstack([np.zeros(50), np.ones(50)]).astype(int)

    # Embaralha os dados
    indices = np.random.permutation(len(y))
    X, y = X[indices], y[indices]

    modelo = Perceptron(taxa_aprendizado=0.1, n_epocas=50)
    modelo.fit(X, y)

    print(f"\nAcurácia final: {modelo.acuracia(X, y) * 100:.1f}%")
    plot_fronteira_decisao(X, y, modelo)


if __name__ == "__main__":
    exemplo_porta_logica()
    exemplo_dados_lineares()
