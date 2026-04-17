"""
Rede Neural Convolucional (CNN) - Classificação de Imagens com Keras/TensorFlow
================================================================================
As CNNs são especializadas no processamento de dados com estrutura de grade,
como imagens. Usam camadas convolucionais para aprender filtros espaciais
hierárquicos automaticamente.

Dataset: MNIST (dígitos manuscritos 0-9)

Arquitetura:
    Input(28x28x1)
    -> Conv2D(32, 3x3, ReLU) -> MaxPooling2D(2x2)
    -> Conv2D(64, 3x3, ReLU) -> MaxPooling2D(2x2)
    -> Flatten
    -> Dense(128, ReLU) -> Dropout(0.5)
    -> Dense(10, Softmax)
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def carregar_dados():
    """Carrega e pré-processa o dataset MNIST."""
    (X_treino, y_treino), (X_teste, y_teste) = keras.datasets.mnist.load_data()

    # Normaliza os pixels para [0, 1] e adiciona dimensão do canal
    X_treino = X_treino.astype("float32") / 255.0
    X_teste = X_teste.astype("float32") / 255.0
    X_treino = X_treino[..., np.newaxis]
    X_teste = X_teste[..., np.newaxis]

    print(f"Dados de treino: {X_treino.shape} | Rótulos: {y_treino.shape}")
    print(f"Dados de teste:  {X_teste.shape}  | Rótulos: {y_teste.shape}")

    return (X_treino, y_treino), (X_teste, y_teste)


def construir_modelo():
    """Constrói e compila o modelo CNN."""
    modelo = keras.Sequential([
        # Bloco Convolucional 1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
                      padding="same", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Bloco Convolucional 2
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Classificador
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="CNN_MNIST")

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def plot_exemplos(X, y, n=10):
    """Plota exemplos do dataset."""
    plt.figure(figsize=(12, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].squeeze(), cmap="gray")
        plt.title(str(y[i]))
        plt.axis("off")
    plt.suptitle("Exemplos do MNIST", y=1.05)
    plt.tight_layout()
    plt.savefig("cnn_exemplos.png", dpi=100)
    plt.show()


def plot_historico(historico):
    """Plota as curvas de acurácia e perda durante o treinamento."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history["accuracy"], label="Treino")
    plt.plot(historico.history["val_accuracy"], label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por Época")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historico.history["loss"], label="Treino")
    plt.plot(historico.history["val_loss"], label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.title("Perda por Época")
    plt.legend()

    plt.tight_layout()
    plt.savefig("cnn_historico.png", dpi=100)
    plt.show()
    print("Gráficos salvos em 'cnn_historico.png'")


def visualizar_predicoes_erradas(modelo, X_teste, y_teste, n=10):
    """Visualiza algumas predições incorretas do modelo."""
    y_pred = np.argmax(modelo.predict(X_teste, verbose=0), axis=1)
    indices_errados = np.where(y_pred != y_teste)[0]

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices_errados[:n]):
        plt.subplot(1, n, i + 1)
        plt.imshow(X_teste[idx].squeeze(), cmap="gray")
        plt.title(f"Real:{y_teste[idx]}\nPred:{y_pred[idx]}", fontsize=8)
        plt.axis("off")
    plt.suptitle("Predições Incorretas")
    plt.tight_layout()
    plt.savefig("cnn_erros.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'cnn_erros.png'")


def main():
    print("=" * 60)
    print("CNN para Classificação de Dígitos MNIST")
    print("=" * 60)

    # Reprodutibilidade
    tf.random.set_seed(42)
    np.random.seed(42)

    # Dados
    (X_treino, y_treino), (X_teste, y_teste) = carregar_dados()
    plot_exemplos(X_treino, y_treino)

    # Modelo
    modelo = construir_modelo()
    modelo.summary()

    # Treinamento
    print("\nIniciando treinamento...")
    historico = modelo.fit(
        X_treino, y_treino,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1,
    )

    # Avaliação
    perda, acuracia = modelo.evaluate(X_teste, y_teste, verbose=0)
    print(f"\nResultados no Conjunto de Teste:")
    print(f"  Acurácia: {acuracia * 100:.2f}%")
    print(f"  Perda:    {perda:.4f}")

    # Visualizações
    plot_historico(historico)
    visualizar_predicoes_erradas(modelo, X_teste, y_teste)

    # Salva o modelo
    modelo.save("cnn_mnist.keras")
    print("\nModelo salvo em 'cnn_mnist.keras'")


if __name__ == "__main__":
    main()
