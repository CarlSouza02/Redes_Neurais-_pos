"""
Autoencoder - Aprendizado de Representações com Keras/TensorFlow
================================================================
Um Autoencoder é uma rede neural que aprende a compactar (codificar) os
dados em uma representação de menor dimensão e depois reconstruí-los
(decodificar). Usado para:
  - Redução de dimensionalidade
  - Remoção de ruído (denoising)
  - Detecção de anomalias
  - Geração de dados

Arquitetura:
    Encoder: Input -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Latente(16)
    Decoder: Latente(16) -> Dense(32, ReLU) -> Dense(64, ReLU) -> Dense(128, ReLU) -> Output

Dataset: MNIST (784 pixels por imagem)
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def carregar_dados():
    """Carrega e pré-processa o MNIST para o autoencoder."""
    (X_treino, y_treino), (X_teste, y_teste) = keras.datasets.mnist.load_data()

    # Normaliza e achata (28x28 -> 784)
    X_treino = X_treino.astype("float32") / 255.0
    X_teste = X_teste.astype("float32") / 255.0
    X_treino = X_treino.reshape(-1, 784)
    X_teste = X_teste.reshape(-1, 784)

    print(f"Dados de treino: {X_treino.shape}")
    print(f"Dados de teste:  {X_teste.shape}")
    return (X_treino, y_treino), (X_teste, y_teste)


def adicionar_ruido(X, nivel_ruido=0.3):
    """Adiciona ruído gaussiano para o Denoising Autoencoder."""
    ruido = nivel_ruido * np.random.randn(*X.shape)
    return np.clip(X + ruido, 0.0, 1.0).astype(np.float32)


def construir_autoencoder(dim_entrada=784, dim_latente=16):
    """
    Constrói o Autoencoder com encoder e decoder separados.
    Retorna: (autoencoder completo, encoder, decoder)
    """
    # Encoder
    encoder_input = layers.Input(shape=(dim_entrada,), name="encoder_input")
    x = layers.Dense(128, activation="relu")(encoder_input)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    latente = layers.Dense(dim_latente, activation="relu", name="latente")(x)
    encoder = keras.Model(encoder_input, latente, name="Encoder")

    # Decoder
    decoder_input = layers.Input(shape=(dim_latente,), name="decoder_input")
    x = layers.Dense(32, activation="relu")(decoder_input)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    decoder_output = layers.Dense(dim_entrada, activation="sigmoid")(x)
    decoder = keras.Model(decoder_input, decoder_output, name="Decoder")

    # Autoencoder completo
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = keras.Model(encoder_input, autoencoder_output, name="Autoencoder")

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
    )
    return autoencoder, encoder, decoder


def plot_reconstrucoes(X_original, X_ruidoso, X_reconstruido, n=10, titulo="Reconstruções"):
    """Plota originais, versões ruidosas e reconstruções lado a lado."""
    plt.figure(figsize=(18, 5))
    for i in range(n):
        # Original
        plt.subplot(3, n, i + 1)
        plt.imshow(X_original[i].reshape(28, 28), cmap="gray")
        if i == 0:
            plt.ylabel("Original", fontsize=9)
        plt.axis("off")

        # Com ruído
        plt.subplot(3, n, n + i + 1)
        plt.imshow(X_ruidoso[i].reshape(28, 28), cmap="gray")
        if i == 0:
            plt.ylabel("Com Ruído", fontsize=9)
        plt.axis("off")

        # Reconstruído
        plt.subplot(3, n, 2 * n + i + 1)
        plt.imshow(X_reconstruido[i].reshape(28, 28), cmap="gray")
        if i == 0:
            plt.ylabel("Reconstruído", fontsize=9)
        plt.axis("off")

    plt.suptitle(titulo, fontsize=12)
    plt.tight_layout()
    plt.savefig("autoencoder_reconstrucoes.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'autoencoder_reconstrucoes.png'")


def plot_espaco_latente(encoder, X_teste, y_teste, n_amostras=2000):
    """Visualiza o espaço latente 2D usando os dois primeiros componentes."""
    # Usa apenas as 2 primeiras dimensões do espaço latente para visualização
    codigo = encoder.predict(X_teste[:n_amostras], verbose=0)
    # Se dim_latente > 2, aplica PCA manual para visualizar em 2D
    if codigo.shape[1] > 2:
        U, S, Vt = np.linalg.svd(codigo - codigo.mean(axis=0), full_matrices=False)
        codigo_2d = U[:, :2] * S[:2]
    else:
        codigo_2d = codigo

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(codigo_2d[:, 0], codigo_2d[:, 1],
                          c=y_teste[:n_amostras], cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(scatter, label="Dígito")
    plt.xlabel("Componente 1 do Espaço Latente")
    plt.ylabel("Componente 2 do Espaço Latente")
    plt.title("Espaço Latente (Projeção 2D)")
    plt.tight_layout()
    plt.savefig("autoencoder_espaco_latente.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'autoencoder_espaco_latente.png'")


def detectar_anomalias(autoencoder, X_normal, X_anomalo, limiar_percentil=95):
    """
    Demonstra detecção de anomalias via erro de reconstrução.
    Dados com alto erro de reconstrução são considerados anômalos.
    """
    erro_normal = np.mean(np.square(X_normal - autoencoder.predict(X_normal, verbose=0)), axis=1)
    erro_anomalo = np.mean(np.square(X_anomalo - autoencoder.predict(X_anomalo, verbose=0)), axis=1)

    limiar = np.percentile(erro_normal, limiar_percentil)

    vp = np.sum(erro_anomalo > limiar)   # Verdadeiro Positivo
    fp = np.sum(erro_normal > limiar)    # Falso Positivo
    fn = np.sum(erro_anomalo <= limiar)  # Falso Negativo
    vn = np.sum(erro_normal <= limiar)   # Verdadeiro Negativo

    print(f"\nDetecção de Anomalias (limiar no percentil {limiar_percentil}):")
    print(f"  Limiar de reconstrução: {limiar:.4f}")
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0
    revocacao = vp / (vp + fn) if (vp + fn) > 0 else 0
    print(f"  Precisão: {precisao * 100:.1f}%")
    print(f"  Revocação: {revocacao * 100:.1f}%")

    plt.figure(figsize=(10, 4))
    plt.hist(erro_normal, bins=50, alpha=0.6, label="Normal (dígito 0)", color="blue", density=True)
    plt.hist(erro_anomalo, bins=50, alpha=0.6, label="Anomalia (dígito 9)", color="red", density=True)
    plt.axvline(limiar, color="black", linestyle="--", linewidth=2, label=f"Limiar ({limiar:.4f})")
    plt.xlabel("Erro de Reconstrução (MSE)")
    plt.ylabel("Densidade")
    plt.title("Distribuição do Erro de Reconstrução")
    plt.legend()
    plt.tight_layout()
    plt.savefig("autoencoder_anomalias.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'autoencoder_anomalias.png'")


def main():
    print("=" * 60)
    print("Autoencoder para Redução de Dimensionalidade e Denoising")
    print("=" * 60)

    tf.random.set_seed(42)
    np.random.seed(42)

    DIM_LATENTE = 16

    # Dados
    (X_treino, y_treino), (X_teste, y_teste) = carregar_dados()

    # Adiciona ruído para treinar o Denoising Autoencoder
    X_treino_ruidoso = adicionar_ruido(X_treino, nivel_ruido=0.3)
    X_teste_ruidoso = adicionar_ruido(X_teste, nivel_ruido=0.3)

    # Modelo
    autoencoder, encoder, decoder = construir_autoencoder(
        dim_entrada=784, dim_latente=DIM_LATENTE
    )
    autoencoder.summary()

    # Treinamento (Denoising: entrada ruidosa, alvo limpo)
    print("\nTreinando o Denoising Autoencoder...")
    historico = autoencoder.fit(
        X_treino_ruidoso, X_treino,
        batch_size=256,
        epochs=20,
        validation_data=(X_teste_ruidoso, X_teste),
        verbose=1,
    )

    # Reconstruções
    X_reconstruido = autoencoder.predict(X_teste_ruidoso[:10], verbose=0)
    plot_reconstrucoes(X_teste[:10], X_teste_ruidoso[:10], X_reconstruido)

    # Espaço latente
    plot_espaco_latente(encoder, X_teste, y_teste)

    # Detecção de anomalias (dígito 0 = normal, dígito 9 = anomalia)
    X_normal = X_teste[y_teste == 0][:200]
    X_anomalo = X_teste[y_teste == 9][:200]
    detectar_anomalias(autoencoder, X_normal, X_anomalo)

    # Salva o modelo
    autoencoder.save("autoencoder_mnist.keras")
    encoder.save("encoder_mnist.keras")
    print("\nModelos salvos: 'autoencoder_mnist.keras', 'encoder_mnist.keras'")


if __name__ == "__main__":
    main()
