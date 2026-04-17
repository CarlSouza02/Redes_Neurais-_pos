"""
Rede Neural Recorrente (RNN/LSTM) - Previsão de Séries Temporais
================================================================
As RNNs são projetadas para processar dados sequenciais, mantendo
um estado interno (memória). A LSTM (Long Short-Term Memory) resolve
o problema do gradiente que desaparece, usando portões de controle.

Dataset: Dados de preço sintético de ações (gerado com NumPy)

Arquitetura:
    Input(sequência) -> LSTM(64) -> LSTM(32) -> Dense(1)

Tarefa: Prever o próximo valor de uma série temporal.
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def gerar_serie_temporal(n_pontos=1000, ruido=0.1, freq=0.1):
    """
    Gera uma série temporal sintética com tendência, sazonalidade e ruído.
    Simula variação de preço de um ativo financeiro.
    """
    np.random.seed(42)
    t = np.arange(n_pontos)
    tendencia = 0.02 * t
    sazonalidade = 10 * np.sin(2 * np.pi * freq * t)
    ruido_gaussiano = ruido * np.random.randn(n_pontos) * 5
    serie = tendencia + sazonalidade + ruido_gaussiano + 50
    return serie.astype(np.float32)


def criar_sequencias(dados, n_passos):
    """
    Transforma a série temporal em pares (X, y) para aprendizado supervisionado.
    X: janela de n_passos valores anteriores
    y: próximo valor a prever
    """
    X, y = [], []
    for i in range(len(dados) - n_passos):
        X.append(dados[i:i + n_passos])
        y.append(dados[i + n_passos])
    return np.array(X), np.array(y)


def normalizar(dados, media=None, desvio=None):
    """Normalização Z-score."""
    if media is None:
        media = dados.mean()
    if desvio is None:
        desvio = dados.std()
    return (dados - media) / desvio, media, desvio


def desnormalizar(dados_norm, media, desvio):
    """Desfaz a normalização."""
    return dados_norm * desvio + media


def construir_modelo_lstm(n_passos):
    """Constrói o modelo LSTM para regressão de séries temporais."""
    modelo = keras.Sequential([
        layers.Input(shape=(n_passos, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name="LSTM_Series_Temporais")

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return modelo


def plot_serie(serie, titulo="Série Temporal Sintética"):
    """Plota a série temporal completa."""
    plt.figure(figsize=(12, 4))
    plt.plot(serie, color="steelblue", linewidth=0.8)
    plt.xlabel("Tempo")
    plt.ylabel("Valor")
    plt.title(titulo)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("rnn_serie_temporal.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'rnn_serie_temporal.png'")


def plot_predicao(y_real, y_pred_treino, y_pred_teste, n_treino, n_passos):
    """Plota os valores reais vs predições."""
    plt.figure(figsize=(14, 5))

    indice_treino = np.arange(n_passos, n_passos + len(y_pred_treino))
    indice_teste = np.arange(n_passos + n_treino, n_passos + n_treino + len(y_pred_teste))

    plt.plot(y_real, label="Série Real", color="steelblue", linewidth=0.8)
    plt.plot(indice_treino, y_pred_treino, label="Predição (Treino)",
             color="green", linewidth=1.2, linestyle="--")
    plt.plot(indice_teste, y_pred_teste, label="Predição (Teste)",
             color="red", linewidth=1.5)
    plt.axvline(x=n_passos + n_treino, color="gray", linestyle=":", label="Divisão Treino/Teste")
    plt.xlabel("Tempo")
    plt.ylabel("Valor")
    plt.title("LSTM - Previsão de Série Temporal")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("rnn_predicao.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'rnn_predicao.png'")


def plot_historico(historico):
    """Plota curva de aprendizado."""
    plt.figure(figsize=(10, 4))
    plt.plot(historico.history["loss"], label="Treino")
    plt.plot(historico.history["val_loss"], label="Validação")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Curva de Aprendizado - LSTM")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("rnn_historico.png", dpi=100)
    plt.show()
    print("Gráfico salvo em 'rnn_historico.png'")


def main():
    print("=" * 60)
    print("LSTM para Previsão de Série Temporal")
    print("=" * 60)

    tf.random.set_seed(42)
    np.random.seed(42)

    # Parâmetros
    N_PONTOS = 1000
    N_PASSOS = 30       # Janela de contexto: 30 passos anteriores
    PROPORCAO_TREINO = 0.8

    # Gera e plota a série temporal
    serie = gerar_serie_temporal(n_pontos=N_PONTOS)
    plot_serie(serie)

    # Normaliza
    serie_norm, media, desvio = normalizar(serie)
    print(f"\nEstatísticas da série: média={serie.mean():.2f}, desvio={serie.std():.2f}")

    # Cria sequências
    X, y = criar_sequencias(serie_norm, N_PASSOS)
    X = X[..., np.newaxis]  # (amostras, passos, features=1)
    print(f"Shape X: {X.shape} | Shape y: {y.shape}")

    # Divisão treino/teste
    n_treino = int(len(X) * PROPORCAO_TREINO)
    X_treino, X_teste = X[:n_treino], X[n_treino:]
    y_treino, y_teste = y[:n_treino], y[n_treino:]
    print(f"Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras")

    # Modelo
    modelo = construir_modelo_lstm(N_PASSOS)
    modelo.summary()

    # Treinamento
    print("\nIniciando treinamento...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    historico = modelo.fit(
        X_treino, y_treino,
        batch_size=32,
        epochs=50,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # Avaliação
    mse_teste, mae_teste = modelo.evaluate(X_teste, y_teste, verbose=0)
    print(f"\nResultados no Teste:")
    print(f"  MSE: {mse_teste:.4f} (normalizado)")
    print(f"  MAE: {mae_teste:.4f} (normalizado)")

    # Predições desnormalizadas
    y_pred_treino = desnormalizar(modelo.predict(X_treino, verbose=0).flatten(), media, desvio)
    y_pred_teste = desnormalizar(modelo.predict(X_teste, verbose=0).flatten(), media, desvio)
    y_real_treino = desnormalizar(y_treino, media, desvio)
    y_real_teste = desnormalizar(y_teste, media, desvio)

    mae_real = np.mean(np.abs(y_real_teste - y_pred_teste))
    print(f"  MAE real (desnormalizado): {mae_real:.2f}")

    # Visualizações
    plot_historico(historico)
    plot_predicao(serie, y_pred_treino, y_pred_teste, n_treino, N_PASSOS)

    # Salva o modelo
    modelo.save("lstm_series_temporais.keras")
    print("\nModelo salvo em 'lstm_series_temporais.keras'")


if __name__ == "__main__":
    main()
