# Rede Neural Recorrente (RNN/LSTM)

## Descrição

Implementação de uma **LSTM** com Keras/TensorFlow para **previsão de séries temporais**, aplicada a dados sintéticos que simulam a variação de preço de um ativo financeiro.

## Conceitos Abordados

- Redes Recorrentes (RNN)
- LSTM (Long Short-Term Memory) e seus portões:
  - Portão de esquecimento (*forget gate*)
  - Portão de entrada (*input gate*)
  - Portão de saída (*output gate*)
- Janela deslizante para supervisão
- Normalização Z-score
- EarlyStopping e ReduceLROnPlateau

## Arquitetura

```
Input(n_passos=30, features=1)
  → LSTM(64, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32)
  → Dropout(0.2)
  → Dense(16, ReLU)
  → Dense(1)  ← Saída: próximo valor
```

## Como Executar

```bash
pip install tensorflow numpy matplotlib
python rnn_lstm.py
```

## Saída Esperada

- Gráfico da série temporal gerada
- Curva de aprendizado (MSE)
- Predições sobrepostas à série real
- Modelo salvo em `lstm_series_temporais.keras`

## Vantagem da LSTM sobre RNN simples

A RNN simples sofre com o problema do **gradiente que desaparece** em sequências longas. A LSTM resolve isso com seus portões de memória, que aprendem a guardar ou descartar informações de longo prazo.
