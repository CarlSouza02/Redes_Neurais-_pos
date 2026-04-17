# Autoencoder

## Descrição

Implementação de um **Denoising Autoencoder** com Keras/TensorFlow no dataset **MNIST**. Demonstra três aplicações práticas: remoção de ruído, visualização do espaço latente e detecção de anomalias.

## Conceitos Abordados

- Arquitetura Encoder-Decoder
- Espaço latente (bottleneck)
- Denoising Autoencoder
- Redução de dimensionalidade não-linear
- Detecção de anomalias via erro de reconstrução

## Arquitetura

```
Encoder:
  Input(784)
    → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU)
    → Dense(16)  ← Espaço Latente

Decoder:
  Latente(16)
    → Dense(32, ReLU) → Dense(64, ReLU) → Dense(128, ReLU)
    → Dense(784, Sigmoid)  ← Reconstrução
```

## Como Executar

```bash
pip install tensorflow numpy matplotlib
python autoencoder.py
```

## Saída Esperada

- Comparação: imagens originais × ruidosas × reconstruídas
- Projeção 2D do espaço latente (colorida por classe)
- Distribuições de erro para detecção de anomalias
- Modelos salvos: `autoencoder_mnist.keras`, `encoder_mnist.keras`

## Aplicações

| Aplicação               | Como funciona |
|-------------------------|---------------|
| Remoção de ruído        | Treina com entrada ruidosa e alvo limpo |
| Redução de dimensão     | Usa o Encoder para obter representações compactas |
| Detecção de anomalias   | Dados anômalos têm alto erro de reconstrução |
| Geração de dados        | Variational Autoencoder (extensão do Autoencoder) |
