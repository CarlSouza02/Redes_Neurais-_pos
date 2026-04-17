# Redes Neurais — Pós-Graduação

Repositório com modelos e projetos de **Redes Neurais** implementados em Python, cobrindo desde o Perceptron básico até arquiteturas modernas com Keras/TensorFlow.

## Projetos

| Pasta | Modelo | Descrição |
|---|---|---|
| [`1_Perceptron/`](1_Perceptron/) | Perceptron | Classificador linear implementado do zero com NumPy |
| [`2_MLP/`](2_MLP/) | Multi-Layer Perceptron | Backpropagation do zero; resolve o problema XOR |
| [`3_CNN/`](3_CNN/) | CNN | Classificação de imagens MNIST (~99% de acurácia) |
| [`4_RNN/`](4_RNN/) | LSTM | Previsão de séries temporais com janela deslizante |
| [`5_Autoencoder/`](5_Autoencoder/) | Autoencoder | Remoção de ruído, redução de dimensão e detecção de anomalias |

## Requisitos

```bash
pip install numpy matplotlib tensorflow
```

> Os projetos `1_Perceptron` e `2_MLP` usam apenas **NumPy** e **Matplotlib**.
> Os demais projetos requerem **TensorFlow ≥ 2.10**.

## Como Executar

Cada projeto é independente. Acesse a pasta desejada e execute o script principal:

```bash
cd 1_Perceptron && python perceptron.py
cd 2_MLP        && python mlp.py
cd 3_CNN        && python cnn.py
cd 4_RNN        && python rnn_lstm.py
cd 5_Autoencoder && python autoencoder.py
```

## Arquitetura dos Modelos

```
Perceptron  →  MLP  →  CNN  →  LSTM  →  Autoencoder
(linear)      (não-   (imagem)  (seq.)    (não-sup.)
               linear)
```

## Referências

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.*
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.*
- LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition.*
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.*
- Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with Neural Networks.*

