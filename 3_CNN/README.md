# Rede Neural Convolucional (CNN)

## Descrição

Implementação de uma **CNN** com Keras/TensorFlow para classificação de imagens no dataset **MNIST** (dígitos manuscritos de 0 a 9).

## Conceitos Abordados

- Camadas convolucionais (Conv2D)
- Pooling (MaxPooling2D)
- Dropout para regularização
- Softmax para classificação multiclasse
- Entropia cruzada categórica

## Arquitetura

```
Input(28×28×1)
  → Conv2D(32 filtros, 3×3, ReLU)
  → MaxPooling2D(2×2)
  → Conv2D(64 filtros, 3×3, ReLU)
  → MaxPooling2D(2×2)
  → Flatten
  → Dense(128, ReLU)
  → Dropout(0.5)
  → Dense(10, Softmax)
```

## Como Executar

```bash
pip install tensorflow numpy matplotlib
python cnn.py
```

## Dependências

```
tensorflow>=2.10
numpy
matplotlib
```

## Saída Esperada

- Acurácia no conjunto de teste (~99%)
- Gráficos de acurácia e perda por época
- Visualização de predições incorretas
- Modelo salvo em `cnn_mnist.keras`

## Por que usar CNN?

As CNNs aprendem filtros espaciais automaticamente, capturando:
- Bordas e texturas (camadas iniciais)
- Formas complexas (camadas mais profundas)

Isso as torna muito mais eficientes que MLPs para imagens.
