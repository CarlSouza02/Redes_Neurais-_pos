# Perceptron Simples

## Descrição

Implementação do **Perceptron** do zero usando apenas NumPy. O Perceptron é o modelo mais básico de rede neural, capaz de aprender fronteiras de decisão lineares.

## Conceitos Abordados

- Neurônio artificial (modelo de McCulloch-Pitts)
- Função de ativação degrau
- Regra de aprendizado do Perceptron
- Convergência para dados linearmente separáveis

## Regra de Atualização dos Pesos

```
w ← w + η * (y_real - y_pred) * x
b ← b + η * (y_real - y_pred)
```

Onde `η` é a taxa de aprendizado.

## Exemplos Demonstrados

1. **Porta Lógica AND** – classificação binária simples com 4 amostras
2. **Dados 2D Linearmente Separáveis** – classificação de duas classes geradas com NumPy

## Como Executar

```bash
pip install numpy matplotlib
python perceptron.py
```

## Saída Esperada

- Pesos e bias aprendidos
- Tabela verdade com predições
- Gráfico da fronteira de decisão
- Gráfico de erros por época

## Limitações

O Perceptron **não consegue** resolver problemas não-linearmente separáveis (como XOR). Para isso, é necessário o MLP com backpropagation.
