# Multi-Layer Perceptron (MLP)

## Descrição

Implementação de uma **Rede Neural Multi-Camadas** do zero com NumPy, incluindo o algoritmo de **backpropagation** para ajuste dos pesos.

## Conceitos Abordados

- Arquitetura de rede neural em camadas
- Funções de ativação: ReLU (camadas ocultas) e Sigmoid (saída)
- Algoritmo de backpropagation
- Gradiente Descendente Estocástico (SGD)
- Inicialização de pesos (método de He)
- Entropia cruzada binária como função de perda

## Arquitetura

```
Entrada → [Dense + ReLU] × n_camadas → Dense + Sigmoid → Saída
```

## Exemplos Demonstrados

1. **Problema XOR** – demonstra que o MLP resolve problemas não-lineares, ao contrário do Perceptron simples
2. **Classificação de Círculos** – fronteira de decisão circular (dados não-linearmente separáveis)

## Como Executar

```bash
pip install numpy matplotlib
python mlp.py
```

## Saída Esperada

- Tabela verdade do XOR com probabilidades
- Gráficos de fronteira de decisão
- Curva de aprendizado (perda × época)

## Comparação com Perceptron

| Característica       | Perceptron | MLP |
|----------------------|-----------|-----|
| Camadas              | 1         | ≥2  |
| Fronteira de decisão | Linear    | Não-linear |
| XOR                  | ✗         | ✓   |
| Backpropagation      | ✗         | ✓   |
