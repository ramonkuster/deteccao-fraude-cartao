# Detecção de fraude em transações de cartão de crédito

Projeto final da disciplina **Modelagem de Análise e Aquisição de Padrões em IA**
Centro Universitário Salesiano (UniSales), Sistemas de Informação, 8º período, 2026/2
Autor: Ramon Bischoli Kuster
Professor: Alexandre Barbosa de Souza

## Sobre o projeto

Sistema de classificação de transações de cartão de crédito em normais ou fraudulentas. São comparados um modelo supervisionado (Regressão Logística) e um modelo de deep learning (MLP), com métricas adequadas a dados fortemente desbalanceados (apenas 0,17% de fraudes).

## Dados

- Base: *Credit Card Fraud Detection*, do Machine Learning Group da ULB (Université Libre de Bruxelles)
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284.807 transações, 31 colunas, 492 fraudes
- As colunas `V1` a `V28` são resultado de PCA aplicado pelos autores, por isso são anônimas
- O arquivo `creditcard.csv` (cerca de 150 MB) **não está neste repositório**; é preciso baixá-lo no link acima

Referência: Dal Pozzolo, A.; Caelen, O.; Johnson, R. A.; Bontempi, G. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE CIDM, 2015.

## Como executar

1. Abra o notebook `projeto_fraude.ipynb` no [Google Colab](https://colab.research.google.com)
2. Envie o `creditcard.csv` para a pasta `/content` do Colab
3. Execute com **Ambiente de execução → Executar tudo**

Bibliotecas: pandas, numpy, matplotlib, scikit-learn e TensorFlow/Keras (todas já disponíveis no Colab).

## Etapas

1. Análise exploratória dos dados
2. Pré-processamento: divisão estratificada 80/20 e padronização de `Time` e `Amount` (scaler ajustado só no treino)
3. PCA (visualização) e LDA (redução de dimensionalidade)
4. Regressão Logística, com ajuste do limiar e do hiperparâmetro `C` (GridSearchCV)
5. MLP (64 e 32 neurônios), com ajuste de arquitetura e peso de classe
6. Validação cruzada estratificada (5 dobras)
7. Comparação por curvas Precision-Recall e ROC, matrizes de confusão e análise de erros

## Principais resultados

Conjunto de teste (56.962 transações, 98 fraudes):

| Modelo | AUC-PR | F1 (limiar na validação) |
|---|---|---|
| Regressão Logística (balanced) | 0,712 | 0,804 |
| Regressão Logística (C = 0,1) | 0,741 | - |
| Regressão Logística + LDA | 0,720 | - |
| MLP | 0,800 | 0,796 |

Validação cruzada (AUC-PR): MLP 0,842 ± 0,031 e Regressão Logística 0,757 ± 0,051.

Os dois modelos principais ficaram praticamente empatados no F1. A vantagem da MLP aparece no AUC-PR, métrica mais informativa neste cenário desbalanceado.

## Limitações

- O teste tem apenas 98 fraudes, então as métricas variam bastante com poucos casos
- As colunas anônimas dificultam a interpretação do modelo
- Os dados cobrem apenas dois dias de transações europeias

## Estrutura do repositório

- `projeto_fraude.ipynb`: notebook completo, com texto, código e resultados
- `README.md`: este arquivo
