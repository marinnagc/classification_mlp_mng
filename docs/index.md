# MLP no Adult Income — Relatório

---

## 1) Objetivo

Neste projeto, buscamos prever se a renda anual de uma pessoa é **>50K** ou **<=50K** utilizando o conjunto **Adult Income**.  
Trata-se de um problema de **classificação binária** com amostra ampla (30k+ observações) e múltiplos atributos, majoritariamente categóricos.

- Não utilizamos datasets clássicos superexpostos (Titanic/Iris/Wine).
- Implementamos um **MLP em NumPy**.

---

## 2) Dataset Selection

**Nome do dataset:** *Adult Income (Census Income)*

**Fonte:** [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/adult-census-income)

**Tamanho utilizado:** aproximadamente **30 000 registros** após limpeza, com cerca de **105 variáveis** numéricas/categóricas (após codificação one-hot) e **1 variável-alvo** (`income`).

**Descrição:** O Adult Census Income Dataset, também conhecido como Census Income ou Adult dataset, foi extraído do Censo norte-americano de 1994 e tem como objetivo prever se a renda anual de um indivíduo ultrapassa US$ 50.000, com base em variáveis demográficas e profissionais. Trata-se de um problema clássico de classificação binária, amplamente utilizado para avaliar métodos de pré-processamento, tratamento de desbalanceamento e modelagem supervisionada em aprendizado de máquina.

**Motivação da escolha:** é um problema **realista e amplamente estudado** em aprendizado de máquina, útil para explorar técnicas de **classificação supervisionada**, **tratamento de desbalanceamento**, e **engenharia de variáveis categóricas**. Além disso, sua **complexidade moderada** (número alto de atributos após codificação) e **desbalanceamento entre classes** tornam o dataset adequado para avaliar métricas além da acurácia, alem de aplicarmos os conhecimentos adquiridos na disciplina.


---

## 3) Dataset Explanation

Todas as exploracoes e processos feitos nesse data set esta no arquivo : (https://github.com/marinnagc/classification_mlp_mng/blob/main/exploracao.ipynb)

### 3.1) Descriptive Statistics

As variáveis numéricas apresentam escalas e distribuições bastante distintas. A média de idade é de aproximadamente 38,6 anos, com desvio padrão de 13,6, indicando uma amostra adulta heterogênea. A jornada média semanal é de 40 horas, coerente com o padrão de tempo integral.
A variável education.num tem média 10, correspondente a um nível educacional médio entre Some College e Bachelors.
Os atributos capital-gain e capital-loss são altamente assimétricos, com a maior parte dos valores igual a zero — o que indica que apenas uma pequena fração da população declarou ganhos ou perdas de capital relevantes.

| Variável       | Média   | Desvio padrão | Mínimo | Máximo    |
| -------------- | ------- | ------------- | ------ | --------- |
| age            | 38.6    | 13.6          | 17     | 90        |
| hours.per.week | 40.4    | 12.3          | 1      | 99        |
| education.num  | 10.1    | 2.6           | 1      | 16        |
| capital.gain   | 1077.6  | 7385.3        | 0      | 99999     |
| capital.loss   | 87.3    | 403.0         | 0      | 4356      |
| fnlwgt         | 189,778 | 105,550       | 12,285 | 1,484,705 |


### 3.2)  Target Distribution

A variável-alvo (income) é desbalanceada, com cerca de 75% dos indivíduos recebendo até US$ 50K e 25% acima desse valor.
Isso sugere a necessidade de métricas de avaliação além da acurácia, como F1-score, ROC AUC e Precision–Recall AUC.

![Target_dist](targ_dist.png)

### 3.3)  Categorical Variables


Ao analisar as variaveis categoricas, a maior parte dos indivíduos pertence à classe de trabalho “Private”, seguida de trabalhadores autônomos (Self-emp-not-inc e Self-emp-inc) e servidores públicos (Local-gov e State-gov).


Em termos de ocupação, destacam-se “Prof-specialty”, “Craft-repair” e “Exec-managerial”, representando setores de maior qualificação.
A esmagadora maioria dos registros refere-se a pessoas nascidas nos Estados Unidos, com poucos exemplos de outros países.

Podemos prover isso com o grafico abaixo:


![categorical_distribution](categorical_distributions_combined.png)


### 3.4)  Correlation Matrix

O heatmap de correlação mostra que não há relações lineares fortes entre as variáveis numéricas.
A maior correlação observada é entre education.num e hours.per.week (r ≈ 0.15), ainda assim bastante baixa.
Isso sugere que cada atributo contribui de forma relativamente independente para o modelo, o que pode favorecer métodos que capturam interações não lineares (como árvores de decisão e ensemble models).

![alt text](correlation.png)


## 4) Data Cleaning and Normalization

Esse processo foi feito em :(https://github.com/marinnagc/classification_mlp_mng/blob/main/limpeza.ipynb)

Durante a etapa de limpeza, substituímos todos os valores representados por “?” por None e preenchemos os valores ausentes com a moda de cada coluna (valor mais frequente).
Essa estratégia preserva a distribuição original dos dados e evita distorções que poderiam surgir com imputações baseadas em média ou mediana.

Na normalização, aplicamos o Z-score às variáveis numéricas, de forma que cada atributo passou a ter média 0 e desvio padrão 1, garantindo uma escala uniforme entre variáveis como age, hours-per-week e capital-gain.


Before → After

| age | workclass | capital.gain | hours.per.week |
| --- | --------- | ------------ | -------------- |
| 39  | ?         | 0            | 40             |
| 50  | Private   | 7688         | 60             |
| 28  | ?         | 0            | 40             |

Após limpeza e normalização:

| age (z) | workclass | capital.gain (z) | hours.per.week (z) |
| ------- | --------- | ---------------- | ------------------ |
| -0.10   | Private   | -0.12            | -0.04              |
| 0.85    | Private   | 1.75             | 1.53               |
| -1.05   | Private   | -0.12            | -0.04              |



## 5) MLP Implementation


Neste projeto, foram utilizadas duas implementações de **MLP (Perceptron Multicamadas)**:
uma com a biblioteca **scikit-learn**, para referência e benchmarking,
e outra **implementada manualmente em NumPy**, para demonstrar os princípios de *forward pass*, *backpropagation* e *early stopping*.

### 🔹 MLP com scikit-learn

```python
# mlp_biblio.py
import numpy as np
from sklearn.neural_network import MLPClassifier

class SklearnMLPModel:
    """MLP com scikit-learn: interface comum .fit(), .predict_proba(), .predict(), .loss_curve()."""
    def __init__(self,
                 hidden_layer_sizes=(64,),
                 lr=1e-3,
                 alpha=1e-4,
                 batch_size=128,
                 max_iter=100,
                 early_stopping=True,
                 random_state=42):
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=lr,
            batch_size=batch_size,
            max_iter=max_iter,
            alpha=alpha,             # Regularização L2
            early_stopping=early_stopping,
            n_iter_no_change=10,
            random_state=random_state
        )
        self.fitted_ = False

    def fit(self, X_tr, y_tr):
        self.clf.fit(X_tr, y_tr)
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        assert self.fitted_, "Treine o modelo antes de prever."
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X, thr=0.5):
        proba = self.predict_proba(X)
        return (proba >= thr).astype(int)

    def loss_curve(self):
        return getattr(self.clf, "loss_curve_", None)
```

**Principais hiperparâmetros:**

* `hidden_layer_sizes`: define o número de neurônios por camada oculta (ex.: `(64,)` → 1 camada com 64 unidades).
* `lr` (`learning_rate_init`): taxa de aprendizado usada pelo otimizador *Adam*. Controla o tamanho do passo nas atualizações de peso.
* `alpha`: termo de regularização L2 (*weight decay*), que ajuda a evitar *overfitting*.
* `batch_size`: tamanho do mini-lote durante o treinamento (padrão = 128).
* `max_iter`: número máximo de épocas (iterações completas sobre o conjunto de treino).
* `early_stopping`: interrompe o treinamento se não houver melhora na validação por várias épocas.
* `random_state`: semente para reprodutibilidade dos resultados.

---

### 🔹 MLP implementado manualmente (NumPy)

```python
# mlp_manual.py
import numpy as np

def _sigmoid(z): return 1 / (1 + np.exp(-z))
def _relu(z): return np.maximum(0, z)
def _relu_grad(z): return (z > 0).astype(z.dtype)

def _bce_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

class _MLPBinary:
    """MLP de uma camada oculta: ReLU → Sigmoid, perda BCE, SGD mini-batch, L2 e early-stopping."""
    def __init__(self, n_in, n_hidden=64, lr=1e-3, l2=1e-4, seed=42):
        rng = np.random.default_rng(seed)
        # Inicialização He para W1 (ReLU) e Xavier para W2
        self.W1 = rng.normal(0, np.sqrt(2.0/n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros((n_hidden,))
        self.W2 = rng.normal(0, np.sqrt(1.0/n_hidden), size=(n_hidden, 1))
        self.b2 = np.zeros((1,))
        self.lr, self.l2 = lr, l2

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        yhat = _sigmoid(z2).ravel()
        return yhat, (X, z1, a1, z2, yhat)

    def backward(self, cache, y_true):
        X, z1, a1, z2, yhat = cache
        N = X.shape[0]
        y_true = y_true.reshape(-1, 1)
        yhat = yhat.reshape(-1, 1)

        dz2 = (yhat - y_true) / N
        dW2 = a1.T @ dz2 + self.l2 * self.W2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * _relu_grad(z1)
        dW1 = X.T @ dz1 + self.l2 * self.W1
        db1 = dz1.sum(axis=0)

        # Atualização dos pesos
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict_proba(self, X):
        yhat, _ = self.forward(X)
        return yhat
```

---

**Principais hiperparâmetros:**

* `n_hidden`: número de neurônios na camada oculta (capacidade do modelo).
* `lr`: *learning rate* (taxa de aprendizado); controla o quão rápido os pesos são atualizados.
* `l2`: coeficiente de regularização L2 (penaliza pesos grandes, reduz *overfitting*).
* `batch_size`: tamanho dos mini-lotes de dados usados em cada atualização.
* `epochs`: número máximo de épocas de treinamento.
* `patience`: número de épocas sem melhora antes do *early stopping*.
* `seed`: semente para reprodutibilidade.

---

### 🔹 Aplicacao:

Utilizamos esse dois arquivos nesse jupyter: (https://github.com/marinnagc/classification_mlp_mng/blob/main/main.ipynb)


## 5) Treinamento e Validação

**Protocolo**
- **Divisão estratificada**: **70%** treino, **15%** validação, **15%** teste.  
- **Padronização** ajustada no **treino** e aplicada a validação e teste.  
- **Early stopping** com monitoramento da **val loss** (paciência = 10) e restauração do melhor estado.  
- **Hiperparâmetros-base**: camadas ocultas `(128, 64)`, `lr = 1e-2`, `batch = 256`, `l2 = 1e-4`.

**Comportamento esperado**
- Redução consistente da perda em treino;  
- Métricas de validação próximas às de treino (pequeno “gap” sugere overfitting sob controle).

---

## 6) Curvas de aprendizado

- ![Loss (train/val)](curve_loss.png)
*Queda consistente da BCE; pouca diferença entre treino e validação → baixo overfitting.*

- ![Accuracy (train/val)](curve_acc.png)
*Accuracy de validação acompanha a de treino; ganhos marginais após ~40–50 épocas.*


**Leitura**  
As curvas tendem a apresentar queda acentuada nas primeiras épocas e estabilização posterior.  
Um distanciamento significativo entre treino e validação indica sobreajuste.


---

## 7) Avaliação (Teste)

**Baseline**: acurácia da classe majoritária ≈ **0,75**.

**Resultados típicos observados (Adult, `threshold = 0,5`)**
- **Accuracy** ≈ **0,84**  
- **Precision** ≈ **0,71**  
- **Recall** ≈ **0,60**  
- **F1** ≈ **0,65**  
- **Matriz de confusão** (TN, FP, FN, TP) ≈ `(3125, 271, 451, 676)`

**Interpretação**
- Superamos o baseline com margem confortável.  
- O compromisso entre precisão e recall é adequado para `threshold = 0,5`; ajustes de limiar podem privilegiar um ou outro conforme o custo de erros.

**Curvas ROC/PR**
- ![ROC (teste)](roc_curve_test.png) 
*AUC ≈ 0,91 → boa separação entre classes. Linha tracejada representa um classificador aleatório.*
e ![PR (teste)](pr_curve_test.png)
*AP ≈ 0,78. Em dados desbalanceados, a PR é mais informativa que a ROC; alta precisão para faixas de recall até ~0,6–0,7.*
  

  
---

## 8) Conclusões

- O MLP superou o **baseline** de 0,75, alcançando cerca de **0,84** de acurácia em teste, com **F1** competitivo.  
- Em dados desbalanceados, **F1** e **PR-AUC** complementam a leitura de **accuracy**.  
- O ajuste de **threshold** permite calibrar o compromisso **precisão vs. recall** conforme o custo de FP/FN.  
- Possibilidades de avanço incluem **pesos por classe** na BCE, **tuning** (número de camadas/neurônios, `lr`, `l2`), **agrupamento de categorias raras** (e.g., `native-country`) e, em bibliotecas, **dropout** e **batch normalization**.

---

## 9) Reprodutibilidade

1. Geramos **`adult_clean.csv`** (pipeline de preparação descrito na Seção 3).  
2. Realizamos **split estratificado** (70/15/15).  
3. Recalculamos o **z-score** com estatísticas do **treino** e aplicamos em validação e teste.  
4. Treinamos o MLP (NumPy) com **early stopping** e salvamos o melhor estado.  
5. Registramos histórico (CSV) e figuras (loss/acc, ROC/PR).  
6. Reportamos as métricas finais de **teste**.

