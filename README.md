# 📈 Previsão Multivariada da Bovespa com Deep Learning

Projeto de **Trabalho de Conclusão** que aplica técnicas de Deep Learning para prever o comportamento do índice Bovespa (^BVSP) com base em variáveis macroeconômicas globais — Dólar, S&P 500, Bolsa de Xangai, Petróleo, Minério de Ferro e Ouro.

---

## 🎯 Objetivo

Diferente dos modelos clássicos univariados (ARIMA/SARIMAX), esta abordagem reconhece que a Bovespa não opera de forma isolada: ela sofre influências contínuas de mercados e commodities globais. O projeto evolui por três estratégias complementares de previsão:

| Abordagem | Descrição |
|---|---|
| **Regressão Diária** | Prevê o ponto exato de fechamento do Ibovespa para o próximo dia útil |
| **Classificação 3 Classes** | Prevê o cenário direcional: Alta / Neutro / Baixa |
| **Classificação 5 Classes** | Prevê o cenário com granularidade de volatilidade: Alta Forte / Leve Alta / Neutro / Leve Baixa / Baixa Forte |

---

## 🗂️ Estrutura do Repositório

```
.
├── Previsao_Multivariada.ipynb       # Notebook principal: EDA, MLP, GRU e LSTM (regressão)
├── Previsao_3_Classes.ipynb          # Treinamento do classificador GRU com 3 classes
├── Previsao_5_Classes.ipynb          # Treinamento do classificador GRU com 5 classes
├── Previsao_Diaria.ipynb             # Previsão diária de ponto (regressão, modelo salvo)
├── Previsao_Diaria_3_Classes.ipynb   # Previsão diária com o modelo de 3 classes
├── Previsao_Diaria_5_Classes.ipynb   # Previsão diária com o modelo de 5 classes
├── modelos/
│   ├── modelo_gru.json                        # Arquitetura do modelo de regressão
│   ├── modelo_gru.weights.h5                  # Pesos do modelo de regressão
│   ├── modelo_gru_classificador_3c.weights.h5 # Pesos do classificador 3 classes
│   ├── modelo_gru_classificador.weights.h5    # Pesos do classificador 5 classes
│   ├── scaler_X.pkl                           # Normalizador das features (regressão)
│   ├── scaler_y.pkl                           # Normalizador do alvo (regressão)
│   ├── scaler_X_classificador_3c.pkl          # Normalizador das features (3 classes)
│   └── scaler_X_classificador.pkl             # Normalizador das features (5 classes)
└── README.md
```

---

## 🌐 Variáveis Utilizadas

Os dados são coletados automaticamente via `yfinance` com histórico de 3 anos:

| Variável | Ticker | Descrição |
|---|---|---|
| Bovespa | `^BVSP` | Índice de referência brasileiro — **alvo da previsão** |
| Dólar | `BRL=X` | Cotação USD/BRL |
| S&P 500 | `^GSPC` | Índice das 500 maiores empresas dos EUA |
| Shanghai | `000001.SS` | Bolsa da China |
| Petróleo | `BZ=F` | Contrato futuro Brent |
| Minério de Ferro | `TIO=F` | Contrato futuro de Minério |
| Ouro | `GC=F` | Contrato futuro de Ouro |

---

## ⚙️ Indicadores Técnicos (Features Adicionais)

Os modelos de classificação enriquecem os retornos brutos com três indicadores técnicos calculados sobre a Bovespa:

- **RSI (14 períodos)** — captura momentum de sobrecompra/sobrevenda
- **Distância para SMA-15** — mede o afastamento da tendência de médio prazo
- **Largura das Bandas de Bollinger (20 períodos)** — quantifica a volatilidade atual

---

## 🧠 Arquitetura dos Modelos

Todos os modelos de classificação compartilham a mesma arquitetura GRU:

```
GRU(50, return_sequences=True)  →  Dropout(0.2)
GRU(50, return_sequences=False) →  Dropout(0.2)
Dense(25, activation='relu')
Dense(N, activation='softmax')   # N = 3 ou 5 classes
```

A janela temporal utilizada é de **7 dias úteis** (`time_steps = 7`).

---

## 📊 Lógica de Classificação

### Modelo de 3 Classes

| Classe | Condição | Significado |
|---|---|---|
| 📉 Baixa | Retorno < -0,2% | Fechamento abaixo do limite inferior |
| ➖ Neutro | Entre -0,2% e +0,2% | Zona de consolidação / indecisão |
| 📈 Alta | Retorno > +0,2% | Fechamento acima do limite superior |

### Modelo de 5 Classes

| Classe | Condição | Significado |
|---|---|---|
| 📉 Baixa Forte | Retorno ≤ -1,0% | Queda expressiva |
| ↘️ Leve Baixa | Entre -1,0% e -0,2% | Correção controlada |
| ➖ Neutro | Entre -0,2% e +0,2% | Sem tendência definida |
| ↗️ Leve Alta | Entre +0,2% e +1,0% | Crescimento marginal |
| 📈 Alta Forte | Retorno > +1,0% | Rompimento de resistência |

---

## 🚀 Como Usar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/previsao-multivariada-bovespa.git
cd previsao-multivariada-bovespa
```

### 2. Crie o ambiente virtual e instale as dependências

```bash
conda create -n SeriesTemporais python=3.12
conda activate SeriesTemporais
pip install yfinance pandas numpy scikit-learn tensorflow ta joblib matplotlib seaborn plotly pmdarima watermark
```

### 3. Treinamento (executar uma única vez)

Execute na ordem:

```
Previsao_Multivariada.ipynb   → Gera modelo de regressão (modelos/)
Previsao_3_Classes.ipynb      → Gera modelo classificador 3 classes (modelos/)
Previsao_5_Classes.ipynb      → Gera modelo classificador 5 classes (modelos/)
```

### 4. Previsão diária (uso rotineiro)

Com os modelos já treinados e salvos na pasta `modelos/`, basta executar:

```
Previsao_Diaria.ipynb             → Ponto de fechamento estimado
Previsao_Diaria_3_Classes.ipynb   → Veredito: Alta / Neutro / Baixa + probabilidades
Previsao_Diaria_5_Classes.ipynb   → Veredito completo com zonas de preço e risco de volatilidade
```

---

## 📋 Exemplo de Output — Radar Bovespa (5 Classes)

```
=======================================================
 🚀 RADAR BOVESPA: 5 CLASSES (RISCO & VOLATILIDADE) 🚀
=======================================================
Cotação Atual Base: 177356 pts (Data: 20/05/2026)
Veredito Principal: 📉 BAIXA FORTE (menor que -1.0%)

📊 RAIO-X DE PROBABILIDADES:
   🟩 Alta Forte:    7.89%
   🟢 Leve Alta:     9.74%
   🟡 Neutro:       25.10%
   🟠 Leve Baixa:   15.10%
   🟥 Baixa Forte:  42.17%
-------------------------------------------------------
   ⚠️ Risco de Movimento Violento (Extremos): 50.06%

🎯 ZONAS DE PREÇO (PRÓXIMO DIA ÚTIL):
   Acima de 179129 pts         ➡️ ALTA FORTE
   Entre 177710 e 179129 pts ➡️ LEVE ALTA
   Entre 177001 e 177710 pts ➡️ NEUTRO
   Entre 175582 e 177001 pts ➡️ LEVE BAIXA
   Abaixo de 175582 pts        ➡️ BAIXA FORTE
=======================================================
```

---

## 🔬 Conclusões do Trabalho

- **Interdependência global:** A Bovespa apresenta forte correlação com Shanghai, S&P 500 e contratos de minério e petróleo. O GRU captura essas interações não lineares de forma mais eficaz do que modelos estatísticos clássicos.
- **Classificação > Regressão:** Modelos probabilísticos por classes oferecem previsões estrategicamente mais seguras do que a estimativa de um ponto nominal único.
- **Pré-processamento é crítico:** O uso de `MinMaxScaler` e o preenchimento de feriados internacionais via `ffill` são etapas indispensáveis para a convergência dos modelos.

---

## 🛠️ Stack Tecnológica

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-GRU%2FLSTM-red?logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MinMaxScaler-green?logo=scikitlearn)
![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-blueviolet)
![ta](https://img.shields.io/badge/ta-Technical%20Analysis-yellow)

---

## 👤 Autor

**Flavio Renan Sant'Anna**

---

> ⚠️ **Aviso:** Este projeto tem finalidade exclusivamente acadêmica e educacional. As previsões geradas não constituem recomendação de investimento.
