# Guia de Uso — `Random_Forest.py` (Random Forest para Previsão de Internações)

Este documento explica **como executar e configurar** a pipeline via **linha de comando (CLI)**, o que ela **gera** e traz **exemplos práticos** para o dia a dia (incluindo dicas para Windows/PowerShell).

---

## Sumário
- [Visão Geral](#visão-geral)
- [Pré‑requisitos](#pré-requisitos)
- [Como Executar](#como-executar)
- [Opções da CLI](#opções-da-cli)
- [Exemplos de Execução](#exemplos-de-execução)
- [Saídas Geradas](#saídas-geradas)
- [Logs e Acompanhamento](#logs-e-acompanhamento)
- [Explicabilidade (SHAP)](#explicabilidade-shap)
- [Dicas & Troubleshooting](#dicas--troubleshooting)
- [FAQ Rápido](#faq-rápido)
- [Contato / Próximos Passos](#contato--próximos-passos)

---

## Visão Geral

O script `Random_Forest.py` treina um **Random Forest** para prever **internações diárias por doenças respiratórias**.  
Ele realiza:
- preparação da série (índice diário, flags de missing, **opção de detrending** do alvo);
- seleção de features;
- busca de hiperparâmetros com validação temporal;
- avaliação em **treino** e **teste**;
- geração e **salvamento de gráficos e artefatos** (incluindo SHAP, se disponível).

---

## Pré‑requisitos

- **Python 3.10+** (recomendado)
- Pacotes principais:
  ```bash
  pip install numpy pandas scikit-learn matplotlib joblib
  ```
- (Opcional, para gráficos de explicabilidade **SHAP**):
  ```bash
  pip install shap
  ```

> **Dica** (Windows): execute os comandos no **PowerShell** ou **CMD** dentro do seu **ambiente virtual** (venv/conda).

---

## Como Executar

No diretório onde está o `Random_Forest.py`, rode:

```bash
python Random_Forest.py [opções]
```

Para ver todas as opções disponíveis:

```bash
python Random_Forest.py -h
```

---

## Opções da CLI

| Opção | Descrição | Padrão |
|---|---|---|
| `--detrend` / `--no-detrend` | Ativa/desativa **remoção da tendência** do alvo antes do treino. | `--detrend` |
| `--plots` / `--no-plots` | Abre janelas de gráfico (`--plots`) ou roda em modo **headless** (`--no-plots`) apenas salvando PNGs. | `--plots` |
| `--start YYYY-MM-DD` | Data **inicial** do recorte temporal (inclusiva). | `2013-12-31` |
| `--end YYYY-MM-DD` | Data **final** do recorte temporal (**exclusiva**). | `2025-01-01` |
| `--k-features N` | Número de **features selecionadas** após _permutation importance_. | `30` |
| `--gap-days N` | **Gap** entre treino e teste para evitar _leakage_ temporal. | `7` |
| `--window-size N` | Tamanho de janela se usar janelamento (para RF com lags). | `14` |
| `--test-size P` | Proporção do conjunto de teste (0‑1). | `0.20` |
| `--random-state S` | Semente aleatória. | `42` |
| `--output-dir PATH` | Pasta onde serão salvos **modelo, CSVs e gráficos**. | `random_forest_results` |

---

## Exemplos de Execução

### 1) Execução padrão (com detrending e gráficos)
```bash
python Random_Forest.py
```

### 2) Sem detrending (modelo aprende direto em `y`)
```bash
python Random_Forest.py --no-detrend
```

### 3) Sem abrir janelas de gráficos (headless; salva apenas PNGs)
> Recomendado para servidores/CI e para evitar problemas de backend gráfico.
```bash
python Random_Forest.py --no-plots
```

### 4) Definindo período de dados (recorte temporal)
```bash
python Random_Forest.py --start 2014-01-01 --end 2024-12-31
```

### 5) Ajustando principais parâmetros
```bash
python Random_Forest.py --k-features 30 --gap-days 7 --window-size 14 --test-size 0.2
```

### 6) Salvando em outra pasta
```bash
python Random_Forest.py --output-dir resultados_rf_novos
```

### 7) Exemplo completo (mix de opções)
```bash
python Random_Forest.py \
  --detrend \
  --no-plots \
  --start 2014-01-01 --end 2024-12-31 \
  --k-features 30 --gap-days 7 --window-size 14 --test-size 0.2 \
  --output-dir random_forest_results
```

> **Windows (PowerShell)** — use a crase para quebrar linhas:
> ```powershell
> python Random_Forest.py `
>   --detrend `
>   --no-plots `
>   --start 2014-01-01 --end 2024-12-31 `
>   --k-features 30 --gap-days 7 --window-size 14 --test-size 0.2 `
>   --output-dir random_forest_results
> ```

---

## Saídas Geradas

Tudo é salvo por padrão em **`random_forest_results/`** (pode ser alterado via `--output-dir`).

- **Modelo**
  - `rf_model.joblib` — modelo treinado (serializável, pronto para reuso).
- **Predições**
  - `predictions.csv` — DataFrame com colunas: `data_dia, y_true, y_pred, split` (train/test).
- **Bias sazonal (mês do ano, TESTE)**
  - `monthly_bias_moy_test.csv` — tabela com `mes, mes_label, bias_medio`.
  - `monthly_bias_moy_test.png` — gráfico do Bias médio por mês do ano.
- **Séries “Real x Previsto”**
  - `real_vs_previsto_train.png` — gráfico do conjunto de treinamento.
  - `real_vs_previsto_test.png` — gráfico do conjunto de teste.
- **Explicabilidade (se `shap` instalado)**
  - `shap_summary_beeswarm.png` — gráfico _beeswarm_ (impacto por amostra).
  - `shap_summary_bar.png` — barras com importância média \|SHAP\| por feature.

> **Nota:** As métricas exibidas no terminal (MAE, RMSE, R², sMAPE, WMAPE e **Bias**) são calculadas sempre na **escala original** após a recomposição da tendência (quando `--detrend` está ativo).

---

## Explicabilidade (SHAP)

Se o pacote **`shap`** estiver instalado, a pipeline gera automaticamente:
- **Beeswarm** (`shap_summary_beeswarm.png`) — mostra a distribuição do impacto das features por amostra (cores = valor da feature; posição = efeito no output).
- **Bar** (`shap_summary_bar.png`) — importância média \|SHAP\| por feature (ranking mais fácil de ler).

> Em cenários com muitos dados, a pipeline amostra um subconjunto para tornar o cálculo mais ágil (sem perder a leitura global).

---

## Dicas & Troubleshooting

- **Quero apenas salvar imagens, sem abrir janelas:** use `--no-plots`.  
  Esse modo utiliza backend **headless** e evita mensagens como _“main thread is not in main loop”_ do Tkinter.
- **SHAP não instalado:** os gráficos de SHAP são **pulados** com um aviso. Para ativar, Rode `pip install shap`.
- **Tempo de execução:** a busca de hiperparâmetros (RandomizedSearchCV) pode levar alguns minutos. Acompanhe pelos **logs**.
- **Reprodutibilidade:** use `--random-state 42` (ou outro inteiro fixo).  
- **Dados insuficientes no início da série:** o detrending causal usa apenas o passado; nos primeiros dias, a tendência é estimada por média acumulada — normal ter um período inicial descartado na modelagem.

---

## FAQ Rápido

**1) O que faz o `--detrend`?**  
Treina o RF **no resíduo** (alvo menos tendência causal) e, na hora de prever, **soma de volta** a tendência do dia. Ajuda a focar o modelo nas flutuações e a evitar _leakage_.

**2) As métricas são calculadas onde?**  
Sempre na **escala original** (após recompor a tendência), comparando `y_pred` vs `y_true`.

**3) O que significa o Bias?**  
É a média de `(y_pred - y_true)`. Se **positivo**, o modelo tende a **superestimar**; se **negativo**, **subestimar**.

**4) Por que dois gráficos de SHAP?**  
O _beeswarm_ mostra impacto **amostra a amostra**; o de **barras** resume a importância média por feature (mais objetivo para relatório).

---
