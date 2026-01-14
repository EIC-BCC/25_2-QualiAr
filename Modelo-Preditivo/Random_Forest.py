
#  SEÇÃO 1: IMPORTAÇÕES E CONFIGURAÇÕES GERAIS
from __future__ import annotations
from typing import Tuple, List, Dict

import argparse
import sys
from datetime import datetime
from pathlib import Path

from joblib import dump

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text

# URLs dos datasets (mantidas conforme seu projeto)
URL_INTERNACOES = (
    "https://raw.githubusercontent.com/EIC-BCC/25_2-QualiAr/refs/heads/main/data/Final_Datasets/INTERNACOES_DOENCA_RESP_RJ_SEAZONALITY_FEATURES.csv"
)
URL_QUALIAR = (
    "https://raw.githubusercontent.com/EIC-BCC/25_2-QualiAr/refs/heads/main/data/Final_Datasets/INTERNACOES_DOENCA_RESP_RJ_QUALIAR_FEATURES.csv"
)

GLOBAL_RANDOM_STATE = 42
np.random.seed(GLOBAL_RANDOM_STATE)


# LOG SIMPLES / ESTADOS DE EXECUÇÃO
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log_step(msg: str) -> None:
    """Imprime um passo com horário (para acompanhar o progresso no terminal)."""
    print(f"[{_now()}] {msg}", flush=True)


#  SEÇÃO 2: CARREGAMENTO E UNIFICAÇÃO DOS DADOS
def load_raw_data(
    url_internacoes: str = URL_INTERNACOES,
    url_qualiar: str = URL_QUALIAR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os datasets brutos de internações e sazonalidade a partir das URLs.

    Retorna:
        df_internacoes: DataFrame com informações de internações.
        df_qualiar: DataFrame com features sazonais/ambientais.
    """
    df_internacoes = pd.read_csv(url_internacoes, encoding="utf-8")
    df_qualiar = pd.read_csv(url_qualiar, encoding="utf-8")
    return df_internacoes, df_qualiar


def build_unified_dataset(
    df_internacoes: pd.DataFrame,
    df_qualiar: pd.DataFrame,
    date_col: str = "data_dia",
    target_col_name: str = "y"
) -> pd.DataFrame:
    """
    Unifica as bases pelo campo de data, ordena e garante alvo numérico.
    """
    
    df = (
        pd.merge(df_internacoes, df_qualiar, on=date_col, how="left")
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    
    cols_drop = ["ano", "mes", "dia", "target_y", "estacao", "estacao_id", "dow"]
    df.drop(columns=cols_drop, inplace=True, errors="ignore")
    df.rename(columns={"target_x": target_col_name}, inplace=True)
    df[target_col_name] = pd.to_numeric(df[target_col_name], errors="coerce").astype(float)
    
    cols = ['stl365_seasonal', 'stl365_season_amp']

    df[cols] = df[cols].shift(1)

    
    return df


def plot_internacoes_series(
    df_unificado: pd.DataFrame,
    date_col: str = "data_dia",
    target_col: str = "y",
    show: bool = True
) -> None:
    
    """Gráfico rápido da série de internações com média móvel."""
    
    ts = df_unificado.set_index(pd.to_datetime(df_unificado[date_col]))[target_col]
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts.values, label="Diário", alpha=0.6, linewidth=1)
    plt.plot(ts.index, ts.rolling(7, center=True).mean(), label="MM7", linewidth=2)
    plt.axhline(ts.mean(), linestyle="--", linewidth=1.5, label=f"Média ({ts.mean():.1f})")
    plt.title("Série Temporal de Internações")
    plt.xlabel("Data"); plt.ylabel("Internações/dia")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.gcf().autofmt_xdate()
    
    if show:
        plt.show()
        
    plt.close()

#  SEÇÃO 3: UTILITÁRIOS (MÉTRICAS E PESOS)
def make_recency_weights(
    index: pd.DatetimeIndex,
    low: float = 0.5,
    high: float = 1.5
) -> pd.Series:
    """
    Gera pesos crescentes no tempo:
        - Amostras mais recentes recebem peso maior.
        - Útil para dar mais importância aos anos finais.

    Retorna:
        Série com mesmo índice temporal e pesos normalizados no intervalo [low, high].
    """
    t = (index - index.min()).days.astype(float)
    w = low + (high - low) * (t - t.min()) / (t.max() - t.min() + 1e-9)
    return pd.Series(w, index=index).clip(lower=low, upper=high)

def smape(y_true, y_pred) -> float:
    """Calcula sMAPE (%) entre valores observados e previstos."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    )

def wmape(y_true, y_pred) -> float:
    """Calcula WMAPE (%) ponderando pelo total observado."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + 1e-9)

def bias(y_true, y_pred) -> float:
    """
    Calcula o Bias médio do modelo.

    Definição:
        Bias = média de (y_pred - y_true)

    Interpretação:
        - Bias < 0: modelo tende a subestimar (underprediction).
        - Bias > 0: modelo tende a superestimar (overprediction).
        - Bias ≈ 0: modelo sem viés sistemático relevante.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))

def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    """
    Calcula métricas padrão de avaliação do modelo.

    Retorna:
        dict com:MAE, RMSE, R2, sMAPE, WMAPE, Bias (médio de y_pred - y_true)
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "WMAPE": wmape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }


#  SEÇÃO 4: PREPARAÇÃO DA SÉRIE (DIÁRIO + TENDÊNCIA)
def make_daily_frame(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    ffill_limit: int = 14,
    categorical_cols: List[str] | None = None
) -> pd.DataFrame:
    """
    Cria uma série diária contínua a partir dos dados originais.

    Passos:
        - Converte coluna de data para datetime e ordena.
        - Reindexa para ter TODAS as datas (freq='D').
        - Cria colunas *_missing indicando pontos de falta.
        - Aplica forward-fill limitado nos preditores (não altera o alvo).

    Obs:
        - Forward-fill é limitado para evitar uso de informação muito distante.
    """
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.sort_values(date_col).set_index(date_col)
    full_idx = pd.date_range(data.index.min(), data.index.max(), freq="D")
    data = data.reindex(full_idx)

    if categorical_cols is None:
        categorical_cols = []
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cat_cols = [c for c in categorical_cols if c in data.columns]

    for c in num_cols + cat_cols:
        data[f"{c}_missing"] = data[c].isna().astype(int)
    for c in num_cols:
        data[c] = data[c].ffill(limit=ffill_limit)
    for c in cat_cols:
        data[c] = data[c].ffill(limit=min(3, ffill_limit))
    return data


def compute_causal_trend(y: pd.Series, window: int = 365, min_periods: int = 90) -> pd.Series:
    """
    Estima uma tendência lenta e CAUSAL para a série de internações.

    Estratégia:
    - Usa média móvel somente com dados do PASSADO (center=False),
      evitando vazamento de informação.
    - Para o início da série (antes da janela completa), usa média
      acumulada crescente como aproximação.

    Parâmetros:
        y: série original (já alinhada no índice diário).
        window: tamanho da janela para tendência de longo prazo (default: 365 dias).
        min_periods: mínimo de observações para começar a calcular a média móvel.

    Retorna:
        Série 'trend' alinhada com y.
    """
    y = y.astype(float)
    trend = y.rolling(window=window, min_periods=min_periods, center=False).mean()
    early_mask = trend.isna()
    if early_mask.any():
        trend[early_mask] = y.expanding(min_periods=1).mean()[early_mask]
    return trend


#  SEÇÃO 5: JANELAMENTO P/ MODELOS SEQUENCIAIS / RF COM LAGS
def apply_rolling_window(
    time_series_array: np.ndarray,
    initial_time_step: int,
    max_time_step: int,
    window_size: int,
    target_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria janelas temporais causais para previsão 1 passo à frente.

    Entradas:
        time_series_array: matriz [tempo, features].
        window_size: tamanho da janela (n dias passados).
        target_idx: índice da coluna do alvo em time_series_array.

    Saída:
        X: [amostras, window_size, n_features]
        y: [amostras] -> valor do alvo imediatamente após cada janela.
    """
    assert 0 <= target_idx < time_series_array.shape[1]
    assert initial_time_step >= 0
    assert max_time_step >= initial_time_step

    start = initial_time_step
    sub_windows = (
        start
        + np.expand_dims(np.arange(window_size), 0)
        + np.expand_dims(np.arange(max_time_step + 1), 0).T
    )
    X = time_series_array[sub_windows]
    y = time_series_array[window_size:(max_time_step + window_size + 1), target_idx]
    if np.any(np.isnan(y)):
        raise ValueError("Há valores NaN em y após criação das janelas.")
    return X, y

def rolling_to_tabular(
    df_windowed: pd.DataFrame,
    window_size: int,
    target_col: str,
    date_index: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Converte uma série multivariada em:

        - X_flat (2D) para modelos tabulares (RF).
        - y (1D) alvo.
        - X_seq (3D) para modelos sequenciais (LSTM).
        - seq_cols: lista de colunas na ordem usada nos canais.

    Estratégia:
        - Usa apply_rolling_window para gerar janelas.
        - Achata as janelas para RF (lags explodidos).
        - Mantém tensor 3D para LSTM.
    """
    cols = df_windowed.columns.tolist()
    target_idx = cols.index(target_col)
    A = df_windowed.to_numpy()
    max_ts = len(df_windowed) - window_size - 1
    X3d, y = apply_rolling_window(A, 0, max_ts, window_size, target_idx)
    ns, w, nf = X3d.shape
    X2d = X3d.reshape(ns, w * nf)
    names = [f"{c}_t-{lag}" for lag in range(1, window_size + 1) for c in cols]
    idx = date_index[window_size:(max_ts + window_size + 1)]
    X_flat = pd.DataFrame(X2d, columns=names, index=idx)
    return X_flat, y, X3d, cols


#  SEÇÃO 6: SELEÇÃO DE FEATURES, TREINO RF E PREDIÇÃO
def select_top_features(
    X: pd.DataFrame,
    y: np.ndarray,
    pre: ColumnTransformer,
    n_splits: int = 4,
    random_state: int = 42,
    k: int = 8,
    gap_days: int = 7
) -> List[str]:
    """
    Seleciona as top-k features com base em importância por permutação.

    Estratégia:
        - Treina um Random Forest com validação temporal (TimeSeriesSplit).
        - Calcula permutation importance nos últimos folds (mais recentes).
        - Retorna as k features mais importantes.

    Observação importante:
        - NÃO aplicamos log1p no alvo aqui, pois em cenários com detrending
          o alvo pode assumir valores negativos (resíduos). Usar log1p
          geraria NaN e quebraria o ajuste.
    """
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    except TypeError: 
        tscv = TimeSeriesSplit(n_splits=n_splits)

    base_rf = RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_leaf=10, min_samples_split=10,
        max_features="sqrt", random_state=random_state, n_jobs=-1
    )

    pipe = Pipeline([("pre", pre), ("rf", base_rf)])
    importances_list = []
    folds = list(tscv.split(X))

    for tr_idx, va_idx in folds[-min(3, len(folds)):]:
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        imp = permutation_importance(
            pipe, X.iloc[va_idx], y[va_idx],
            n_repeats=10, random_state=random_state,
            scoring="neg_mean_absolute_error"
        )
        importances_list.append(pd.Series(imp.importances_mean, index=X.columns))

    mean_importances = pd.concat(importances_list, axis=1).mean(axis=1)
    return mean_importances.sort_values(ascending=False).head(k).index.tolist()


def tune_and_fit_rf_simplified(
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    n_splits: int = 5,
    random_state: int = 42,
    use_log_transform: bool = True,
    gap_days: int = 7
) -> Pipeline:
    """
    Cria pipeline (preprocessamento + RF envolto em TransformedTargetRegressor)
    e executa RandomizedSearchCV com validação temporal.

    Obs:
        - Aplica log1p no alvo quando 'use_log_transform=True'.
        - Usa pesos de recência no ajuste final.
        - TODO: expor espaço de busca como parâmetro externo.
    """
    base_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=10,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state
    )

    if use_log_transform:
        ttr = TransformedTargetRegressor(
            regressor=base_rf,
            func=np.log1p,
            inverse_func=np.expm1
        )
    else:
        ttr = TransformedTargetRegressor(
            regressor=base_rf,
            transformer=FunctionTransformer(validate=False)
        )

    pipe = Pipeline([("pre", preprocessor), ("rf", ttr)])

    param_dist = {
        "rf__regressor__n_estimators": [300, 400, 600, 800, 1200],
        "rf__regressor__max_depth": [None, 6, 8, 10, 12],
        "rf__regressor__min_samples_leaf": [5, 10, 20, 40],
        "rf__regressor__min_samples_split": [5, 10, 20, 40],
        "rf__regressor__max_features": ["sqrt", 0.3, 0.5, 0.8],
        "rf__regressor__max_samples": [0.5, 0.7, 0.9, None], 
    }

    recency_weights = make_recency_weights(X.index, low=0.7, high=2.0).values
    
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    except TypeError:
        tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=24,                 # ↑ mais tentativas costuma ajudar
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=1,
    )

    search.fit(X, y, rf__sample_weight=recency_weights)
    return search.best_estimator_


def print_one_tree(model: Pipeline, feature_names: List[str], estimator_idx: int = 0) -> None:
    """Imprime uma das árvores da floresta (inspeção)."""
    rf = model.named_steps["rf"].regressor_
    tree = rf.estimators_[estimator_idx]
    txt = export_text(tree, feature_names=feature_names[:tree.n_features_in_])
    print(txt)


def predict_and_plot_rf(
    model: Pipeline,
    X: pd.DataFrame,
    y_true: np.ndarray,
    title: str,
    trend: pd.Series | None = None,
    show_plot: bool = True,
    output_path: Path | None = None, 
) -> Dict:
    """
    Gera previsões com RF, com ou sem detrending.

    - Se 'trend' for fornecido, assume que o modelo previu o componente
      detrendido e soma a tendência de volta.
    - Caso contrário, usa diretamente a saída do modelo.
    """
    y_hat_model = model.predict(X)
    if trend is not None:
        trend_aligned = trend.loc[X.index]
        y_hat = y_hat_model + trend_aligned.values
    else:
        y_hat = y_hat_model

    y_hat = np.clip(y_hat, 0, None)
    metrics = evaluate_predictions(y_true, y_hat)

    df_plot = pd.DataFrame({"y_true": y_true, "y_pred": y_hat}, index=X.index)
    plt.figure(figsize=(12, 5))
    plt.plot(df_plot.index, df_plot["y_true"], label="Observado", lw=1.2)
    plt.plot(df_plot.index, df_plot["y_pred"], label="Previsto (RF)", lw=1.2)
    plt.title(title); plt.xlabel("Data"); plt.ylabel("Internações / dia")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close()

    return {"metrics": metrics, "pred_df": df_plot}

#  SEÇÃO 7: PREPARAÇÃO PARA RF (SPLIT, PREPROCESS, FEATURE SELECTION)
def prepare_rolling_dataset(
    df_unificado: pd.DataFrame,
    date_col: str = "data_dia",
    target_col: str = "y",
    categorical_cols: List[str] = ("Qualidade_do_Ar",),
    window_size: int = 14,
    test_size: float = 0.2,
    gap_days: int = 7,
    k_features: int = 12,
    random_state: int = 42,
    use_windowing_for_rf: bool = False,
    detrend_target: bool = True,
) -> Dict:
    """
    Prepara dados apenas para o caminho TABULAR (Random Forest).

    Passos:
        1. Garante série diária contínua (make_daily_frame).
        2. Define X e y:
            - Opção padrão (use_windowing_for_rf=False):
                usa diretamente as features já criadas (lags, médias, etc.).
            - Opção alternativa:
                aplica janelamento (rolling_to_tabular) para gerar lags explodidos.
        3. Separa treino e teste respeitando a ordem temporal e aplicando um gap.
        4. Monta pré-processador (numéricas + categóricas).
        5. Faz seleção de top-k features via permutation importance.

    Retorna:
        dict com:
            X_train, y_train, X_test, y_test,
            preprocessor (ColumnTransformer),
            features_kept,
            idx_train, idx_test.
    """
    base = make_daily_frame(
        df_unificado.copy(), date_col=date_col, target_col=target_col,
        ffill_limit=14, categorical_cols=list(categorical_cols)
    ).sort_index()

    if detrend_target:
        base["trend_causal_365"] = compute_causal_trend(base[target_col])
        base["y_detrended"] = base[target_col] - base["trend_causal_365"]
        target_used = "y_detrended"
    else:
        base["trend_causal_365"] = np.nan
        target_used = target_col

    if use_windowing_for_rf:
        feat_cols_for_rf = [c for c in base.columns if c not in (target_col, target_used, "trend_causal_365")]
        X_full_tab, y_full_tab, _, _ = rolling_to_tabular(
            base[feat_cols_for_rf + [target_used]], window_size=window_size,
            target_col=target_used, date_index=base.index
        )
        trend_series = None if not detrend_target else base["trend_causal_365"].iloc[window_size:-1]
        y_full_orig = base[target_col].to_numpy()[window_size:-1]
    else:
        mask_y = base[target_used].notna()
        base_model = base.loc[mask_y, :]
        feat_cols_for_rf = [c for c in base_model.columns
                            if c not in (target_col, target_used, "trend_causal_365")]
        X_full_tab = base_model[feat_cols_for_rf].copy()
        y_full_tab = base_model[target_used].astype(float).to_numpy()
        trend_series = None if not detrend_target else base_model["trend_causal_365"]
        y_full_orig = base_model[target_col].astype(float).to_numpy()

    # Split temporal com gap
    n = len(X_full_tab)
    cut = int(np.floor((1 - test_size) * n))
    train_idx = slice(0, max(cut - gap_days, 0))
    test_idx = slice(cut, n)

    X_train, X_test = X_full_tab.iloc[train_idx], X_full_tab.iloc[test_idx]
    y_train, y_test = y_full_tab[train_idx], y_full_tab[test_idx]

    if detrend_target:
        trend_train = trend_series.iloc[train_idx]
        trend_test = trend_series.iloc[test_idx]
        y_train_orig = y_full_orig[train_idx]
        y_test_orig = y_full_orig[test_idx]
    else:
        trend_train = trend_test = None
        y_train_orig, y_test_orig = y_train, y_test

    # Preprocess e seleção de features
    cat_cols = [c for c in X_train.columns if c in set(categorical_cols)]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre_tmp = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )

    topk = select_top_features(X_train, y_train, pre=pre_tmp, n_splits=4,
                               random_state=random_state, k=k_features)
    features_kept = list(topk)

    X_train_sel, X_test_sel = X_train[features_kept].copy(), X_test[features_kept].copy()

    cat_cols_sel = [c for c in features_kept if c in set(categorical_cols)]
    num_cols_sel = [c for c in features_kept if c not in cat_cols_sel]

    pre_sel = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols_sel),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols_sel),
        ],
        remainder="drop"
    )

    return {
        "X_train": X_train_sel, "y_train": y_train,
        "X_test": X_test_sel, "y_test": y_test,
        "y_train_orig": y_train_orig, "y_test_orig": y_test_orig,
        "trend_train": trend_train, "trend_test": trend_test,
        "preprocessor": pre_sel, "features_kept": features_kept,
        "idx_train": X_train_sel.index, "idx_test": X_test_sel.index,
        "detrend_target": detrend_target,
    }


#  SEÇÃO 8: PÓS-PROCESSO — SALVAR PREDIÇÕES E PLOTAR BIAS MENSAL
def save_predictions_csv(
    train_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    output_dir: Path
) -> Path:
    """
    Salva um CSV com (data, y_true, y_pred, split) para train e test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train = train_pred_df.copy()
    train["data_dia"] = pd.to_datetime(train.index)
    train["split"] = "train"

    test = test_pred_df.copy()
    test["data_dia"] = pd.to_datetime(test.index)
    test["split"] = "test"

    out = pd.concat([train, test], axis=0, ignore_index=True)
    out = out[["data_dia", "y_true", "y_pred", "split"]].sort_values("data_dia")

    csv_path = output_dir / "predictions.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def compute_monthly_bias_moy(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o Bias médio por MÊS DO ANO (sazonalidade de Bias).
    Retorna tabela com colunas: mes (1..12), bias_medio.
    """
    df = pred_df.copy()
    df["data_dia"] = pd.to_datetime(df.index)
    df["erro"] = df["y_pred"] - df["y_true"]
    grp = df.groupby(df["data_dia"].dt.month)["erro"].mean().rename("bias_medio")
    out = grp.reset_index().rename(columns={"data_dia": "mes"})
    month_labels = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
                    7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
    out["mes_label"] = out["mes"].map(month_labels)
    return out[["mes", "mes_label", "bias_medio"]]


def plot_monthly_bias_moy(
    monthly_bias_df: pd.DataFrame,
    title: str = "Bias médio por mês (mês do ano)",
    output_path: Path | None = None,
    show: bool = True
) -> None:
    """
    Plota o Bias médio por mês do ano e opcionalmente salva em PNG.
    """
    x = monthly_bias_df["mes"]
    y = monthly_bias_df["bias_medio"]
    labels = monthly_bias_df["mes_label"]

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xticks(x, labels)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(title); plt.xlabel("Mês do ano"); plt.ylabel("Bias médio (y_pred - y_true)")
    plt.grid(alpha=0.3); plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

#  SEÇÃO 8.1: INTERPRETABILIDADE — SHAP VALUES (RF)
def _get_feature_names_from_preprocessor(pre: ColumnTransformer) -> List[str]:
    """
    Tenta extrair nomes das features pós-ColumnTransformer (incluindo OHE).
    Compatível com versões recentes do scikit-learn; tem fallback manual.
    """
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        pass

    feature_names = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder" and trans == "drop":
            continue

        # Se for um Pipeline, pegamos o último estágio
        if isinstance(trans, Pipeline):
            last_step = trans.steps[-1][1]
            # OneHotEncoder expande nomes
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    fn = last_step.get_feature_names_out(cols)
                    feature_names.extend(list(fn))
                    continue
                except Exception:
                    pass
            # Sem método: devolve os nomes originais
            feature_names.extend(list(cols))
        else:
            # Transformador simples
            if hasattr(trans, "get_feature_names_out"):
                try:
                    fn = trans.get_feature_names_out(cols)
                    feature_names.extend(list(fn))
                    continue
                except Exception:
                    pass
            feature_names.extend(list(cols))
    return feature_names


def generate_and_save_shap_plots(
    model: Pipeline,
    X_train: pd.DataFrame,
    output_dir: Path,
    max_samples: int = 2000,
    max_display: int = 25,
) -> Dict[str, Path]:
    """
    Gera e salva gráficos de SHAP (beeswarm + bar) para o RandomForestRegressor
    dentro do pipeline (pre + TTR + RF).

    Retorna dict com caminhos dos arquivos gerados.
    """
    outputs: Dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _SHAP_AVAILABLE:
        log_step("[WARN] Pacote 'shap' não encontrado. Pulei os gráficos de SHAP. "
                 "Instale com: pip install shap")
        return outputs

    # Extrai objetos do pipeline
    pre = model.named_steps["pre"]
    rf  = model.named_steps["rf"].regressor_  # RandomForestRegressor dentro do TTR

    # Transforma o X de treino como o modelo viu
    X_pre = pre.transform(X_train)  # pode ser numpy ou scipy.sparse
    try:
        import scipy.sparse as sp
        if sp.issparse(X_pre):
            X_pre = X_pre.toarray()
    except Exception:
        pass

    # Nomes das colunas após o preprocessor (inclui OHE)
    try:
        feat_names = _get_feature_names_from_preprocessor(pre)
    except Exception:
        feat_names = [f"f{i}" for i in range(X_pre.shape[1])]

    # Amostragem (para não pesar)
    if X_pre.shape[0] > max_samples:
        idx = np.linspace(0, X_pre.shape[0] - 1, num=max_samples, dtype=int)
        X_sample = X_pre[idx]
    else:
        X_sample = X_pre

    # Explainer (mantém compatibilidade de versões)
    try:
        # API nova
        explainer = shap.Explainer(rf, X_sample)
        sv = explainer(X_sample)
        shap_values = sv.values
    except Exception:
        # API clássica
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)

    # --- Beeswarm (summary) ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names,
                      show=False, max_display=max_display)
    beeswarm_path = output_dir / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    outputs["beeswarm"] = beeswarm_path

    # --- Barras (importância média |SHAP|) ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names,
                      plot_type="bar", show=False, max_display=max_display)
    bar_path = output_dir / "shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    outputs["bar"] = bar_path

    log_step(f"Gráficos SHAP salvos em: {beeswarm_path.name}, {bar_path.name}")
    return outputs

#  SEÇÃO 9: EXECUÇÃO COMPLETA — RF + SALVAMENTO
def run_pipeline_rf_rolling(
    df_unificado: pd.DataFrame,
    date_col: str = "data_dia",
    target_col: str = "y",
    categorical_cols: List[str] = ("Qualidade_do_Ar",),
    window_size: int = 14,
    test_size: float = 0.2,
    gap_days: int = 7,
    k_features: int = 12,
    random_state: int = 42,
    detrend_target: bool = True,
    show_plots: bool = True,
    output_dir: Path | None = None,  
) -> Dict:
    """
    Executa a pipeline completa do Random Forest:

        1. Prepara dados (prepare_rolling_dataset).
        2. Faz tuning simplificado do RF com validação temporal.
        3. Gera previsões em treino e teste, com gráficos.
        4. Exibe uma árvore para interpretação.

    Retorna:
        dict com modelo, features usadas, métricas e DataFrames de previsão.
    """
    log_step("Preparando dataset (diário, flags de missing e detrending opcional)...")
    prepared = prepare_rolling_dataset(
        df_unificado=df_unificado, date_col=date_col, target_col=target_col,
        categorical_cols=categorical_cols, window_size=window_size,
        test_size=test_size, gap_days=gap_days, k_features=k_features,
        random_state=random_state, use_windowing_for_rf=False,
        detrend_target=detrend_target,
    )

    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_test,  y_test  = prepared["X_test"],  prepared["y_test"]
    y_train_orig, y_test_orig = prepared["y_train_orig"], prepared["y_test_orig"]
    trend_train, trend_test = prepared["trend_train"], prepared["trend_test"]
    pre_sel, topk = prepared["preprocessor"], prepared["features_kept"]
    detrended = prepared["detrend_target"]

    log_step(f"Selecionadas {len(topk)} features. Iniciando busca de hiperparâmetros (RF)...")
    best_model = tune_and_fit_rf_simplified(
        X_train, y_train, preprocessor=pre_sel, n_splits=5,
        random_state=random_state, use_log_transform=not detrended,
        gap_days=gap_days
    )

    # Caminhos de saída para os plots Real x Previsto
    train_plot_path = (output_dir / "real_vs_previsto_train.png") if output_dir else None
    test_plot_path  = (output_dir / "real_vs_previsto_test.png")  if output_dir else None

    log_step("Avaliando no conjunto de TREINO...")
    train_res = predict_and_plot_rf(
        best_model, X_train, y_train_orig,
        "Real x Previsto — RF (Treino)",
        trend=trend_train if detrended else None,
        show_plot=show_plots,
        output_path=train_plot_path,   
    )
    log_step("Avaliando no conjunto de TESTE...")
    test_res = predict_and_plot_rf(
        best_model, X_test, y_test_orig,
        "Real x Previsto — RF (Teste)",
        trend=trend_test if detrended else None,
        show_plot=show_plots,
        output_path=test_plot_path,    
    )

    shap_paths = {}
    if output_dir is not None:
        log_step("Gerando gráficos de SHAP (pode levar alguns segundos)...")
        shap_paths = generate_and_save_shap_plots(best_model, X_train, output_dir)

    return {
        "model": best_model,
        "features_kept": topk,
        "train_metrics": train_res["metrics"],
        "test_metrics": test_res["metrics"],
        "train_pred_df": train_res["pred_df"],
        "test_pred_df": test_res["pred_df"],
        "X_train": X_train,
        "X_test": X_test,
        "shap_plots": shap_paths,
    }


#  SEÇÃO 10: CLI (ARGPARSE) E RUNNER
def main_cli(argv: List[str] | None = None) -> int:
    """
    CLI para executar a pipeline via terminal.

    Exemplos:
        python pipeline.py --detrend                       # com detrending (default)
        python pipeline.py --no-detrend                    # sem detrending
        python pipeline.py --no-plots                      # sem abrir janelas de gráfico
        python pipeline.py --start 2014-01-01 --end 2024-12-31
        python pipeline.py --k-features 30 --window-size 14 --gap-days 7
        python pipeline.py --output-dir random_forest_results_v2
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de previsão de internações (Random Forest) com opção de detrending."
    )
    # Flags / opções
    detrend_group = parser.add_mutually_exclusive_group()
    detrend_group.add_argument("--detrend", dest="detrend", action="store_true",
                               help="Ativa detrending do alvo (padrão).")
    detrend_group.add_argument("--no-detrend", dest="detrend", action="store_false",
                               help="Desativa detrending do alvo.")
    parser.set_defaults(detrend=True)

    parser.add_argument("--window-size", type=int, default=14, help="Tamanho da janela para janelamento (se usado).")
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporção para teste (0-1).")
    parser.add_argument("--gap-days", type=int, default=7, help="Gap entre treino e teste.")
    parser.add_argument("--k-features", type=int, default=30, help="Top-k features após seleção.")
    parser.add_argument("--random-state", type=int, default=GLOBAL_RANDOM_STATE, help="Seed aleatória.")
    parser.add_argument("--start", type=str, default="2013-12-31", help="Data inicial (YYYY-MM-DD) do recorte.")
    parser.add_argument("--end", type=str, default="2025-01-01", help="Data final (YYYY-MM-DD) do recorte (exclusiva).")
    parser.add_argument("--output-dir", type=str, default="random_forest_results",
                        help="Diretório para salvar modelo e artefatos.")
    plots_group = parser.add_mutually_exclusive_group()
    plots_group.add_argument("--plots", dest="plots", action="store_true", help="Exibe gráficos.")
    plots_group.add_argument("--no-plots", dest="plots", action="store_false", help="Não exibe gráficos.")
    parser.set_defaults(plots=True)

    args = parser.parse_args(argv)

    if not args.plots:
        matplotlib.use("Agg")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================= EXECUÇÃO =========================
    log_step("Carregando dados brutos...")
    df_internacoes, df_qualiar = load_raw_data()

    log_step("Unificando datasets e preparando recorte temporal...")
    df_unificado = build_unified_dataset(df_internacoes, df_qualiar)
    
    # Recorte temporal
    df_unificado = df_unificado[
        (df_unificado["data_dia"] > args.start) &
        (df_unificado["data_dia"] < args.end)
    ].reset_index(drop=True)

    if args.plots:
        log_step("Plotando série de internações (visão geral)...")
        plot_internacoes_series(df_unificado, show=True)

    # Executa pipeline RF
    log_step(f"Iniciando pipeline RF (detrend={args.detrend})...")
    out_rf = run_pipeline_rf_rolling(
        df_unificado,
        date_col="data_dia", target_col="y",
        window_size=args.window_size, test_size=args.test_size,
        gap_days=args.gap_days, k_features=args.k_features,
        random_state=args.random_state, detrend_target=args.detrend,
        show_plots=args.plots,
        output_dir=output_dir,  
    )

    # Relatório básico
    print("\n[RF] Features usadas:", out_rf["features_kept"])
    print("[RF] Métricas Treino:", out_rf["train_metrics"])
    print("[RF] Métricas Teste :", out_rf["test_metrics"])

    # Pasta de resultados
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Modelo
    log_step("Salvando modelo treinado...")
    model_path = output_dir / "rf_model.joblib"
    dump(out_rf["model"], model_path)
    log_step(f"Modelo salvo em: {model_path.resolve()}")

    # 2) Predições (train + test)
    log_step("Salvando predições (train + test) em CSV...")
    preds_csv = save_predictions_csv(out_rf["train_pred_df"], out_rf["test_pred_df"], output_dir)
    log_step(f"Predições salvas em: {preds_csv.resolve()}")

    # 3) Bias médio mensal do TESTE (mês do ano)
    log_step("Calculando bias médio mensal (mês do ano) no TESTE...")
    monthly_bias_df = compute_monthly_bias_moy(out_rf["test_pred_df"])
    monthly_bias_csv = output_dir / "monthly_bias_moy_test.csv"
    monthly_bias_df.to_csv(monthly_bias_csv, index=False, encoding="utf-8")
    log_step(f"Tabela de bias mensal salva em: {monthly_bias_csv.resolve()}")

    monthly_bias_png = output_dir / "monthly_bias_moy_test.png"
    plot_monthly_bias_moy(monthly_bias_df, output_path=monthly_bias_png, show=args.plots)
    log_step(f"Gráfico de bias mensal salvo em: {monthly_bias_png.resolve()}")

    log_step("Pipeline finalizada com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
