"""am1_util: funções de apoio à Entrega 2 de CIN0209 Aprendizado de Máquina 1.

Graduação em Inteligência Artificial, UFPE/CIn, 2026.2.

Este módulo funciona como uma pequena biblioteca: as funções estão prontas e
testadas. O trabalho da dupla é aplicá-las ao próprio problema, escolher e
justificar as grades de hiperparâmetros e a métrica principal, e interpretar
os resultados. Usar o módulo não dispensa entender o que cada função faz: na
arguição, qualquer decisão do notebook pode ser questionada.

Dependências: apenas o ambiente padrão do Google Colab (numpy, pandas,
scikit-learn, scipy, matplotlib). O LightGBM só é importado se a dupla pedir a
família opcional ``"lightgbm"`` (extensão bônus).

Protocolo que o módulo implementa
---------------------------------
1. A divisão TREINO/VALIDAÇÃO/TESTE da Entrega 1 é mantida, com a mesma
   semente. Treino + validação formam o conjunto de DESENVOLVIMENTO.
2. Toda escolha (pré-processamento, hiperparâmetros, família campeã) é feita
   por validação cruzada estratificada (k = 5) dentro do desenvolvimento.
3. O pré-processamento fica DENTRO de um ``Pipeline``: em cada fold, a
   imputação, a padronização e a codificação são ajustadas apenas na parte de
   treino do fold. Isso evita vazamento de informação da parte de validação.
4. O teste é usado uma única vez, no fim, com os campeões reajustados no
   desenvolvimento inteiro (``test_table`` registra e protege esse uso).

Famílias disponíveis (chaves de ``FAMILIES``): ``"arvore"``, ``"knn"``,
``"naive_bayes"``, ``"logistica"``, ``"perceptron"``, ``"mlp"``. A chave
opcional ``"lightgbm"`` existe apenas para a extensão bônus.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import validation_curve as _sk_validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

__version__ = "2026.2.0"

__all__ = [
    "EXTENSION_GRIDS",
    "FAMILIES",
    "FAMILY_LABELS",
    "SUGGESTED_CURVES",
    "SUGGESTED_GRIDS",
    "best_model_params",
    "build_pipeline",
    "compare_cv_test",
    "compute_metrics",
    "confusion_table",
    "curve_point",
    "cv_fold_scores",
    "cv_search",
    "cv_summary",
    "development_set",
    "generalization_gap",
    "load_split",
    "make_cv",
    "plot_confusion",
    "plot_generalization_gap",
    "plot_validation_curve",
    "preprocessor",
    "recall_per_class",
    "refit_with",
    "save_split",
    "select_complexity",
    "split_from_e1",
    "split_from_frames",
    "split_report",
    "stratified_subsample",
    "test_table",
    "validation_curve",
]

# ---------------------------------------------------------------------------
# Famílias, grades e curvas sugeridas
# ---------------------------------------------------------------------------

FAMILIES = ("arvore", "knn", "naive_bayes", "logistica", "perceptron", "mlp")
"""Famílias obrigatórias da Entrega 2, na ordem em que aparecem no curso."""

FAMILY_LABELS = {
    "arvore": "Árvore de decisão",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes (Gaussiano)",
    "logistica": "Regressão logística",
    "perceptron": "Perceptron (calibrado)",
    "mlp": "MLP",
    "lightgbm": "LightGBM (extensão)",
}

SUGGESTED_GRIDS = {
    "arvore": {
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_leaf": [1, 5, 20, 50],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 11, 21, 41],
        "model__weights": ["uniform", "distance"],
    },
    "naive_bayes": {
        "model__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
    },
    "logistica": {
        "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "perceptron": [
        {"model__estimator__penalty": [None]},
        {
            "model__estimator__penalty": ["l2"],
            "model__estimator__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        },
    ],
    "mlp": {
        "model__hidden_layer_sizes": [(32,), (64,)],
        "model__alpha": [1e-4, 1e-2, 1e-1, 1.0],
    },
}
"""Grades SUGERIDAS (ponto de partida, não resposta). Chaves no formato do
Pipeline (``model__<parâmetro>``). A dupla deve ajustar as grades ao próprio
problema e justificar: por exemplo, ampliar a faixa quando o melhor valor cai
na borda da grade. A regra vale para hiperparâmetros numéricos que admitem
extensão; quando o ótimo cai em None, num valor mínimo natural ou num
hiperparâmetro categórico, basta comentar."""

SUGGESTED_CURVES = {
    "arvore": ("max_depth", [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, None], "lower"),
    "knn": ("n_neighbors", [1, 3, 5, 7, 11, 15, 21, 31, 51, 81], "higher"),
    "mlp": ("alpha", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], "higher"),
}
"""Curvas de validação obrigatórias: família -> (parâmetro, valores, direção
do modelo mais simples). ``"lower"`` significa que valores menores geram
modelos mais simples (árvore rasa); ``"higher"`` significa o contrário (mais
vizinhos ou mais regularização produzem fronteiras mais suaves)."""

EXTENSION_GRIDS = {
    "lightgbm": {
        "model__num_leaves": [15, 31],
        "model__learning_rate": [0.05, 0.1],
    },
}
"""Grade sugerida para a extensão bônus opcional (mesmo protocolo)."""

_DEFAULT_SCALE = {
    "arvore": False,
    "knn": True,
    "naive_bayes": True,
    "logistica": True,
    "perceptron": True,
    "mlp": True,
    "lightgbm": False,
}

_CALIBRATOR_PARAMS = {"method", "cv", "ensemble", "n_jobs", "estimator"}


# ---------------------------------------------------------------------------
# Divisão: salvar, carregar, reconstruir e verificar
# ---------------------------------------------------------------------------


def _as_index_array(idx, name):
    arr = np.asarray(idx)
    if arr.ndim != 1:
        raise ValueError(f"{name}: os índices devem formar um vetor unidimensional.")
    if arr.size and arr.dtype.kind not in "iu":
        raise TypeError(
            f"{name}: esperados índices POSICIONAIS inteiros (0 a n-1). "
            "Se vocês têm DataFrames da E1, usem split_from_frames."
        )
    return arr.astype(np.int64)


def _fingerprint(y):
    """Resumo (hash) do alvo na ordem atual, para detectar reordenação."""
    values = pd.Series(np.asarray(y)).astype(str).to_numpy()
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()
    return digest[:16]


def _validate_split(split, n_total=None):
    parts = {k: split[k] for k in ("train", "val", "test")}
    for k, arr in parts.items():
        if arr.size and arr.min() < 0:
            raise ValueError(f"Índices negativos em '{k}'.")
        if len(np.unique(arr)) != len(arr):
            raise ValueError(f"Índices repetidos dentro de '{k}'.")
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        inter = np.intersect1d(parts[a], parts[b])
        if inter.size:
            raise ValueError(
                f"Os conjuntos '{a}' e '{b}' compartilham {inter.size} registros: "
                "a divisão não é válida."
            )
    if n_total is not None:
        top = max((arr.max() for arr in parts.values() if arr.size), default=-1)
        if top >= n_total:
            raise ValueError(
                f"Há índice {top} fora do intervalo dos dados (n = {n_total}). "
                "Confiram se X e y são exatamente os da E1, na mesma ordem."
            )


def save_split(path, train_idx, val_idx, test_idx, seed, y=None, notes=""):
    """Salva a divisão da E1 em disco (``.json`` ou ``.npz``).

    Parâmetros
    ----------
    path : str ou Path
        Arquivo de destino. No Colab, salvem no Google Drive (por exemplo,
        ``/content/drive/MyDrive/CIN0209_projeto/divisao_e1.json``): arquivos
        em ``/content`` são apagados quando a sessão termina.
    train_idx, val_idx, test_idx : sequências de inteiros
        Índices POSICIONAIS (0 a n-1) de cada conjunto, sobre os dados na
        ordem exata em que ficaram após a preparação da E1.
    seed : int
        Semente usada na divisão da E1 (será reutilizada na validação cruzada).
    y : array-like, opcional
        Alvo completo, na mesma ordem. Se informado, grava o tamanho total, a
        contagem por classe e um resumo (hash) que ``development_set`` usa
        para detectar se os dados mudaram de ordem ou de conteúdo.
    notes : str
        Observação livre (ex.: como a divisão foi gerada).

    Retorna
    -------
    dict
        A divisão no formato usado pelo módulo (chaves ``train``, ``val``,
        ``test``, ``seed``, ``n_total``, ``fingerprint``, ``notes``).
    """
    split = {
        "train": _as_index_array(train_idx, "train_idx"),
        "val": _as_index_array(val_idx, "val_idx"),
        "test": _as_index_array(test_idx, "test_idx"),
        "seed": int(seed),
        "n_total": len(y) if y is not None else None,
        "fingerprint": _fingerprint(y) if y is not None else None,
        "notes": str(notes),
    }
    _validate_split(split, split["n_total"])
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: split[k] for k in ("seed", "n_total", "fingerprint", "notes")}
    if path.suffix.lower() == ".npz":
        np.savez(
            path,
            train=split["train"],
            val=split["val"],
            test=split["test"],
            meta=np.array(json.dumps(meta)),
        )
    elif path.suffix.lower() == ".json":
        payload = dict(meta)
        payload.update({k: split[k].tolist() for k in ("train", "val", "test")})
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        raise ValueError("Use extensão .json ou .npz no caminho da divisão.")
    return split


def load_split(path):
    """Carrega uma divisão salva por ``save_split``.

    Retorna o mesmo dicionário devolvido por ``save_split`` (índices como
    arrays de inteiros).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de divisão não encontrado: {path}. No Colab, confiram se o "
            "Drive foi montado e se o caminho está correto."
        )
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            parts = {k: data[k].astype(np.int64) for k in ("train", "val", "test")}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parts = {k: np.asarray(payload[k], dtype=np.int64) for k in ("train", "val", "test")}
        meta = {k: payload.get(k) for k in ("seed", "n_total", "fingerprint", "notes")}
    split = dict(parts)
    split.update(meta)
    _validate_split(split, split.get("n_total"))
    return split


def split_from_e1(y, test_size, val_size, seed, stratify=True, notes=""):
    """Reconstrói a divisão da E1 refazendo as DUAS chamadas de ``train_test_split``.

    Supõe o procedimento mais comum na E1::

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=seed)

    O sorteio de ``train_test_split`` depende apenas do número de linhas, do
    alvo (quando estratificado) e da semente, e não do conteúdo de X. Por isso,
    repetir as chamadas sobre ``np.arange(n)`` devolve exatamente as posições
    de cada conjunto. Informem os valores LITERAIS usados na E1 (por exemplo,
    ``val_size=0.25`` se a segunda chamada recebeu 0.25, mesmo que isso
    represente 20% do total).

    O alvo deve ter exatamente a mesma codificação da E1 (mesmos rótulos,
    por exemplo texto ou 0/1): recodificar as classes muda o sorteio, sem
    alterar tamanhos nem proporções.

    Se a E1 usou outro procedimento (outra ordem, semente diferente em cada
    chamada, ``stratify`` ausente em uma delas), reproduzam as chamadas da E1
    no notebook e usem ``split_from_frames`` ou ``save_split`` com os índices
    obtidos.

    Parâmetros
    ----------
    y : array-like
        Alvo completo, na mesma ordem da E1.
    test_size, val_size : float ou int
        Valores passados à primeira e à segunda chamada, respectivamente.
    seed : int
        Semente da E1 (``random_state``).
    stratify : bool
        Se as chamadas da E1 usaram ``stratify``.

    Retorna
    -------
    dict
        Divisão no formato de ``save_split`` (ainda não gravada em disco).
    """
    y_arr = np.asarray(y)
    idx = np.arange(len(y_arr))
    rest, test = train_test_split(
        idx,
        test_size=test_size,
        stratify=y_arr if stratify else None,
        random_state=seed,
    )
    train, val = train_test_split(
        rest,
        test_size=val_size,
        stratify=y_arr[rest] if stratify else None,
        random_state=seed,
    )
    split = {
        "train": train.astype(np.int64),
        "val": val.astype(np.int64),
        "test": test.astype(np.int64),
        "seed": int(seed),
        "n_total": len(y_arr),
        "fingerprint": _fingerprint(y_arr),
        "notes": notes
        or f"split_from_e1(test_size={test_size}, val_size={val_size}, stratify={stratify})",
    }
    _validate_split(split, split["n_total"])
    return split


def split_from_frames(X, X_train, X_val, X_test, seed, y=None, notes=""):
    """Obtém os índices posicionais a partir dos DataFrames da E1.

    Útil quando o notebook da E1 produziu ``X_train``, ``X_val`` e ``X_test``
    como DataFrames que preservam o índice original de ``X``. O índice de ``X``
    deve ser único (use ``X.reset_index(drop=True)`` ANTES da divisão, como na
    E1, se for o caso).
    """
    if not X.index.is_unique:
        raise ValueError("O índice de X tem rótulos repetidos; não é possível localizar as linhas.")
    parts = {}
    for name, part in (("train", X_train), ("val", X_val), ("test", X_test)):
        pos = X.index.get_indexer(part.index)
        if (pos < 0).any():
            raise ValueError(f"Há linhas de X_{name} que não estão em X.")
        parts[name] = pos.astype(np.int64)
    split = dict(parts)
    split.update(
        {
            "seed": int(seed),
            "n_total": len(X),
            "fingerprint": _fingerprint(y) if y is not None else None,
            "notes": notes or "split_from_frames",
        }
    )
    _validate_split(split, split["n_total"])
    return split


def _take(obj, idx):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.iloc[idx]
    return np.asarray(obj)[idx]


def development_set(X, y, split):
    """Separa o conjunto de desenvolvimento (treino + validação da E1) e o teste.

    Verifica antes: conjuntos disjuntos, índices dentro do intervalo, tamanho
    total e, se a divisão foi salva com ``y``, se o alvo continua idêntico e na
    mesma ordem (caso contrário a divisão não seria a mesma da E1).

    Retorna
    -------
    X_dev, y_dev, X_test, y_test
        Mesmo tipo de entrada (DataFrame/Series ou arrays). As linhas de
        desenvolvimento vêm na ordem treino seguido de validação; a ordem não
        afeta a validação cruzada, que embaralha com a semente.
    """
    n = len(y)
    if len(X) != n:
        raise ValueError("X e y têm números de linhas diferentes.")
    if split.get("n_total") is not None and split["n_total"] != n:
        raise ValueError(
            f"A divisão foi salva para {split['n_total']} registros, mas os dados têm {n}. "
            "A preparação dos dados mudou desde a E1?"
        )
    if split.get("fingerprint") and split["fingerprint"] != _fingerprint(y):
        raise ValueError(
            "O alvo não é idêntico ao usado quando a divisão foi salva (ordem ou conteúdo "
            "mudou). Refaçam a preparação exatamente como na E1."
        )
    _validate_split(split, n)
    dev = np.concatenate([split["train"], split["val"]])
    return _take(X, dev), _take(y, dev), _take(X, split["test"]), _take(y, split["test"])


def split_report(y, split):
    """Tabela de verificação: tamanho e proporção de cada classe por conjunto.

    Linhas: treino, validação, desenvolvimento (treino + validação), teste e
    total. Colunas: ``n``, ``fracao`` (do total) e a proporção de cada classe.
    Em uma divisão estratificada, as proporções por classe devem ser
    praticamente iguais em todas as linhas.
    """
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    groups = {
        "treino": split["train"],
        "validação": split["val"],
        "desenvolvimento": np.concatenate([split["train"], split["val"]]),
        "teste": split["test"],
        "total": np.arange(len(y_arr)),
    }
    rows = []
    for name, idx in groups.items():
        yy = y_arr[idx]
        row = {"conjunto": name, "n": len(idx), "fracao": len(idx) / len(y_arr)}
        for c in classes:
            row[f"prop_{c}"] = float(np.mean(yy == c)) if len(yy) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("conjunto")


def stratified_subsample(X, y, n, seed):
    """Subamostra estratificada de ``n`` registros (para bases muito grandes).

    Regras de uso (decididas A PRIORI, antes de olhar resultados):

    * aplica-se somente ao conjunto de DESENVOLVIMENTO, para tornar viável a
      busca por validação cruzada (por exemplo, Covertype com 581 mil linhas);
    * o tamanho ``n`` e a semente são fixados e justificados no notebook antes
      de qualquer busca, e a mesma subamostra é usada por TODAS as famílias;
    * o teste NUNCA é subamostrado: os campeões são avaliados no teste
      completo. Predizer é barato; se mesmo a predição no teste for inviável
      (raro), a redução do teste deve ser feita uma única vez, com semente
      fixa, registrada e declarada como limitação.

    Retorna ``X_sub, y_sub``. Se ``n >= len(y)``, devolve os dados intactos.
    """
    if n >= len(y):
        return X, y
    idx = np.arange(len(y))
    keep, _ = train_test_split(idx, train_size=n, stratify=np.asarray(y), random_state=seed)
    keep = np.sort(keep)
    return _take(X, keep), _take(y, keep)


# ---------------------------------------------------------------------------
# Pré-processamento e pipelines
# ---------------------------------------------------------------------------


def _to_categorical_object(X):
    """Converte colunas categóricas para texto, preservando ausentes como NaN."""
    frame = pd.DataFrame(X).copy()
    for col in frame.columns:
        missing = frame[col].isna().to_numpy()
        values = frame[col].astype(str).astype(object).to_numpy()
        values[missing] = np.nan
        frame[col] = values
    return frame.astype(object)


def preprocessor(
    X=None,
    scale=True,
    dense=False,
    numeric=None,
    categorical=None,
    max_categories=20,
    min_frequency=None,
    handle_unknown="ignore",
):
    """Cria o ``ColumnTransformer`` de pré-processamento (ainda NÃO ajustado).

    Numéricos: imputação pela mediana com indicador de ausência
    (``add_indicator=True``: a ausência pode carregar informação) e, se
    ``scale=True``, ``StandardScaler``.

    Categóricos: conversão para texto, imputação pela moda e
    ``OneHotEncoder(handle_unknown="ignore")``. Uma categoria que aparece só no
    teste vira um vetor de zeros, sem erro. ``max_categories`` (ou
    ``min_frequency``) agrupa categorias raras em "infrequentes", limitando a
    cardinalidade.

    Por que a padronização depende da família:

    * kNN, regressão logística, Perceptron e MLP dependem da escala (distâncias,
      penalização e gradiente): ``scale=True``.
    * Árvore de decisão compara cada atributo com limiares e é invariante a
      transformações monótonas por atributo: ``scale=False`` basta.
    * Naive Bayes Gaussiano ajusta média e variância por atributo e por classe;
      sua decisão é invariante a mudanças de escala por atributo, EXCETO pela
      suavização ``var_smoothing``, proporcional à MAIOR variância entre os
      atributos. Sem padronização, um atributo de grande escala torna a
      suavização enorme para os demais. Por isso o padrão aqui é
      ``scale=True`` também para o NB. O NB exige matriz densa (``dense=True``,
      tratado automaticamente por ``build_pipeline``).

    Sem vazamento: este objeto só aprende estatísticas (medianas, modas,
    médias, desvios, categorias) quando ``fit`` é chamado. Dentro de um
    ``Pipeline`` avaliado por validação cruzada, o ``fit`` acontece apenas na
    parte de treino de cada fold.

    Parâmetros
    ----------
    X : DataFrame ou array, opcional
        Usado apenas para identificar as colunas (nada é aprendido aqui). Se
        omitido, as colunas são escolhidas pelo tipo no momento do ``fit``
        (numéricas: dtype numérico; categóricas: as demais, inclusive bool).
    scale : bool
        Aplica ``StandardScaler`` aos numéricos.
    dense : bool
        Força saída densa. Caso contrário, a saída é esparsa quando a matriz
        resultante tiver muitos zeros (muitas categorias).
    numeric, categorical : listas de colunas, opcionais
        Definição explícita (ex.: códigos inteiros que são categorias).
    max_categories, min_frequency, handle_unknown
        Repassados ao ``OneHotEncoder``.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_steps = [
        (
            "to_text",
            FunctionTransformer(_to_categorical_object, feature_names_out="one-to-one"),
        ),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown=handle_unknown,
                max_categories=max_categories,
                min_frequency=min_frequency,
                sparse_output=not dense,
            ),
        ),
    ]

    if X is not None and not isinstance(X, pd.DataFrame):
        n_cols = np.asarray(X).shape[1]
        num_cols = list(range(n_cols)) if numeric is None else list(numeric)
        cat_cols = [] if categorical is None else list(categorical)
    elif X is not None:
        cols = list(X.columns)
        if numeric is None and categorical is None:
            num_cols = list(X.select_dtypes(include="number").columns)
            cat_cols = [c for c in cols if c not in num_cols]
        elif categorical is not None:
            cat_cols = list(categorical)
            num_cols = list(numeric) if numeric is not None else [c for c in cols if c not in cat_cols]
        else:
            num_cols = list(numeric)
            cat_cols = [c for c in cols if c not in num_cols]
    else:
        num_cols = list(numeric) if numeric is not None else make_column_selector(dtype_include="number")
        cat_cols = (
            list(categorical) if categorical is not None else make_column_selector(dtype_exclude="number")
        )

    return ColumnTransformer(
        [("num", Pipeline(num_steps), num_cols), ("cat", Pipeline(cat_steps), cat_cols)],
        remainder="drop",
        sparse_threshold=0.0 if dense else 0.3,
    )


def _model_key(family, key):
    """Normaliza o nome de um parâmetro para o formato do Pipeline."""
    if key.startswith(("model__", "prep__")):
        return key
    if family == "perceptron" and not key.startswith("estimator__") and key not in _CALIBRATOR_PARAMS:
        return "model__estimator__" + key
    return "model__" + key


def _full_params(family, params):
    """Parâmetros com os nomes completos do Pipeline (``model__...``)."""
    return {_model_key(family, k): v for k, v in dict(params or {}).items()}


def _prefix_grid(family, grid):
    if isinstance(grid, dict):
        return {_model_key(family, k): list(v) for k, v in grid.items()}
    return [_prefix_grid(family, g) for g in grid]


def _make_estimator(family, seed):
    if isinstance(family, BaseEstimator):
        return clone(family)
    if family == "arvore":
        return DecisionTreeClassifier(random_state=seed)
    if family == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if family == "naive_bayes":
        return GaussianNB()
    if family == "logistica":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if family == "perceptron":
        return CalibratedClassifierCV(
            estimator=Perceptron(max_iter=1000, tol=1e-3, random_state=seed),
            method="sigmoid",
            cv=3,
        )
    if family == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            alpha=1e-4,
            max_iter=300,
            early_stopping=False,
            random_state=seed,
        )
    if family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LightGBM não está instalado neste ambiente (no Colab ele já vem instalado)."
            ) from exc
        return LGBMClassifier(n_estimators=200, random_state=seed, n_jobs=1, verbose=-1)
    raise ValueError(
        f"Família desconhecida: {family!r}. Use uma de {FAMILIES} (ou 'lightgbm'), "
        "ou passe um estimador do scikit-learn."
    )


def build_pipeline(family, seed=0, X=None, scale=None, prep_kwargs=None, **params):
    """Monta ``Pipeline([("prep", preprocessor), ("model", estimador)])``.

    Configurações-base de cada família:

    * ``"arvore"``: ``DecisionTreeClassifier(random_state=seed)``, sem padronização.
    * ``"knn"``: ``KNeighborsClassifier()``, com padronização.
    * ``"naive_bayes"``: ``GaussianNB()``, com padronização e saída densa.
    * ``"logistica"``: ``LogisticRegression(max_iter=2000)``, com padronização.
    * ``"perceptron"``: ``Perceptron`` envolvido por
      ``CalibratedClassifierCV(method="sigmoid", cv=3)``. O Perceptron só
      produz uma pontuação de margem; a calibração sigmoide (Platt) ajustada
      em validação interna a transforma em probabilidade, necessária para AUC
      e log loss. Parâmetros do Perceptron entram como
      ``model__estimator__<nome>``.
    * ``"mlp"``: ``MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
      early_stopping=False)``. Sem parada antecipada, para que a regularização
      seja controlada explicitamente por ``alpha`` e a curva de validação seja
      interpretável. Com ``alpha`` pequeno a rede continua reduzindo a perda
      de treino até ``max_iter`` (aviso de convergência): é o regime de
      sobreajuste que a curva de validação deve mostrar. Se o modelo
      escolhido atingir ``max_iter``, ``cv_search`` avisa; aumentem
      ``max_iter`` e verifiquem se o resultado muda.
    * ``"lightgbm"`` (extensão opcional): ``LGBMClassifier``, sem padronização.
    * Um estimador do scikit-learn (instância), para modelos opcionais como o
      LVQ da dupla: é clonado e recebe padronização.

    Parâmetros
    ----------
    family : str ou estimador
    seed : int
        ``random_state`` dos modelos que têm componente aleatório.
    X : DataFrame, opcional
        Usado só para identificar colunas numéricas e categóricas.
    scale : bool, opcional
        Sobrescreve a escolha padrão de padronização da família.
    prep_kwargs : dict, opcional
        Argumentos extras para ``preprocessor`` (ex.: ``{"categorical": [...]}``).
    **params
        Hiperparâmetros do modelo, com ou sem o prefixo ``model__``.
    """
    key = family if isinstance(family, str) else None
    if scale is None:
        scale = _DEFAULT_SCALE.get(key, True)
    prep_kwargs = dict(prep_kwargs or {})
    if key == "naive_bayes":
        prep_kwargs["dense"] = True
    prep = preprocessor(X, scale=scale, **prep_kwargs)
    model = _make_estimator(family, seed)
    pipe = Pipeline([("prep", prep), ("model", model)])
    if params:
        pipe.set_params(**{_model_key(key, k): v for k, v in params.items()})
    return pipe


# ---------------------------------------------------------------------------
# Validação cruzada
# ---------------------------------------------------------------------------


def make_cv(seed, n_splits=5):
    """``StratifiedKFold`` embaralhado com a semente da dupla.

    Estratificar mantém a proporção das classes em cada fold; fixar a semente
    torna os folds idênticos para todas as famílias (comparação pareada).
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _family_name(family, name=None):
    if name:
        return name
    if isinstance(family, str):
        return family
    return type(family).__name__


def cv_search(
    family,
    X_dev,
    y_dev,
    grid=None,
    cv=None,
    scoring="balanced_accuracy",
    n_jobs=-1,
    *,
    seed=0,
    name=None,
    scale=None,
    prep_kwargs=None,
    fixed_params=None,
    verbose=0,
):
    """Busca em grade por validação cruzada no conjunto de desenvolvimento.

    Cada combinação da grade é avaliada nos mesmos k folds; em cada fold o
    Pipeline inteiro (pré-processamento + modelo) é ajustado só na parte de
    treino do fold. Ao final, a melhor combinação é REAJUSTADA no
    desenvolvimento inteiro (``refit=True``): ``search.best_estimator_`` é o
    campeão da família que irá ao teste.

    Parâmetros
    ----------
    family : str ou estimador
        Uma das chaves de ``FAMILIES``, ``"lightgbm"`` ou um estimador.
    X_dev, y_dev
        Conjunto de desenvolvimento (treino + validação da E1).
    grid : dict ou lista de dicts, opcional
        Grade de hiperparâmetros. Chaves com ou sem o prefixo ``model__``.
        Se omitida, usa ``SUGGESTED_GRIDS`` (ou ``EXTENSION_GRIDS``).
    cv : objeto de validação cruzada, opcional
        Padrão: ``make_cv(seed)``. Use o MESMO objeto para todas as famílias.
    scoring : str
        Métrica principal (uma só), no formato do scikit-learn, por exemplo
        ``"balanced_accuracy"``, ``"f1_macro"``, ``"roc_auc"`` (binário),
        ``"roc_auc_ovr"`` (multiclasse) ou ``"neg_log_loss"``.
    n_jobs : int
        Processos paralelos (``-1`` usa todos os núcleos; o Colab gratuito tem 2).
    seed : int
        Semente dos modelos (e dos folds, se ``cv`` for omitido).
    name : str, opcional
        Nome exibido nas tabelas (útil para estimadores próprios).
    fixed_params : dict, opcional
        Hiperparâmetros fixos fora da grade.

    Retorna
    -------
    GridSearchCV ajustado, com ``return_train_score=True`` (necessário para o
    estudo de overfitting) e os atributos extras ``family_`` (nome),
    ``elapsed_`` (segundos), ``scoring_``, ``seed_``.
    """
    key = family if isinstance(family, str) else None
    if grid is None:
        if key in SUGGESTED_GRIDS:
            grid = SUGGESTED_GRIDS[key]
        elif key in EXTENSION_GRIDS:
            grid = EXTENSION_GRIDS[key]
        else:
            raise ValueError("Informe a grade (grid) para este modelo.")
    grid = _prefix_grid(key, grid)
    pipe = build_pipeline(
        family, seed=seed, X=X_dev, scale=scale, prep_kwargs=prep_kwargs, **(fixed_params or {})
    )
    search = GridSearchCV(
        pipe,
        grid,
        scoring=scoring,
        cv=cv if cv is not None else make_cv(seed),
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        error_score="raise",
        verbose=verbose,
    )
    start = time.perf_counter()
    search.fit(X_dev, y_dev)
    search.elapsed_ = time.perf_counter() - start
    search.family_ = _family_name(family, name)
    search.family_key_ = family
    search.scoring_ = scoring
    search.seed_ = seed
    search.scale_ = scale
    search.prep_kwargs_ = prep_kwargs
    search.fixed_params_ = _full_params(key, fixed_params)
    final =search.best_estimator_.steps[-1][1]
    n_iter, max_iter = getattr(final, "n_iter_", None), getattr(final, "max_iter", None)
    if n_iter is not None and max_iter is not None and np.max(n_iter) >= max_iter:
        warnings.warn(
            f"{search.family_}: o modelo escolhido atingiu max_iter={max_iter} sem convergir. "
            "Considerem aumentar max_iter e verificar se o resultado muda.",
            stacklevel=2,
        )
    return search


def best_model_params(search, drop=None):
    """Melhores hiperparâmetros de uma busca, sem o prefixo ``model__``.

    ``drop`` remove um parâmetro (ex.: o que vai variar numa curva de validação).
    """
    params = {}
    for k, v in search.best_params_.items():
        k = k.removeprefix("model__")
        params[k] = v
    if drop is not None:
        drop_keys = {drop, "estimator__" + drop}
        params = {k: v for k, v in params.items() if k not in drop_keys}
    return params


def refit_with(search, X_dev, y_dev, **params):
    """Reajusta no desenvolvimento inteiro o melhor modelo da busca, alterando parâmetros.

    Uso típico: a curva de validação com a regra de 1 erro padrão indicou um
    valor mais simples (ex.: ``max_depth=6`` em vez de 12). A decisão de qual
    versão vai ao teste deve ser tomada ANTES de usar o teste.

    Mantém os hiperparâmetros fixos da busca (``fixed_params``) e o mesmo
    pré-processamento. Para que o modelo reajustado seja o mesmo avaliado na
    curva de validação, passem também os parâmetros que a curva fixou de forma
    diferente da busca (ex.: ``weights="uniform"`` no kNN).
    """
    family = getattr(search, "family_key_", None)
    if family is None:
        raise ValueError("Esta busca não foi criada por cv_search.")
    key = family if isinstance(family, str) else None
    merged = dict(getattr(search, "fixed_params_", {}) or {})
    merged.update(search.best_params_)
    merged.update(_full_params(key, params))
    pipe = build_pipeline(
        family,
        seed=search.seed_,
        X=X_dev,
        scale=search.scale_,
        prep_kwargs=search.prep_kwargs_,
        **merged,
    )
    return pipe.fit(X_dev, y_dev)


def _short_params(params):
    out = {}
    for k, v in params.items():
        for prefix in ("model__estimator__", "model__"):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        out[k] = v
    return out


def _as_search_dict(searches):
    if isinstance(searches, dict):
        return dict(searches)
    return {getattr(s, "family_", f"modelo_{i}"): s for i, s in enumerate(searches)}


def cv_summary(searches):
    """Tabela comparativa da validação cruzada.

    Uma linha por família: média, desvio padrão e erro padrão (desvio / raiz de
    k) da métrica nos k folds para a melhor combinação, média de treino,
    melhores hiperparâmetros, número de combinações avaliadas e tempo total da
    busca (segundos). Ordenada da melhor para a pior média.

    ``searches``: dicionário ``{nome: GridSearchCV}`` (ou lista).
    """
    rows = []
    for name, s in _as_search_dict(searches).items():
        res = s.cv_results_
        i = s.best_index_
        k = s.n_splits_
        rows.append(
            {
                "familia": name,
                "metrica": getattr(s, "scoring_", s.scoring),
                "media_cv": res["mean_test_score"][i],
                "dp_cv": res["std_test_score"][i],
                "ep_cv": res["std_test_score"][i] / np.sqrt(k),
                "media_treino": res["mean_train_score"][i],
                "melhores_parametros": _short_params(s.best_params_),
                "n_config": len(res["params"]),
                "tempo_s": round(getattr(s, "elapsed_", np.nan), 1),
            }
        )
    return pd.DataFrame(rows).set_index("familia").sort_values("media_cv", ascending=False)


def cv_fold_scores(searches):
    """Métrica de validação de cada fold (linhas) para a melhor configuração de cada família (colunas).

    Como os folds são os mesmos para todas as famílias, as comparações podem
    ser feitas fold a fold (comparação pareada), por exemplo com um boxplot.
    """
    data = {}
    for name, s in _as_search_dict(searches).items():
        i = s.best_index_
        data[name] = [s.cv_results_[f"split{j}_test_score"][i] for j in range(s.n_splits_)]
    frame = pd.DataFrame(data)
    frame.index = [f"fold {j + 1}" for j in range(len(frame))]
    return frame


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def compute_metrics(y_true, proba, classes):
    """Métricas de classificação a partir das probabilidades previstas.

    Parâmetros
    ----------
    y_true : array-like
        Rótulos verdadeiros.
    proba : array (n, n_classes)
        Probabilidades na ordem de ``classes`` (como ``model.classes_``).
    classes : array-like
        Rótulos das colunas de ``proba``.

    Retorna
    -------
    dict com ``accuracy``, ``balanced_accuracy``, ``f1_macro``, ``auc`` (binária
    com a classe ``classes[1]`` como positiva, ou média macro um-contra-todos
    no caso multiclasse) e ``log_loss``. A predição é a classe de maior
    probabilidade.
    """
    classes = np.asarray(classes)
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    y_pred = classes[np.argmax(proba, axis=1)]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
    }
    try:
        if len(classes) == 2:
            out["auc"] = roc_auc_score(y_true == classes[1], proba[:, 1])
        else:
            out["auc"] = roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=classes
            )
    except ValueError:
        out["auc"] = np.nan
    proba_clip = np.clip(proba, 1e-15, 1.0)
    proba_clip = proba_clip / proba_clip.sum(axis=1, keepdims=True)
    out["log_loss"] = log_loss(y_true, proba_clip, labels=classes)
    return out


def _predict_proba(model, X):
    """Probabilidades e classes; se o modelo não tiver ``predict_proba``, usa a predição (0/1)."""
    classes = np.asarray(model.classes_)
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X), classes, True
        except AttributeError:
            pass
    pred = np.asarray(model.predict(X))
    proba = (pred[:, None] == classes[None, :]).astype(float)
    return proba, classes, False


# ---------------------------------------------------------------------------
# Estudo de overfitting
# ---------------------------------------------------------------------------


def generalization_gap(searches):
    """Lacuna treino-validação da melhor configuração de cada família.

    Para cada fold, calcula (métrica no treino do fold) menos (métrica na
    validação do fold) e resume: média e erro padrão da lacuna. Lacuna grande
    indica variância alta (o modelo se ajusta a particularidades do treino);
    lacuna pequena com desempenho baixo indica viés alto.

    Cuidado de leitura: no kNN com ``weights="distance"`` e no kNN com k = 1,
    o desempenho de treino é sempre (quase) perfeito, porque cada ponto é seu
    próprio vizinho; a lacuna fica inflada por construção.

    Retorna DataFrame com ``treino``, ``validacao``, ``lacuna``, ``lacuna_ep``
    e ``lacuna_relativa`` (lacuna dividida pelo valor absoluto do treino),
    ordenado da maior para a menor lacuna.
    """
    rows = []
    for name, s in _as_search_dict(searches).items():
        res = s.cv_results_
        i = s.best_index_
        k = s.n_splits_
        tr = np.array([res[f"split{j}_train_score"][i] for j in range(k)])
        va = np.array([res[f"split{j}_test_score"][i] for j in range(k)])
        diff = tr - va
        rows.append(
            {
                "familia": name,
                "treino": tr.mean(),
                "validacao": va.mean(),
                "lacuna": diff.mean(),
                "lacuna_ep": diff.std(ddof=0) / np.sqrt(k),
                "lacuna_relativa": diff.mean() / abs(tr.mean()) if tr.mean() else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("familia").sort_values("lacuna", ascending=False)


def validation_curve(
    family,
    param,
    values,
    X_dev,
    y_dev,
    cv=None,
    scoring="balanced_accuracy",
    fixed_params=None,
    n_jobs=-1,
    *,
    seed=0,
    scale=None,
    prep_kwargs=None,
):
    """Curva de validação de complexidade por validação cruzada.

    Varia um único hiperparâmetro (``param``) sobre ``values``, mantendo os
    demais fixos, e mede a métrica no treino e na validação de cada fold.

    Parâmetros
    ----------
    family : str ou estimador
    param : str
        Nome do hiperparâmetro do modelo, sem prefixo (ex.: ``"max_depth"``).
    values : lista
        Valores a avaliar (podem incluir ``None``, como ``max_depth=None``).
    fixed_params : dict ou GridSearchCV, opcional
        Demais hiperparâmetros. Se for uma busca de ``cv_search``, usa os
        melhores parâmetros dela e os seus parâmetros fixos (exceto ``param``),
        com a mesma configuração de pré-processamento.
    cv, scoring, n_jobs, seed
        Como em ``cv_search``. Use o MESMO ``cv`` da busca. Passem também
        ``seed`` igual ao da busca: a semente não é herdada de ``fixed_params``.

    Retorna
    -------
    DataFrame com as colunas ``valor``, ``treino_media``, ``treino_ep``,
    ``val_media``, ``val_ep`` (erro padrão = desvio entre folds / raiz de k).
    ``curve.attrs`` guarda família, parâmetro, métrica e k.
    """
    key = family if isinstance(family, str) else None
    if isinstance(fixed_params, GridSearchCV):
        search = fixed_params
        fixed = dict(getattr(search, "fixed_params_", {}) or {})
        fixed.update(search.best_params_)
        if scale is None:
            scale = getattr(search, "scale_", None)
        if prep_kwargs is None:
            prep_kwargs = getattr(search, "prep_kwargs_", None)
    else:
        fixed = _full_params(key, fixed_params)
    fixed.pop(_model_key(key, param), None)
    pipe = build_pipeline(family, seed=seed, X=X_dev, scale=scale, prep_kwargs=prep_kwargs, **fixed)
    cv = cv if cv is not None else make_cv(seed)
    values = list(values)
    train_scores, val_scores = _sk_validation_curve(
        pipe,
        X_dev,
        y_dev,
        param_name=_model_key(key, param),
        param_range=values,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        error_score="raise",
    )
    k = train_scores.shape[1]
    curve = pd.DataFrame(
        {
            "valor": pd.Series(values, dtype=object),
            "treino_media": train_scores.mean(axis=1),
            "treino_ep": train_scores.std(axis=1) / np.sqrt(k),
            "val_media": val_scores.mean(axis=1),
            "val_ep": val_scores.std(axis=1) / np.sqrt(k),
        }
    )
    curve.attrs.update(
        {"family": _family_name(family), "param": param, "scoring": scoring, "n_splits": k}
    )
    return curve


def _order_key(value):
    if value is None:
        return np.inf
    return float(value)


def _py(value):
    return value.item() if isinstance(value, np.generic) else value


def select_complexity(curve, rule="1se", simpler="lower"):
    """Escolhe o valor do hiperparâmetro a partir de uma curva de validação.

    ``rule="max"``: valor com a maior média de validação.

    ``rule="1se"`` (regra de 1 erro padrão): calcula o limiar = (melhor média)
    menos (erro padrão dessa melhor média) e escolhe, entre os valores cuja
    média de validação é maior ou igual ao limiar, o MAIS SIMPLES. A ideia:
    diferenças menores que um erro padrão não são distinguíveis do ruído da
    validação cruzada; entre modelos estatisticamente equivalentes,
    preferimos o mais simples, que tende a ter menor variância. É uma
    heurística (os folds não são independentes), não um teste formal.
    O erro padrão usa o desvio padrão populacional (ddof=0), como o
    GridSearchCV.

    ``simpler="lower"``: valores menores são mais simples (``max_depth``;
    ``None`` é tratado como infinito, o mais complexo). ``simpler="higher"``:
    valores maiores são mais simples (``n_neighbors``, ``alpha``).

    Retorna o valor escolhido (tipo Python).
    """
    if rule not in ("1se", "max"):
        raise ValueError("rule deve ser '1se' ou 'max'.")
    if simpler not in ("lower", "higher"):
        raise ValueError("simpler deve ser 'lower' ou 'higher'.")
    best = int(np.argmax(curve["val_media"].to_numpy()))
    if rule == "max":
        return _py(curve["valor"].iloc[best])
    threshold = curve["val_media"].iloc[best] - curve["val_ep"].iloc[best]
    candidates = curve.loc[curve["val_media"] >= threshold - 1e-12, "valor"].tolist()
    chooser = min if simpler == "lower" else max
    return _py(chooser(candidates, key=_order_key))


def curve_point(curve, value):
    """Estimativa de validação cruzada (média, EP) de um ponto da curva.

    Uso típico: a versão levada ao teste foi a da regra de 1 EP (por exemplo,
    ``max_depth=5``); a estimativa de CV dessa versão é o ponto correspondente
    da curva, e não a melhor configuração da busca. Passem o resultado a
    ``compare_cv_test(..., cv_estimates={familia: curve_point(...)})``.
    """
    for _, row in curve.iterrows():
        v = row["valor"]
        if (v is None and value is None) or (v is not None and value is not None and v == value):
            return float(row["val_media"]), float(row["val_ep"])
    raise ValueError(f"O valor {value!r} não está na curva.")


def plot_validation_curve(curve, ax=None, simpler=None, title=None, logx=None):
    """Plota a curva de validação com faixas de mais ou menos 1 erro padrão.

    Mostra treino e validação (média e faixa de 1 EP), marca o melhor valor e,
    se ``simpler`` for informado (``"lower"`` ou ``"higher"``), o valor
    escolhido pela regra de 1 EP e o limiar correspondente. Valores não
    numéricos (como ``None``) são exibidos em posições igualmente espaçadas.
    ``logx=None`` escolhe escala logarítmica automaticamente quando os valores
    são positivos e o maior é pelo menos 50 vezes o menor (por exemplo, k de 1
    a 81 no kNN ou ``alpha`` de 0,0001 a 10 na MLP).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))
    values = list(curve["valor"])
    numeric = all(v is not None and np.isscalar(v) and not isinstance(v, str) for v in values)
    if numeric:
        x = np.array([float(v) for v in values])
        if logx is None:
            logx = bool((x > 0).all() and x.max() / x.min() >= 50)
    else:
        x = np.arange(len(values))
        logx = False
    tr, tr_ep = curve["treino_media"].to_numpy(), curve["treino_ep"].to_numpy()
    va, va_ep = curve["val_media"].to_numpy(), curve["val_ep"].to_numpy()
    ax.plot(x, tr, marker="o", color="tab:blue", label="treino (média CV)")
    ax.fill_between(x, tr - tr_ep, tr + tr_ep, color="tab:blue", alpha=0.2)
    ax.plot(x, va, marker="o", color="tab:orange", label="validação (média CV)")
    ax.fill_between(x, va - va_ep, va + va_ep, color="tab:orange", alpha=0.2)
    best = int(np.argmax(va))
    ax.axvline(x[best], color="tab:orange", linestyle=":", label=f"melhor: {values[best]}")
    if simpler is not None:
        chosen = select_complexity(curve, rule="1se", simpler=simpler)
        pos = next(i for i, v in enumerate(values) if v == chosen or (v is None and chosen is None))
        ax.axhline(va[best] - va_ep[best], color="gray", linestyle="--", linewidth=1, label="limiar 1 EP")
        ax.axvline(x[pos], color="tab:green", linestyle="--", label=f"regra 1 EP: {chosen}")
    if logx:
        ax.set_xscale("log")
    if not numeric:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(curve.attrs.get("param", "valor"))
    ax.set_ylabel(str(curve.attrs.get("scoring", "métrica")))
    ax.set_title(title or f"Curva de validação: {curve.attrs.get('family', '')}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_generalization_gap(gap_table, ax=None, title=None):
    """Barras de treino e validação por família, com a lacuna anotada."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4))
    names = list(gap_table.index)
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, gap_table["treino"], width=w, label="treino (CV)", color="tab:blue")
    ax.bar(
        x + w / 2,
        gap_table["validacao"],
        width=w,
        label="validação (CV)",
        color="tab:orange",
        yerr=None,
    )
    for i, (top, gap) in enumerate(zip(gap_table[["treino", "validacao"]].max(axis=1), gap_table["lacuna"])):
        ax.text(i, top, f"{gap:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    low = float(gap_table[["treino", "validacao"]].min().min())
    high = float(gap_table[["treino", "validacao"]].max().max())
    span = max(high - low, 1e-3)
    ax.set_ylim(low - 0.5 * span, high + 0.25 * span)
    ax.set_ylabel("métrica (média nos folds)")
    ax.set_title(title or "Lacuna treino-validação por família")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Erros por classe
# ---------------------------------------------------------------------------


def confusion_table(y_true, y_pred, classes=None, normalize=None):
    """Matriz de confusão como DataFrame (linhas: classe real; colunas: prevista).

    ``normalize="true"`` divide cada linha pelo total da classe real (a diagonal
    vira o recall de cada classe).
    """
    if classes is None:
        classes = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize=normalize)
    return pd.DataFrame(
        cm,
        index=pd.Index(classes, name="real"),
        columns=pd.Index(classes, name="previsto"),
    )


def recall_per_class(y_true, y_pred, classes=None):
    """Suporte, recall, precisão e F1 de cada classe."""
    if classes is None:
        classes = np.unique(np.asarray(y_true))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return pd.DataFrame(
        {"suporte": sup, "recall": rec, "precisao": prec, "f1": f1},
        index=pd.Index(classes, name="classe"),
    )


def plot_confusion(y_true, y_pred, classes=None, normalize="true", ax=None, title=None):
    """Plota a matriz de confusão (por padrão normalizada por classe real)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=classes,
        normalize=normalize,
        ax=ax,
        colorbar=False,
        values_format=".2f" if normalize else "d",
    )
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title(title or "Matriz de confusão")
    return ax


# ---------------------------------------------------------------------------
# Avaliação final no teste (uma única vez)
# ---------------------------------------------------------------------------

_LOG_FIELDS = ["data_hora", "execucao", "forcada", "modelos", "n_teste", "observacao"]


def _read_log(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_table(fitted_models, X_test, y_test, log_path="test_log.csv", force=False, note=""):
    """Avalia os campeões no teste UMA ÚNICA VEZ e devolve a tabela comparativa.

    Antes de avaliar, consulta o arquivo de registro ``log_path``. Se já houver
    um uso registrado, RECUSA a execução com uma mensagem clara, a menos que
    ``force=True``; nesse caso a nova execução é registrada como forçada (e
    deve ser declarada e justificada no notebook). Cada uso grava data e hora,
    número da execução, modelos e tamanho do teste.

    O registro é um compromisso metodológico, não um mecanismo de segurança:
    ele torna visível qualquer reuso do teste. No Colab, mantenham o registro
    no Drive (ex.: ``PASTA + "/test_log.csv"``), junto da divisão.

    Por que só uma vez: se o teste orientar qualquer escolha (modelo,
    hiperparâmetro, limiar), ele deixa de ser uma estimativa sem viés do
    desempenho em dados novos. Todas as escolhas são feitas antes, por
    validação cruzada.

    Parâmetros
    ----------
    fitted_models : dict ``{nome: modelo}``
        Modelos já ajustados no desenvolvimento inteiro (por exemplo,
        ``search.best_estimator_``) ou os próprios objetos ``GridSearchCV``.
    X_test, y_test
        Conjunto de teste da E1.
    log_path : str
        Arquivo de registro (CSV simples).
    force : bool
        Permite nova execução, registrando-a como forçada.
    note : str
        Observação gravada no registro.

    Retorna
    -------
    DataFrame (uma linha por modelo) com ``accuracy``, ``balanced_accuracy``,
    ``f1_macro``, ``auc`` e ``log_loss``. Para modelos sem ``predict_proba``,
    ``auc`` e ``log_loss`` ficam como NaN.
    """
    previous = _read_log(log_path)
    if previous and not force:
        first = previous[0]
        raise RuntimeError(
            "O conjunto de teste já foi usado: registro de "
            f"{first.get('data_hora')} em '{log_path}' ({len(previous)} uso(s)). "
            "O teste deve ser avaliado uma única vez, ao final. Usem a tabela já obtida "
            "(salvem-na em CSV). Se for realmente necessário reavaliar, chamem com "
            "force=True: o novo uso será registrado como forçado e deve ser declarado "
            "e justificado no notebook."
        )
    if force and previous:
        warnings.warn(
            f"Execução FORÇADA do teste (uso número {len(previous) + 1}); declarem no notebook.",
            stacklevel=2,
        )
    rows = []
    no_proba = []
    for name, model in fitted_models.items():
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_
        proba, classes, has_proba = _predict_proba(model, X_test)
        metrics = compute_metrics(y_test, proba, classes)
        if not has_proba:
            metrics["auc"] = np.nan
            metrics["log_loss"] = np.nan
            no_proba.append(name)
        metrics["modelo"] = name
        rows.append(metrics)
    if no_proba:
        warnings.warn(
            f"Sem predict_proba (AUC e log loss omitidos): {', '.join(no_proba)}",
            stacklevel=2,
        )
    table = pd.DataFrame(rows).set_index("modelo")
    stamp = _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    new_file = not os.path.exists(log_path)
    parent = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(parent, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "data_hora": stamp,
                "execucao": len(previous) + 1,
                "forcada": bool(force and previous),
                "modelos": ";".join(map(str, fitted_models)),
                "n_teste": len(y_test),
                "observacao": note,
            }
        )
    table.attrs["data_hora"] = stamp
    return table


_SCORING_TO_TEST = {
    "accuracy": ("accuracy", 1.0),
    "balanced_accuracy": ("balanced_accuracy", 1.0),
    "f1_macro": ("f1_macro", 1.0),
    "roc_auc": ("auc", 1.0),
    "roc_auc_ovr": ("auc", 1.0),
    "neg_log_loss": ("log_loss", -1.0),
}


def compare_cv_test(searches, test_df, cv_estimates=None):
    """Compara, para cada família, a métrica principal na CV e no teste.

    Usa a métrica de ``scoring`` das buscas (convertida para a coluna
    equivalente de ``test_table``; ``neg_log_loss`` vira ``log_loss`` com sinal
    trocado). Colunas: ``cv_media``, ``cv_ep``, ``teste``, ``diferenca``
    (teste menos CV) e ``diferenca_em_ep`` (diferença em unidades do erro
    padrão da CV). Diferenças de poucos EPs são esperadas; diferenças grandes
    e sistemáticas merecem investigação (mudança de distribuição, vazamento,
    teste pequeno).

    ``cv_estimates`` (opcional): ``{familia: (media, ep)}`` na mesma unidade da
    busca, para as famílias cuja versão levada ao teste NÃO é a melhor
    configuração da busca (por exemplo, a da regra de 1 EP, com
    ``curve_point``). A coluna ``origem_cv`` indica de onde veio a estimativa.
    """
    summary = cv_summary(searches)
    cv_estimates = dict(cv_estimates or {})
    unknown = set(cv_estimates) - set(summary.index)
    if unknown:
        raise ValueError(f"cv_estimates com famílias sem busca: {sorted(unknown)}")
    rows = []
    for name, row in summary.iterrows():
        if name not in test_df.index:
            continue
        scoring = row["metrica"]
        if scoring not in _SCORING_TO_TEST:
            raise ValueError(f"Métrica {scoring!r} sem equivalente em test_table.")
        col, sign = _SCORING_TO_TEST[scoring]
        if name in cv_estimates:
            mean, ep = cv_estimates[name]
            origin = "informada"
        else:
            mean, ep = row["media_cv"], row["ep_cv"]
            origin = "busca"
        cv_mean = sign * mean
        test_value = test_df.loc[name, col]
        diff = test_value - cv_mean
        rows.append(
            {
                "familia": name,
                "metrica": col,
                "origem_cv": origin,
                "cv_media": cv_mean,
                "cv_ep": ep,
                "teste": test_value,
                "diferenca": diff,
                "diferenca_em_ep": diff / ep if ep > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("familia")


# Evita que o pytest trate esta função como um teste quando importada.
test_table.__test__ = False
