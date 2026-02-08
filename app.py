
# Fabrica de modelos --  (CSV -> Limpeza -> 30+ Modelos -> Ranking -> Dashboard -> PDF)
# Interface: Streamlit
# AutoML: PyCaret (classificação/regressão)
# PDF: reportlab
# Explicabilidade: interpret_model (SHAP quando disponível) + fallback feature importance
# Deploy: save_model -> best_model_pipeline.pkl

import io
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader


# ------------------------------------------------------------
# 1) FUNÇÕES DE SUPORTE: limpeza, inferência de tarefa, target etc.
# ------------------------------------------------------------

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza rápida e segura:
    - remove colunas 100% vazias
    - remove duplicados
    - remove colunas constantes
    - tenta converter colunas object com cara de data para datetime
    """
    df = df.copy()

    # Remove colunas totalmente vazias
    df = df.dropna(axis=1, how="all")

    # Remove duplicados
    df = df.drop_duplicates()

    # Remove colunas constantes (mesmo valor em todas as linhas)
    nun = df.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols)

    # Tentativa de converter datas automaticamente (somente colunas object)
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue

        # Heurística: se a maioria contém separadores de data/hora
        looks_like_date = sample.str.contains(r"[-/:]").mean() > 0.6
        if looks_like_date:
            conv = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            # Só troca se uma fração razoável virou data
            if conv.notna().mean() > 0.6:
                df[col] = conv

    return df


def infer_task_type(y: pd.Series) -> str:
    """
    Detecta automaticamente se é classificação ou regressão.
    Heurística:
    - bool -> classificação
    - numérico com muitos valores únicos -> regressão
    - poucos valores únicos -> classificação
    - object/categórico -> classificação
    """
    y = y.dropna()
    if y.empty:
        return "classification"

    n_unique = y.nunique(dropna=True)
    n = len(y)

    if pd.api.types.is_bool_dtype(y):
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        ratio_unique = n_unique / max(n, 1)
        # Poucas categorias (inteiros): tende a ser classe
        if n_unique <= 20:
            return "classification"
        # Se tem pouca variedade em relação ao tamanho
        if ratio_unique < 0.05 and n_unique <= 50:
            return "classification"
        return "regression"

    return "classification"


def detect_id_columns(df: pd.DataFrame, threshold_unique_ratio: float = 0.98):
    """
    Detecta colunas com cara de ID (quase únicas):
    - se nunique/len >= threshold_unique_ratio, provavelmente é ID
    """
    n = len(df)
    if n == 0:
        return []
    id_cols = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun / n >= threshold_unique_ratio:
            id_cols.append(c)
    return id_cols


def suggest_target(df: pd.DataFrame):
    """
    Sugere automaticamente um target.
    Estratégia:
    - remove candidatos que parecem ID
    - remove candidatos com > 60% nulos
    - prioriza colunas com cara de "classe" (2..50 valores únicos) ou numérica com boa variedade
    """
    id_cols = set(detect_id_columns(df))
    candidates = []
    n = len(df)

    for c in df.columns:
        if c in id_cols:
            continue
        miss = df[c].isna().mean()
        if miss > 0.60:
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        candidates.append(c)

    if not candidates:
        return df.columns[-1]

    def score(col):
        s = df[col]
        nun = s.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(s):
            # regressão provável
            if nun > max(50, int(0.05 * max(n, 1))):
                return 80
            # inteiro “categórico”
            if nun <= 20:
                return 70
            return 60
        else:
            # classificação provável
            if 2 <= nun <= 50:
                return 85
            if nun <= 200:
                return 65
            return 30

    return max(candidates, key=score)


def find_latest_plot_file(search_dirs=("Models", "."), exts=(".png", ".jpg", ".jpeg", ".webp")):
    """
    Tenta localizar a imagem mais recente salva em diretórios (ex.: Models/)
    Útil para anexar no PDF a imagem do SHAP/feature importance caso exista.
    """
    newest = None
    newest_mtime = -1
    for d in search_dirs:
        p = Path(d)
        if not p.exists() or not p.is_dir():
            continue
        for f in p.glob("**/*"):
            if f.is_file() and f.suffix.lower() in exts:
                mt = f.stat().st_mtime
                if mt > newest_mtime:
                    newest_mtime = mt
                    newest = f
    return str(newest) if newest else None


# ------------------------------------------------------------
# 2) PDF FINAL
# ------------------------------------------------------------

def build_pdf_report(
    df_shape,
    target_col,
    task_type,
    best_model_name,
    leaderboard_df: pd.DataFrame,
    id_cols_removed,
    metric_name,
    cv_score,
    holdout_score,
    overfit_gap,
    expl_image_path=None
) -> bytes:
    """
    Gera um PDF simples e útil com:
    - resumo do dataset
    - target + tipo de tarefa
    - colunas ID removidas
    - CV vs Holdout
    - ranking top 20
    - (opcional) imagem explicabilidade
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # -------- Página 1: Resumo + Ranking
    x = 2 * cm
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Relatório AutoML - Resultado Final")
    y -= 1.1 * cm

    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 0.7 * cm
    c.drawString(x, y, f"Linhas x Colunas: {df_shape[0]} x {df_shape[1]}")
    y -= 0.7 * cm
    c.drawString(x, y, f"Target: {target_col}")
    y -= 0.7 * cm
    c.drawString(x, y, f"Tipo: {'Classificação' if task_type=='classification' else 'Regressão'}")
    y -= 0.7 * cm
    c.drawString(x, y, f"Melhor modelo: {best_model_name}")
    y -= 0.9 * cm

    # IDs removidos
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Tratamento")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    ids_text = ", ".join(id_cols_removed) if id_cols_removed else "Nenhuma"
    c.drawString(x, y, f"Colunas ID removidas (quase únicas): {ids_text}"[:120])
    y -= 0.9 * cm

    # CV vs Holdout
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Validação (CV vs Holdout)")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    if metric_name and cv_score is not None and holdout_score is not None:
        c.drawString(x, y, f"Métrica principal: {metric_name}")
        y -= 0.6 * cm
        c.drawString(x, y, f"CV: {cv_score:.4f} | Holdout: {holdout_score:.4f} | Gap: {overfit_gap:.4f}")
        y -= 0.6 * cm
        if overfit_gap is not None and overfit_gap > 0.05:
            c.drawString(x, y, "Alerta: possível overfitting (queda relevante no holdout).")
        else:
            c.drawString(x, y, "Sem sinal forte de overfitting pelo gap CV→Holdout.")
    else:
        c.drawString(x, y, "Não foi possível calcular CV vs Holdout automaticamente.")
    y -= 1.0 * cm

    # Ranking top 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Ranking de Modelos (top 20)")
    y -= 0.7 * cm

    top = leaderboard_df.head(20).copy()
    cols = list(top.columns[:6])  # limita pra caber
    top = top[cols]

    c.setFont("Helvetica", 9)
    row_h = 0.55 * cm

    header = " | ".join([str(col) for col in cols])
    c.drawString(x, y, header[:110])
    y -= row_h

    c.setFont("Helvetica", 8)
    for i in range(len(top)):
        line = " | ".join([str(top.iloc[i][col]) for col in cols])
        c.drawString(x, y, line[:120])
        y -= row_h
        if y < 2.5 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Helvetica", 8)

    # -------- Página 2: Explicabilidade (imagem)
    c.showPage()
    x = 2 * cm
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Explicabilidade (SHAP / Feature Importance)")
    y -= 1.0 * cm

    if expl_image_path and os.path.exists(expl_image_path):
        img = ImageReader(expl_image_path)
        img_w = width - 4 * cm
        img_h = height - 5 * cm
        c.drawImage(img, 2 * cm, 2 * cm, width=img_w, height=img_h,
                    preserveAspectRatio=True, anchor='c')
    else:
        c.setFont("Helvetica", 11)
        c.drawString(x, y, "Não foi possível anexar imagem da explicabilidade.")

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer.getvalue()


# ------------------------------------------------------------
# 3) APP STREAMLIT
# ------------------------------------------------------------

st.set_page_config(
    page_title="AutoML Premium (CSV → 30+ modelos → Dashboard → PDF)",
    layout="wide"
)

st.title("🚀 AutoML Premium: CSV → Limpeza → 30+ Modelos → Ranking → Dashboard → PDF")

# Sidebar: configurações
with st.sidebar:
    st.header("Configurações")
    uploaded = st.file_uploader("Envie um CSV", type=["csv"])

    sep = st.text_input("Separador (padrão: ,)", value=",")
    encoding = st.text_input("Encoding (padrão: utf-8)", value="utf-8")

    test_size = st.slider("Tamanho do teste (holdout)", 0.1, 0.4, 0.2, 0.05)
    fold = st.slider("Cross-validation (fold)", 3, 10, 5, 1)

    turbo = st.checkbox("Turbo (mais rápido)", value=True)

    fix_imbalance = st.checkbox("Balancear classes (somente classificação)", value=True)
    remove_outliers = st.checkbox("Remover outliers (somente regressão)", value=False)

    session_seed = st.number_input("Seed", min_value=1, max_value=999999, value=42)

# Se não subiu CSV, para aqui
if not uploaded:
    st.info("Envie um CSV na barra lateral para começar.")
    st.stop()

# Carrega CSV
try:
    df_raw = pd.read_csv(uploaded, sep=sep, encoding=encoding)
except Exception as e:
    st.error(f"Erro ao ler CSV: {e}")
    st.stop()

# Limpeza básica
df = basic_cleaning(df_raw)

# Painel geral
st.subheader("1) Visão geral dos dados")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Linhas", f"{df.shape[0]:,}".replace(",", "."))
c2.metric("Colunas", f"{df.shape[1]}")
c3.metric("Nulos (total)", f"{int(df.isna().sum().sum()):,}".replace(",", "."))
c4.metric("Duplicados removidos", f"{int(df_raw.duplicated().sum()):,}".replace(",", "."))

with st.expander("Prévia do dataset (top 50)"):
    st.dataframe(df.head(50), use_container_width=True)

# Sugestão automática do target
st.subheader("2) Escolha do target (com sugestão automática)")
suggested = suggest_target(df)
target = st.selectbox("Target", options=df.columns, index=list(df.columns).index(suggested))
st.caption(f"Sugestão automática: **{suggested}** (você pode trocar)")

# Detecta tarefa
task = infer_task_type(df[target])
st.write(f"✅ Tipo detectado: **{'Classificação' if task=='classification' else 'Regressão'}**")

# Detecta colunas ID e remove (exceto target)
id_cols = detect_id_columns(df)
id_cols_removed = [c for c in id_cols if c != target]
if id_cols_removed:
    st.warning(f"Removendo colunas com cara de ID (quase únicas): {id_cols_removed}")
    df = df.drop(columns=id_cols_removed)

# Botão para treinar
st.subheader("3) Treinar e comparar modelos (30+)")
train_btn = st.button("▶️ Rodar AutoML agora", type="primary")

# Estado
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None

# Métricas extras
for k in ["metric_pref", "cv_best", "holdout_best", "overfit_gap", "expl_image_path", "id_cols_removed"]:
    if k not in st.session_state:
        st.session_state[k] = None

# Treino
if train_btn:
    with st.spinner("Treinando e comparando modelos... (pode variar com o tamanho do CSV)"):
        metric_pref = None
        cv_best = None
        holdout_best = None
        overfit_gap = None
        expl_image_path = None

        # ---- Classificação ----
        if task == "classification":
            # Import sob demanda para reduzir custo de carregamento inicial
            from pycaret.classification import (
                setup, compare_models, pull, predict_model,
                interpret_model, plot_model, finalize_model, save_model
            )

            # Setup: PyCaret cuida de imputação, encoding, normalização etc.
            _ = setup(
                data=df,
                target=target,
                session_id=int(session_seed),
                train_size=float(1.0 - test_size),
                fold=int(fold),
                use_gpu=False,
                fix_imbalance=bool(fix_imbalance),
                silent=True,
                html=False,
                turbo=bool(turbo),
            )

            # Métrica principal: AUC para multi-classe/binary (quando possível); senão Accuracy
            metric_pref = "AUC" if df[target].nunique() > 2 else "Accuracy"

            # Treina vários modelos e ranqueia automaticamente
            best = compare_models(sort=metric_pref)
            leaderboard = pull()  # tabela de comparação (CV)

            # Avaliação no holdout
            _ = predict_model(best)
            holdout_metrics = pull()  # tabela de métricas no holdout

            # Pega melhor score no CV
            if metric_pref in leaderboard.columns:
                cv_best = float(leaderboard.iloc[0][metric_pref])

            # Pega métrica no holdout (tabela: Metric / Value)
            if "Metric" in holdout_metrics.columns and "Value" in holdout_metrics.columns:
                row = holdout_metrics[holdout_metrics["Metric"] == metric_pref]
                if not row.empty:
                    holdout_best = float(row["Value"].iloc[0])

            if cv_best is not None and holdout_best is not None:
                overfit_gap = cv_best - holdout_best

            # Explicabilidade: tenta SHAP (summary)
            # OBS: nem todo ambiente/modelo vai gerar imagem fácil para o PDF.
            # Mostra no app; depois tentamos achar o arquivo salvo mais recente.
            try:
                st.subheader("🔎 Explicabilidade (SHAP) — Melhor modelo")
                interpret_model(best, plot="summary")
                expl_image_path = find_latest_plot_file()
            except Exception:
                pass

            # Fallback: Feature Importance salva como imagem (quando o modelo suportar)
            try:
                plot_model(best, plot="feature", save=True)
                expl_image_path = find_latest_plot_file(search_dirs=("Models", "."))
            except Exception:
                pass

            # Finaliza modelo (treina no dataset todo) e salva pipeline+modelo para deploy
            final_best = finalize_model(best)
            save_model(final_best, "best_model_pipeline")  # gera best_model_pipeline.pkl

        # ---- Regressão ----
        else:
            from pycaret.regression import (
                setup, compare_models, pull, predict_model,
                interpret_model, plot_model, finalize_model, save_model
            )

            _ = setup(
                data=df,
                target=target,
                session_id=int(session_seed),
                train_size=float(1.0 - test_size),
                fold=int(fold),
                use_gpu=False,
                remove_outliers=bool(remove_outliers),
                silent=True,
                html=False,
                turbo=bool(turbo),
            )

            metric_pref = "R2"
            best = compare_models(sort=metric_pref)
            leaderboard = pull()

            _ = predict_model(best)
            holdout_metrics = pull()

            if metric_pref in leaderboard.columns:
                cv_best = float(leaderboard.iloc[0][metric_pref])

            if "Metric" in holdout_metrics.columns and "Value" in holdout_metrics.columns:
                row = holdout_metrics[holdout_metrics["Metric"] == metric_pref]
                if not row.empty:
                    holdout_best = float(row["Value"].iloc[0])

            if cv_best is not None and holdout_best is not None:
                overfit_gap = cv_best - holdout_best

            try:
                st.subheader("🔎 Explicabilidade (SHAP) — Melhor modelo")
                interpret_model(best, plot="summary")
                expl_image_path = find_latest_plot_file()
            except Exception:
                pass

            try:
                plot_model(best, plot="feature", save=True)
                expl_image_path = find_latest_plot_file(search_dirs=("Models", "."))
            except Exception:
                pass

            final_best = finalize_model(best)
            save_model(final_best, "best_model_pipeline")

        # Salva no session_state para dashboard + PDF
        st.session_state.leaderboard = leaderboard
        st.session_state.best_model_name = str(best)
        st.session_state.metric_pref = metric_pref
        st.session_state.cv_best = cv_best
        st.session_state.holdout_best = holdout_best
        st.session_state.overfit_gap = overfit_gap
        st.session_state.expl_image_path = expl_image_path
        st.session_state.id_cols_removed = id_cols_removed


# ------------------------------------------------------------
# 4) DASHBOARD + DOWNLOADS
# ------------------------------------------------------------

leaderboard = st.session_state.leaderboard
best_model_name = st.session_state.best_model_name

if leaderboard is None:
    st.info("Clique em **Rodar AutoML agora** para treinar os modelos e ver o dashboard.")
    st.stop()

st.subheader("4) Dashboard — Ranking dos modelos")

left, right = st.columns([2, 1])

with left:
    st.markdown("### 📋 Tabela (do melhor → pior)")
    st.dataframe(leaderboard, use_container_width=True)

with right:
    st.markdown("### 🏆 Melhor modelo")
    st.success(best_model_name)

    st.markdown("### 📊 Ranking (top 15)")
    topn = leaderboard.head(15).copy()

    # tenta achar uma métrica relevante para o gráfico
    metric_col = None
    for col in ["AUC", "Accuracy", "R2", "MAE", "RMSE", "MSE"]:
        if col in topn.columns:
            metric_col = col
            break

    if metric_col and "Model" in topn.columns:
        fig = plt.figure()
        plt.barh(topn["Model"].astype(str)[::-1], topn[metric_col][::-1])
        plt.xlabel(metric_col)
        plt.ylabel("Model")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Não consegui identificar automaticamente uma métrica para plotar.")


# Validação CV vs Holdout
st.subheader("🧪 Validação (CV vs Holdout) e Overfitting")

metric_pref = st.session_state.metric_pref
cv_best = st.session_state.cv_best
holdout_best = st.session_state.holdout_best
overfit_gap = st.session_state.overfit_gap

if metric_pref and cv_best is not None and holdout_best is not None:
    st.write(f"**Métrica principal:** {metric_pref}")
    st.write(f"**CV:** {cv_best:.4f}")
    st.write(f"**Holdout:** {holdout_best:.4f}")
    st.write(f"**Gap (CV - Holdout):** {overfit_gap:.4f}")

    if overfit_gap > 0.05:
        st.warning("Possível overfitting: queda relevante no holdout.")
    else:
        st.success("Sem sinal forte de overfitting pelo gap CV→Holdout.")
else:
    st.info("Não foi possível calcular automaticamente CV vs Holdout.")


# Download do modelo+pipeline
st.subheader("💾 Deploy — baixar modelo + pipeline (.pkl)")

pkl_path = "best_model_pipeline.pkl"
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        st.download_button(
            "⬇️ Baixar best_model_pipeline.pkl",
            data=f.read(),
            file_name="best_model_pipeline.pkl",
            mime="application/octet-stream"
        )
else:
    st.info("Arquivo .pkl ainda não encontrado. Rode o AutoML para gerar.")


# PDF final
st.subheader("📄 Exportar relatório em PDF")

pdf_bytes = build_pdf_report(
    df_shape=df.shape,
    target_col=target,
    task_type=task,
    best_model_name=best_model_name,
    leaderboard_df=leaderboard,
    id_cols_removed=st.session_state.id_cols_removed or [],
    metric_name=metric_pref,
    cv_score=cv_best,
    holdout_score=holdout_best,
    overfit_gap=overfit_gap,
    expl_image_path=st.session_state.expl_image_path
)

st.download_button(
    label="⬇️ Baixar Relatório PDF",
    data=pdf_bytes,
    file_name="relatorio_automl.pdf",
    mime="application/pdf"
)

st.caption("Dica: se a imagem de explicabilidade não aparecer no PDF, o app ainda mostra no painel — alguns ambientes não salvam automaticamente o plot em arquivo.")
