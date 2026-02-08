
# Sistema : (CSV -> Limpeza -> 30+ Modelos -> Ranking -> Dashboard -> PDF)
# + Hard Mode (semáforo + fixes + robustez)
# Interface: Streamlit
# AutoML: PyCaret (classificação/regressão)
# PDF: reportlab

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
# 1) FUNÇÕES: limpeza, inferência de tarefa, target, etc.
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

    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()

    nun = df.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue
        looks_like_date = sample.str.contains(r"[-/:]").mean() > 0.6
        if looks_like_date:
            conv = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if conv.notna().mean() > 0.6:
                df[col] = conv

    return df


def infer_task_type(y: pd.Series) -> str:
    """
    Detecta se é classificação ou regressão.
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
        if n_unique <= 20:
            return "classification"
        if ratio_unique < 0.05 and n_unique <= 50:
            return "classification"
        return "regression"

    return "classification"


def detect_id_columns(df: pd.DataFrame, threshold_unique_ratio: float = 0.98):
    """
    Colunas com cara de ID: nunique/len >= threshold
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
    Sugere automaticamente o target:
    - ignora IDs
    - ignora colunas com muitos nulos
    - prioriza baixa/média cardinalidade (classificação) ou numérica com boa variedade (regressão)
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
            if nun > max(50, int(0.05 * max(n, 1))):
                return 80
            if nun <= 20:
                return 70
            return 60
        else:
            if 2 <= nun <= 50:
                return 85
            if nun <= 200:
                return 65
            return 30

    return max(candidates, key=score)


def find_latest_plot_file(search_dirs=("Models", "."), exts=(".png", ".jpg", ".jpeg", ".webp")):
    """
    Acha a imagem mais recente em diretórios (para anexar no PDF).
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
# 2) HARD MODE: semáforo + recomendações + fixes seguros
# ------------------------------------------------------------

def dataset_health_report(df: pd.DataFrame, target: str):
    report = {}
    flags = {}
    actions = []

    n_rows, n_cols = df.shape
    report["rows"] = n_rows
    report["cols"] = n_cols

    y = df[target]
    report["target_missing_ratio"] = float(y.isna().mean())
    report["target_unique"] = int(y.dropna().nunique())

    if report["target_unique"] < 2:
        flags["target"] = "red"
        actions.append("Target inválido: menos de 2 valores distintos. Troque o target ou revise os dados.")
    else:
        flags["target"] = "green"

    if n_rows < 50:
        flags["rows"] = "red"
        actions.append("Poucas linhas (< 50): treino instável. Tente mais dados.")
    elif n_rows < 200:
        flags["rows"] = "yellow"
        actions.append("Dataset pequeno (< 200): resultados podem variar. Cuidado com overfitting.")
    else:
        flags["rows"] = "green"

    total_missing = int(df.isna().sum().sum())
    report["missing_total"] = total_missing
    report["missing_ratio"] = float(total_missing / max(n_rows * max(n_cols, 1), 1))

    if report["missing_ratio"] > 0.30:
        flags["missing"] = "yellow"
        actions.append("Muitos valores faltantes (> 30%). Considere remover colunas com muitos nulos.")
    else:
        flags["missing"] = "green"

    null_thresh = 0.85
    high_null_cols = [c for c in df.columns if c != target and df[c].isna().mean() > null_thresh]
    report["high_null_cols"] = high_null_cols
    if high_null_cols:
        flags["high_null_cols"] = "yellow"
        actions.append(f"Remover colunas com > {int(null_thresh*100)}% nulos: {high_null_cols}")
    else:
        flags["high_null_cols"] = "green"

    id_cols = detect_id_columns(df, threshold_unique_ratio=0.98)
    id_cols = [c for c in id_cols if c != target]
    report["id_cols"] = id_cols
    if id_cols:
        flags["id_cols"] = "yellow"
        actions.append(f"Remover colunas tipo ID (quase únicas): {id_cols}")
    else:
        flags["id_cols"] = "green"

    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    high_card_text = []
    for c in text_cols:
        nun = df[c].nunique(dropna=True)
        if nun > min(500, int(0.5 * max(n_rows, 1))):
            high_card_text.append(c)
    report["high_card_text"] = high_card_text
    if high_card_text:
        flags["high_card_text"] = "yellow"
        actions.append(f"Textos de alta cardinalidade (podem explodir encoding): {high_card_text}")
    else:
        flags["high_card_text"] = "green"

    dup = int(df.duplicated().sum())
    report["duplicates"] = dup
    if dup > 0:
        flags["duplicates"] = "yellow"
        actions.append(f"Remover duplicados: {dup} linhas.")
    else:
        flags["duplicates"] = "green"

    severity_order = {"green": 0, "yellow": 1, "red": 2}
    overall = max(flags.values(), key=lambda x: severity_order.get(x, 0)) if flags else "green"
    report["overall"] = overall

    return report, flags, actions


def show_health_dashboard(report, flags, actions):
    color_map = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

    st.subheader("🛡️ Hard Mode — Saúde do Dataset (Semáforo)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", report["rows"])
    c2.metric("Colunas", report["cols"])
    c3.metric("Nulos (total)", report["missing_total"])
    c4.metric("Severidade", f"{color_map.get(report['overall'],'🟢')} {report['overall'].upper()}")

    st.markdown("### Checks")
    checks = [
        ("Target válido", flags.get("target", "green")),
        ("Tamanho do dataset", flags.get("rows", "green")),
        ("Valores faltantes (geral)", flags.get("missing", "green")),
        ("Colunas com muitos nulos", flags.get("high_null_cols", "green")),
        ("Colunas tipo ID", flags.get("id_cols", "green")),
        ("Texto alta cardinalidade", flags.get("high_card_text", "green")),
        ("Duplicados", flags.get("duplicates", "green")),
    ]
    for name, sev in checks:
        st.write(f"{color_map.get(sev,'🟢')} **{name}**")

    if actions:
        st.markdown("### Recomendações automáticas")
        for a in actions:
            st.write(f"- {a}")
    else:
        st.success("Dataset com boa saúde para treinar.")


def hard_mode_apply_fixes(df: pd.DataFrame, target: str):
    """
    Correções seguras:
    - remove colunas com >85% nulos (exceto target)
    - remove textos de alta cardinalidade (exceto target)
    """
    logs = []
    df2 = df.copy()

    null_thresh = 0.85
    bad_null_cols = [c for c in df2.columns if c != target and df2[c].isna().mean() > null_thresh]
    if bad_null_cols:
        df2 = df2.drop(columns=bad_null_cols)
        logs.append(f"Removidas colunas com > {int(null_thresh*100)}% nulos: {bad_null_cols}")

    n = len(df2)
    text_cols = df2.select_dtypes(include=["object"]).columns.tolist()
    high_card_text = []
    for c in text_cols:
        if c == target:
            continue
        nun = df2[c].nunique(dropna=True)
        if nun > min(500, int(0.5 * max(n, 1))):
            high_card_text.append(c)

    if high_card_text:
        df2 = df2.drop(columns=high_card_text)
        logs.append(f"Removidos textos de alta cardinalidade: {high_card_text}")

    return df2, logs


# ------------------------------------------------------------
# 3) PDF FINAL
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
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

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

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Tratamento")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    ids_text = ", ".join(id_cols_removed) if id_cols_removed else "Nenhuma"
    c.drawString(x, y, f"Colunas ID removidas (quase únicas): {ids_text}"[:120])
    y -= 0.9 * cm

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

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Ranking de Modelos (top 20)")
    y -= 0.7 * cm

    top = leaderboard_df.head(20).copy()
    cols = list(top.columns[:6])
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
# 4) APP STREAMLIT
# ------------------------------------------------------------

st.set_page_config(page_title="AutoML (Treino)", layout="wide")
st.title("AutoML: CSV + Limpeza + 30+ Modelos + Ranking + Dashboard + PDF")

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

    st.divider()
    hard_mode = st.checkbox("Hard Mode (mais robusto, menos erro)", value=True)

if not uploaded:
    st.info("Envie um CSV na barra lateral para começar.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded, sep=sep, encoding=encoding)
except Exception as e:
    st.error(f"Erro ao ler CSV: {e}")
    st.stop()

df = basic_cleaning(df_raw)

st.subheader("1) Visão geral dos dados")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Linhas", f"{df.shape[0]:,}".replace(",", "."))
c2.metric("Colunas", f"{df.shape[1]}")
c3.metric("Nulos (total)", f"{int(df.isna().sum().sum()):,}".replace(",", "."))
c4.metric("Duplicados removidos", f"{int(df_raw.duplicated().sum()):,}".replace(",", "."))

with st.expander("Prévia do dataset (top 50)"):
    st.dataframe(df.head(50), use_container_width=True)

st.subheader("2) Target (com sugestão automática)")
suggested = suggest_target(df)
target = st.selectbox("Target", options=df.columns, index=list(df.columns).index(suggested))
st.caption(f"Sugestão automática: **{suggested}** (você pode trocar)")

task = infer_task_type(df[target])
st.write(f"✅ Tipo detectado: **{'Classificação' if task=='classification' else 'Regressão'}**")

# IDs removidos (sempre)
id_cols = detect_id_columns(df)
id_cols_removed = [c for c in id_cols if c != target]
if id_cols_removed:
    st.warning(f"Removendo colunas com cara de ID (quase únicas): {id_cols_removed}")
    df = df.drop(columns=id_cols_removed)

# Hard Mode (semáforo + bloqueio + fixes)
if hard_mode:
    report, flags, actions = dataset_health_report(df, target)
    show_health_dashboard(report, flags, actions)

    if report["overall"] == "red":
        st.error("Hard Mode bloqueou o treino porque há erros críticos no dataset/target.")
        st.stop()

    df_fixed, fix_logs = hard_mode_apply_fixes(df, target)
    if fix_logs:
        st.info("Hard Mode aplicou correções automáticas:")
        for log in fix_logs:
            st.write(f"- {log}")
    df = df_fixed

st.subheader("3) Treinar e comparar modelos (30+)")
train_btn = st.button("▶️ Rodar AutoML agora", type="primary")

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
for k in ["metric_pref", "cv_best", "holdout_best", "overfit_gap", "expl_image_path", "id_cols_removed"]:
    if k not in st.session_state:
        st.session_state[k] = None

if train_btn:
    with st.spinner("Treinando e comparando modelos..."):
        metric_pref = None
        cv_best = None
        holdout_best = None
        overfit_gap = None
        expl_image_path = None

        if task == "classification":
            from pycaret.classification import (
                setup, compare_models, pull, predict_model,
                interpret_model, plot_model, finalize_model, save_model
            )

            _ = setup(
                data=df,
                target=target,
                session_id=int(session_seed),
                train_size=float(1.0 - test_size),
                fold=int(fold),
                verbose=False,

                # Hardening (robustez)
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                normalize=True,
                normalize_method="zscore",
                fold_shuffle=True,

                fix_imbalance=bool(fix_imbalance),
                use_gpu=False
            )

            metric_pref = "AUC" if df[target].nunique() > 2 else "Accuracy"

            best = compare_models(
                sort=metric_pref,
                n_select=1,
                errors="ignore" if hard_mode else "raise",
                turbo=bool(turbo)
            )
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
                silent=True,
                html=False,

                # Hardening (robustez)
                remove_multicollinearity=True,
                multicollinearity_threshold=0.95,
                normalize=True,
                normalize_method="zscore",
                fold_shuffle=True,

                remove_outliers=bool(remove_outliers),
                use_gpu=False
            )

            metric_pref = "R2"

            best = compare_models(
                sort=metric_pref,
                n_select=1,
                errors="ignore" if hard_mode else "raise"
            )
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

        st.session_state.leaderboard = leaderboard
        st.session_state.best_model_name = str(best)
        st.session_state.metric_pref = metric_pref
        st.session_state.cv_best = cv_best
        st.session_state.holdout_best = holdout_best
        st.session_state.overfit_gap = overfit_gap
        st.session_state.expl_image_path = expl_image_path
        st.session_state.id_cols_removed = id_cols_removed


leaderboard = st.session_state.leaderboard
best_model_name = st.session_state.best_model_name

if leaderboard is None:
    st.info("Clique em **Rodar AutoML agora** para treinar os modelos e ver o dashboard.")
    st.stop()

st.subheader("4) Dashboard — Ranking dos modelos")
left, right = st.columns([1.6, 1.4])


with left:
    st.markdown("### 🥇 Top 3 modelos")

    if "Model" in leaderboard.columns:
        top3 = leaderboard.head(3).copy()

        metric_col = None
        for col in ["AUC", "Accuracy", "R2", "MAE", "RMSE", "MSE"]:
            if col in top3.columns:
                metric_col = col
                break

        medals = ["🥇", "🥈", "🥉"]
        for idx in range(min(3, len(top3))):
            m = medals[idx]
            model_name = str(top3.iloc[idx]["Model"])
            score_txt = ""
            if metric_col:
                score_txt = f" — {metric_col}: {top3.iloc[idx][metric_col]}"
            st.success(f"{m} {model_name}{score_txt}")


with right:
    # Gráfico simples do ranking (top 15) - MAIOR e mais legível
    st.markdown("### 📊 Ranking (top 15)")
    topn = leaderboard.head(15).copy()

    metric_col = None
    for col in ["AUC", "Accuracy", "R2", "MAE", "RMSE", "MSE"]:
        if col in topn.columns:
            metric_col = col
            break

    if metric_col and "Model" in topn.columns:
        # Ajusta para 15 modelos
        n = len(topn)

        # ✅ FIGURA BEM MAIOR (altura cresce com quantidade de modelos)
        #    (quanto mais alto, mais espaço entre as linhas)
        fig_w = 18
        fig_h = max(10, 0.75 * n + 4)  # altura dinâmica
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

        # Ordem invertida para mostrar o melhor no topo
        models = topn["Model"].astype(str)[::-1]
        values = topn[metric_col][::-1]

        ax.barh(models, values)

        # ✅ fontes maiores
        ax.set_xlabel(metric_col, fontsize=18)
        ax.set_ylabel("Model", fontsize=18)
        ax.tick_params(axis="both", labelsize=18)

        # ✅ mais espaço para os nomes dos modelos (margem esquerda)
        plt.subplots_adjust(left=0.40, right=0.98, top=0.95, bottom=0.08)

        # ✅ grade leve pra facilitar leitura (sem cor explícita)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        # ✅ mostrar o valor no final de cada barra (ajuda muito)
        for i, v in enumerate(values):
            ax.text(v, i, f"  {v:.4f}" if isinstance(v, (float, np.floating)) else f"  {v}",
                    va="center", fontsize=11)

        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Não consegui identificar automaticamente a métrica principal para plotar.")


st.subheader("🧪 Validação (CV vs Holdout) e Overfitting")
metric_pref = st.session_state.metric_pref
cv_best = st.session_state.cv_best
holdout_best = st.session_state.holdout_best
overfit_gap = st.session_state.overfit_gap

if metric_pref and cv_best is not None and holdout_best is not None:
    st.write(f"**Métrica:** {metric_pref}")
    st.write(f"**CV:** {cv_best:.4f}")
    st.write(f"**Holdout:** {holdout_best:.4f}")
    st.write(f"**Gap (CV - Holdout):** {overfit_gap:.4f}")

    if hard_mode and overfit_gap is not None:
        if overfit_gap > 0.08:
            st.error("🔴 Hard Mode: overfitting forte detectado (gap alto).")
        elif overfit_gap > 0.05:
            st.warning("🟡 Hard Mode: possível overfitting (gap moderado).")
        else:
            st.success("🟢 Hard Mode: gap baixo, ok.")
    else:
        if overfit_gap > 0.05:
            st.warning("Possível overfitting: queda relevante no holdout.")
        else:
            st.success("Sem sinal forte de overfitting pelo gap CV→Holdout.")
else:
    st.info("Não foi possível calcular automaticamente CV vs Holdout.")

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
    "⬇️ Baixar Relatório PDF",
    data=pdf_bytes,
    file_name="relatorio_automl.pdf",
    mime="application/pdf"
)
