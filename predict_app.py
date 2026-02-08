# Deploy App (Streamlit) - Predições com Probabilidades
# - Carrega best_model_pipeline.pkl (PyCaret) ou upload do .pkl
# - Upload de CSV novo
# - Detecta Classificação vs Regressão
# - Classificação: prediction_label + prediction_score + Score_<classe> (raw_score=True)
# - Download do CSV com previsões

import io
import os
import tempfile
from typing import Optional, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Deploy Premium - Predições com Probabilidades", layout="wide")
st.title("🧠 Deploy Premium: Predições (Classificação com Probabilidades)")


def read_csv_uploaded(uploaded_file, sep: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=sep, encoding=encoding)


def ensure_model_file_on_disk(uploaded_model) -> Tuple[str, str]:
    """
    Salva o .pkl em diretório temporário e devolve:
      - model_dir
      - model_base (sem .pkl)
    """
    filename = uploaded_model.name
    if not filename.lower().endswith(".pkl"):
        raise ValueError("O modelo precisa ser um .pkl (ex.: best_model_pipeline.pkl).")

    model_base = filename[:-4]
    tmpdir = tempfile.mkdtemp(prefix="pycaret_model_")
    model_path = os.path.join(tmpdir, filename)

    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())

    return tmpdir, model_base


def try_classification_predict(model_base: str, df_new: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        from pycaret.classification import load_model as cls_load_model, predict_model as cls_predict_model
        model = cls_load_model(model_base)
        pred = cls_predict_model(model, data=df_new, raw_score=True)  # garante Score_<classe> quando suportado
        return pred
    except Exception:
        return None


def try_regression_predict(model_base: str, df_new: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        from pycaret.regression import load_model as reg_load_model, predict_model as reg_predict_model
        model = reg_load_model(model_base)
        pred = reg_predict_model(model, data=df_new)
        return pred
    except Exception:
        return None


def detect_probability_columns(pred_df: pd.DataFrame) -> list:
    prob_cols = []
    for c in pred_df.columns:
        lc = c.lower()
        if c == "prediction_score":
            continue
        if lc.startswith("score_") or ("score" in lc and c != "prediction_score"):
            prob_cols.append(c)
    return prob_cols


def apply_threshold_binary(pred_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    prob_cols = [c for c in pred_df.columns if c.lower().startswith("score_")]
    if len(prob_cols) != 2:
        return pred_df

    prob_cols_sorted = sorted(prob_cols)
    pos_col = prob_cols_sorted[-1]
    neg_col = prob_cols_sorted[0]

    out = pred_df.copy()
    out["prediction_label_thresholded"] = out[pos_col].apply(
        lambda p: pos_col.replace("Score_", "") if p >= threshold else neg_col.replace("Score_", "")
    )
    out["threshold_used"] = threshold
    out["positive_class_used"] = pos_col.replace("Score_", "")
    out["negative_class_used"] = neg_col.replace("Score_", "")
    return out


with st.sidebar:
    st.header("1) Modelo (.pkl)")
    st.write("Você pode subir o `.pkl` ou usar `best_model_pipeline.pkl` na pasta.")
    uploaded_model = st.file_uploader("Upload do modelo .pkl", type=["pkl"])

    st.divider()
    st.header("2) CSV de entrada")
    uploaded_csv = st.file_uploader("Upload do CSV para prever", type=["csv"])
    sep = st.text_input("Separador (padrão ,)", value=",")
    encoding = st.text_input("Encoding (padrão utf-8)", value="utf-8")

    st.divider()
    st.header("3) Opções")
    show_full_table = st.checkbox("Mostrar tabela completa", value=False)
    max_preview = st.slider("Linhas de preview", 20, 500, 100, 10)


if uploaded_csv is None:
    st.info("Envie um CSV na barra lateral para gerar previsões.")
    st.stop()

try:
    df_new = read_csv_uploaded(uploaded_csv, sep=sep, encoding=encoding)
except Exception as e:
    st.error(f"Erro ao ler CSV: {e}")
    st.stop()

st.subheader("📄 Prévia do CSV de entrada")
st.dataframe(df_new.head(max_preview), use_container_width=True)
st.caption(f"Linhas: {df_new.shape[0]} | Colunas: {df_new.shape[1]}")

# modelo: upload ou arquivo padrão na pasta
model_dir = ""
model_base = ""

if uploaded_model is not None:
    try:
        model_dir, model_base = ensure_model_file_on_disk(uploaded_model)
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    default_base = "best_model_pipeline"
    default_path = f"{default_base}.pkl"
    if not os.path.exists(default_path):
        st.warning(
            f"Não enviou modelo. Para usar o padrão, coloque **{default_path}** na pasta do app, "
            "ou faça upload do .pkl."
        )
        st.stop()
    model_base = default_base
    model_dir = os.getcwd()

# PyCaret procura pelo nome base no diretório atual
os.chdir(model_dir)

st.subheader("🤖 Modelo selecionado")
st.write(f"**Nome base:** `{model_base}`")
st.write(f"**Diretório:** `{model_dir}`")

st.subheader("▶️ Rodar predição")
run_btn = st.button("Rodar agora", type="primary")
if not run_btn:
    st.stop()

with st.spinner("Carregando modelo e gerando previsões..."):
    pred_cls = try_classification_predict(model_base, df_new)
    if pred_cls is not None:
        task = "classification"
        pred_df = pred_cls
    else:
        pred_reg = try_regression_predict(model_base, df_new)
        if pred_reg is None:
            st.error("Não consegui carregar/predizer nem como classificação nem como regressão.")
            st.info("Possíveis causas: .pkl não é PyCaret, dependências faltando, ou arquivo incompatível.")
            st.stop()
        task = "regression"
        pred_df = pred_reg

if task == "classification":
    st.success("✅ Modelo detectado como **CLASSIFICAÇÃO**")

    prob_cols = detect_probability_columns(pred_df)

    front = [c for c in ["prediction_label", "prediction_score"] if c in pred_df.columns]
    rest = [c for c in pred_df.columns if c not in set(front + prob_cols)]
    pred_df = pred_df[front + prob_cols + rest]

    st.subheader("📊 Probabilidades")
    if prob_cols:
        st.write("Colunas de probabilidade por classe detectadas:", prob_cols)
    else:
        st.warning("Não encontrei colunas `Score_*`. Ainda assim `prediction_score` pode existir.")

    # threshold para binário (se houver 2 Score_*)
    score_cols = [c for c in pred_df.columns if c.lower().startswith("score_")]
    if len(score_cols) == 2:
        st.subheader("🎚️ Threshold (binário)")
        thr = st.slider("Threshold para classe positiva", 0.0, 1.0, 0.5, 0.01)
        pred_out = apply_threshold_binary(pred_df, thr)
        st.caption("Coluna criada: `prediction_label_thresholded`.")
    else:
        pred_out = pred_df

    st.subheader("✅ Resultado")
    if show_full_table:
        st.dataframe(pred_out, use_container_width=True)
    else:
        st.dataframe(pred_out.head(max_preview), use_container_width=True)

else:
    st.success("✅ Modelo detectado como **REGRESSÃO**")
    st.subheader("✅ Resultado")
    if show_full_table:
        st.dataframe(pred_df, use_container_width=True)
    else:
        st.dataframe(pred_df.head(max_preview), use_container_width=True)
    pred_out = pred_df

st.subheader("⬇️ Baixar CSV com previsões")
out = io.StringIO()
pred_out.to_csv(out, index=False)

st.download_button(
    "Baixar predições (CSV)",
    data=out.getvalue().encode("utf-8"),
    file_name="predicoes.csv",
    mime="text/csv",
)

st.caption("Classificação: usamos `raw_score=True` para obter probabilidades por classe como `Score_<classe>` quando suportado.")
