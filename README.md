
Sistema completo em Python com interface (Streamlit) para:
- Upload de CSV
- Limpeza e tratamento automático
- Detecção de tarefa (Classificação ou Regressão)
- Treino com 30+ modelos (PyCaret) + ranking do melhor ao pior
- Validação CV vs Holdout + alerta de overfitting
- Explicabilidade (SHAP quando possível + fallback feature importance)
- Exportar relatório final em PDF
- Salvar e baixar o pipeline+modelo (.pkl) para deploy
- Deploy separado com probabilidades (classificação) e download de CSV

---

## 1) Estrutura do projeto



Fabricademodelosml/
app.py
predict_app.py
README.md


---

## 2) Requisitos

- Python 3.9+ (recomendado)
- Pacotes: streamlit, pandas, numpy, matplotlib, scikit-learn, pycaret, reportlab, shap

---

## 3) Instalação (1 vez)

### 3.1 Criar e ativar ambiente virtual (recomendado)

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate


Linux/Mac:

python -m venv .venv
source .venv/bin/activate

3.2 Instalar dependências
pip install -U pip
pip install streamlit pandas numpy matplotlib scikit-learn pycaret reportlab shap

4) Rodar o TREINO (AutoML)

Na pasta do projeto:

streamlit run app.py

Checklist no app

Faça upload do CSV

Confirme/ajuste o target (o app sugere automaticamente)

(Recomendado) Deixe Hard Mode ligado

Clique em Rodar AutoML agora

Veja o ranking e o dashboard

Baixe:

relatorio_automl.pdf

best_model_pipeline.pkl (modelo + pipeline)

Observação: o arquivo best_model_pipeline.pkl também fica na pasta do projeto.

5) Rodar o DEPLOY (Predição com Probabilidades)

Após gerar o best_model_pipeline.pkl:

streamlit run predict_app.py

Checklist no app

Faça upload do CSV para prever

Use o modelo:

padrão best_model_pipeline.pkl na pasta, ou

faça upload do .pkl pela interface

Clique em Rodar agora

Baixe o arquivo predicoes.csv

Saída no deploy

Classificação:

prediction_label (classe prevista)

prediction_score (score/prob da classe prevista)

Score_<classe> (probabilidade por classe, quando suportado por raw_score=True)

(se binário) opção de threshold e prediction_label_thresholded

Regressão:

prediction_label (valor previsto)

6) Solução de problemas (rápido)
6.1 Não gerou Score_<classe> no deploy

Alguns modelos/versões podem não expor probabilidades por classe.

Ainda assim prediction_score geralmente aparece.

Para garantir 100% por classe, dá pra usar predict_proba() do estimador interno (ajuste avançado).

6.2 O treino quebra em algum modelo

Ligue o Hard Mode (ele ativa errors="ignore" no compare_models e validações).

Remova colunas com muitos nulos ou textos de alta cardinalidade.

6.3 CSV novo no deploy dá erro

Confirme que o CSV novo tem colunas compatíveis com as do treino.

Evite mudar nomes de colunas/formatos.

7) Fluxo recomendado (melhor prática)

Treinar no app.py com Hard Mode ligado

Baixar o best_model_pipeline.pkl e guardar versão com data (ex.: best_model_pipeline_2026-02-06.pkl)

Usar no predict_app.py para predições e probabilidades

Re-treinar quando entrar dados novos (mensal/quinzenal)


---