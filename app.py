"""
Orientador de Estudos — RAG com a grade curricular do curso
Rodar local:  pip install -r requirements.txt  →  streamlit run app.py
"""
import os
import tempfile

import streamlit as st
from openai import OpenAI

from rag import BaseVetorial, Embedder, gerar_roadmap


def segredo(nome, padrao=None):
    try:
        return st.secrets[nome]
    except Exception:
        return os.getenv(nome, padrao)


LLM_BASE_URL = segredo("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = segredo("LLM_API_KEY")
MODELO = segredo("LLM_MODEL", "gemini-flash-latest")


@st.cache_resource(show_spinner="Carregando modelo de embeddings e banco vetorial...")
def carregar_base():
    base = BaseVetorial(Embedder())
    if base.total() == 0:  # primeira execução: indexa a pasta docs/
        base.indexar_pasta()
    return base


@st.cache_resource
def carregar_llm():
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


st.set_page_config(page_title="Orientador de Estudos", page_icon="🎓")
st.title("🎓 Orientador de Estudos")
st.caption("Pergunte sobre o curso e receba um roadmap baseado na grade e na bibliografia oficial.")

base = carregar_base()

with st.sidebar:
    st.header("📚 Base de conhecimento")
    st.metric("Trechos indexados", base.total())
    novos = st.file_uploader("Adicionar PDFs (grade, ementas, PPC)", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
    if novos and st.button("Indexar arquivos"):
        with st.spinner("Indexando..."):
            for arq in novos:
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(arq.name)[1], delete=False) as tmp:
                    tmp.write(arq.getvalue())
                base.indexar_arquivo(tmp.name, nome=arq.name)
        st.rerun()
    if st.button("Reindexar pasta docs/"):
        base.limpar()
        base.indexar_pasta()
        st.rerun()
    k = st.slider("Trechos recuperados", 3, 15, 8)

perfil = st.text_input("Seu perfil (opcional)", placeholder="Ex: 3º semestre, sei Python básico, quero estágio em dados")
pergunta = st.text_area("Sua pergunta",
                        placeholder="Ex: Quero trabalhar com Machine Learning. O que devo estudar e em que ordem?")

if st.button("Gerar roadmap", type="primary") and pergunta.strip():
    if base.total() == 0:
        st.error("A base está vazia. Coloque os PDFs na pasta docs/ ou envie pela barra lateral.")
    else:
        trechos = base.buscar(f"{perfil}\n{pergunta}", k=k)
        with st.spinner("Montando seu roadmap..."):
            st.markdown(gerar_roadmap(carregar_llm(), MODELO, pergunta, trechos, perfil))
        with st.expander("🔎 Trechos usados (fontes)"):
            for t in trechos:
                st.markdown(f"**{t['fonte']}, p. {t['pagina']}** — similaridade {t['similaridade']}")
                st.caption(t["texto"][:600] + ("..." if len(t["texto"]) > 600 else ""))
