# 🎓 Orientador de Estudos — RAG com a grade curricular

Pergunte sobre o curso e receba um roadmap de estudos baseado nas ementas e na bibliografia oficial.

**Stack:** ChromaDB (banco vetorial) · fastembed multilíngue (embeddings locais, grátis) · DeepSeek (LLM) · Streamlit

## Arquitetura
PDFs do curso → chunks (1200 caracteres, sobreposição de 200) → embeddings (384 dim) → ChromaDB
Pergunta → embedding → busca top-k por similaridade de cosseno → DeepSeek gera o roadmap citando as fontes

## Rodar local
```bash
pip install -r requirements.txt
# copie .streamlit/secrets.toml.exemplo para .streamlit/secrets.toml e coloque sua chave
streamlit run app.py
```
Coloque os PDFs em `docs/` antes de rodar. Na primeira execução ele baixa o modelo de embeddings (~220 MB) e indexa tudo.
Mudou os PDFs? Clique em "Reindexar pasta docs/" na barra lateral.

## Subir na nuvem (Streamlit Community Cloud — grátis)
1. Crie um repositório no GitHub e envie esta pasta (com os PDFs em `docs/`).
2. Acesse share.streamlit.io → "Create app" → escolha o repositório e o arquivo `app.py`.
3. Em "Advanced settings → Secrets", cole:
   ```
   DEEPSEEK_API_KEY = "sua_chave"
   DEEPSEEK_MODEL = "deepseek-v4-flash"
   ```
4. Deploy. A primeira carga demora 1–2 min (download do modelo + indexação).
