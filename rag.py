"""
Núcleo do RAG: leitura dos documentos, chunking, embeddings e banco vetorial (ChromaDB).
"""
import os
import sys
import hashlib
from pathlib import Path

# Streamlit Cloud/Linux: o ChromaDB precisa de um SQLite >= 3.35
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from pypdf import PdfReader

BASE = Path(__file__).parent
DOCS_DIR = BASE / "docs"
DB_DIR = BASE / "chroma_db"
COLECAO = "grade_curricular"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # multilíngue, 384 dim, ~220 MB


class Embedder:
    """Gera embeddings localmente (CPU, sem custo) com fastembed."""

    def __init__(self, modelo=EMBED_MODEL):
        from fastembed import TextEmbedding
        self.modelo = TextEmbedding(modelo)

    def __call__(self, textos):
        return [v.tolist() for v in self.modelo.embed(textos)]


# ----------------------------------------------------------------------
# Leitura e chunking
# ----------------------------------------------------------------------
def ler_documento(caminho):
    """Retorna lista de (pagina, texto)."""
    caminho = Path(caminho)
    if caminho.suffix.lower() == ".pdf":
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(caminho).pages)]
    return [(1, caminho.read_text(encoding="utf-8", errors="ignore"))]


def quebrar_em_chunks(texto, tamanho=1200, sobreposicao=200):
    texto = " ".join(texto.split())
    chunks, inicio = [], 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        if fim < len(texto):  # tenta cortar no fim de uma frase
            ponto = texto.rfind(". ", inicio + tamanho // 2, fim)
            if ponto != -1:
                fim = ponto + 1
        chunks.append(texto[inicio:fim].strip())
        if fim >= len(texto):
            break
        inicio = fim - sobreposicao
    return [c for c in chunks if len(c) > 50]


# ----------------------------------------------------------------------
# Banco vetorial
# ----------------------------------------------------------------------
class BaseVetorial:
    def __init__(self, embedder, db_dir=DB_DIR):
        self.embedder = embedder
        self.cliente = chromadb.PersistentClient(path=str(db_dir))
        self.colecao = self.cliente.get_or_create_collection(COLECAO, metadata={"hnsw:space": "cosine"})

    def total(self):
        return self.colecao.count()

    def limpar(self):
        self.cliente.delete_collection(COLECAO)
        self.colecao = self.cliente.get_or_create_collection(COLECAO, metadata={"hnsw:space": "cosine"})

    def indexar_arquivo(self, caminho, nome=None):
        nome = nome or Path(caminho).name
        docs, metas, ids = [], [], []
        for pagina, texto in ler_documento(caminho):
            for j, chunk in enumerate(quebrar_em_chunks(texto)):
                docs.append(chunk)
                metas.append({"fonte": nome, "pagina": pagina})
                ids.append(hashlib.md5(f"{nome}|{pagina}|{j}|{chunk[:50]}".encode()).hexdigest())
        for i in range(0, len(docs), 64):  # em lotes
            self.colecao.upsert(ids=ids[i:i + 64], documents=docs[i:i + 64],
                                metadatas=metas[i:i + 64], embeddings=self.embedder(docs[i:i + 64]))
        return len(docs)

    def indexar_pasta(self, pasta=DOCS_DIR):
        total = 0
        for arq in sorted(Path(pasta).glob("*")):
            if arq.suffix.lower() in {".pdf", ".txt", ".md"}:
                total += self.indexar_arquivo(arq)
        return total

    def buscar(self, pergunta, k=8):
        r = self.colecao.query(query_embeddings=self.embedder([pergunta]), n_results=min(k, max(self.total(), 1)))
        return [{"texto": d, "fonte": m["fonte"], "pagina": m["pagina"], "similaridade": round(1 - dist, 3)}
                for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


# ----------------------------------------------------------------------
# Geração do roadmap
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """Você é um orientador acadêmico do curso do aluno. Você responde SOMENTE com base nos
trechos da grade curricular, ementas e bibliografia fornecidos no CONTEXTO.

Monte a resposta em Markdown neste formato:

## 🎯 Diagnóstico
1-3 frases sobre o objetivo do aluno e quais disciplinas do curso se relacionam com ele.

## 🗺️ Roadmap de estudos
Divida em 3 a 5 etapas, da base ao avançado, respeitando pré-requisitos. Para cada etapa:
### Etapa N — <título> (<duração sugerida>)
- **Disciplinas do curso:** ...
- **O que estudar:** tópicos concretos tirados das ementas
- **Bibliografia do curso:** livros com autor, exatamente como aparecem no contexto
- **Prática:** um exercício ou miniprojeto

## 📌 Próximo passo
Uma ação concreta para hoje.

REGRAS:
- NUNCA invente disciplinas, livros ou autores que não estejam no CONTEXTO.
- Se o contexto não tiver bibliografia para uma etapa, escreva "não consta na bibliografia do curso".
- Ao citar algo, indique a fonte entre colchetes, ex: [grade.pdf, p. 3].
"""


def montar_contexto(trechos):
    return "\n\n".join(f"[{t['fonte']}, p. {t['pagina']}]\n{t['texto']}" for t in trechos)


def gerar_roadmap(cliente_llm, modelo, pergunta, trechos, perfil=""):
    mensagem = f"PERFIL DO ALUNO: {perfil or 'não informado'}\n\nPERGUNTA: {pergunta}\n\nCONTEXTO:\n{montar_contexto(trechos)}"
    resp = cliente_llm.chat.completions.create(
        model=modelo, temperature=0.3,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": mensagem}],
    )
    return resp.choices[0].message.content
