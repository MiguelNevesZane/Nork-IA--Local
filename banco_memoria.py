"""
banco_memoria.py
Base de dados de memória semântica com pipeline bi-encoder + rerank.

Pipeline de busca:
    1. Bi-encoder (all-MiniLM-L6-v2)  → top-K candidatos por similaridade de embedding
    2. Cross-encoder (ms-marco)        → reranqueia os K candidatos por relevância real
    3. Retorna top-N após rerank

Instalação:
    pip install chromadb sentence-transformers
"""

import json
import re
import time
from pathlib import Path

PASTA_DB       = Path("./memoria_db")
COLECAO_NOME   = "memoria_assistente"
MODELO_EMBED   = "all-MiniLM-L6-v2"                          # bi-encoder, ~90MB
MODELO_RERANK  = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # cross-encoder, ~65MB

TOP_K_INICIAL  = 20   # candidatos pelo bi-encoder
TOP_N_FINAL    = 5    # retornados após rerank

# Padrões que indicam que o usuário quer salvar algo ou revelou fato pessoal
_TRIGGERS = [
    r"\bmeu nome\b",
    r"\beu sou\b",
    r"\beu trabalho\b",
    r"\beu uso\b",
    r"\beu prefiro\b",
    r"\bnão gosto\b",
    r"\bgosto de\b",
    r"\blembra que\b",
    r"\bsalva (?:que|isso|este)\b",
    r"\badiciona (?:ao banco|na memoria|isso)\b",
    r"\bsou desenvolvedor\b",
    r"\bsou (?:um |uma )?\w+",
    r"\bmoro (?:em|no|na)\b",
    r"\btenho \d+",
]


class BancoMemoria:
    """
    Memória semântica persistente com carregamento lazy dos modelos.

    Fase 1 (startup, rápida): conecta ao ChromaDB — sem download.
    Fase 2 (lazy, primeira busca/escrita): carrega bi-encoder + cross-encoder.
    Assim o chat inicia imediatamente; modelos são baixados (~160MB) só quando usados.
    """

    def __init__(self):
        self.disponivel      = False   # True após ChromaDB conectar
        self.modelos_prontos = False   # True após modelos carregarem
        self._colecao        = None
        self._encoder        = None
        self._reranker       = None
        self._init_db()

    # ── Fase 1: conecta ao banco (sem modelos) ───────────────

    def _init_db(self):
        try:
            import chromadb

            PASTA_DB.mkdir(exist_ok=True)
            client        = chromadb.PersistentClient(path=str(PASTA_DB))
            self._colecao = client.get_or_create_collection(
                name=COLECAO_NOME,
                metadata={"hnsw:space": "cosine"},
            )
            self.disponivel = True
            print(f"  Banco de memoria: {self._colecao.count()} registro(s) | modelos: lazy")

        except ImportError:
            print("  Banco de memoria desativado. Instale: pip install chromadb sentence-transformers")
        except Exception as e:
            print(f"  Banco de memoria desativado: {e}")

    # ── Fase 2: carrega modelos (lazy, uma vez) ──────────────

    def _garantir_modelos(self) -> bool:
        """Carrega bi-encoder e cross-encoder na primeira chamada que precisar deles."""
        if self.modelos_prontos:
            return True
        if not self.disponivel:
            return False
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder

            print("  Carregando modelos de embedding (primeira vez, aguarde)...", end="", flush=True)
            self._encoder  = SentenceTransformer(MODELO_EMBED)
            self._reranker = CrossEncoder(MODELO_RERANK)
            self.modelos_prontos = True
            print(" OK")
            return True

        except ImportError:
            print("\n  sentence-transformers nao instalado: pip install sentence-transformers")
            self.disponivel = False
            return False
        except Exception as e:
            print(f"\n  Falha ao carregar modelos: {e}")
            self.disponivel = False
            return False

    # ── Escrita ──────────────────────────────────────────────

    def adicionar(self, texto: str, fonte: str = "conversa", extra: dict | None = None) -> str:
        """
        Adiciona um documento ao banco.
        Dispara carregamento dos modelos na primeira chamada.
        Retorna o ID gerado ou string vazia se banco indisponível.
        """
        if not self.disponivel or not texto.strip():
            return ""
        # adicionar não precisa dos modelos (ChromaDB faz embedding automático)
        # mas garante consistência com o pipeline de busca
        _ = self._garantir_modelos()  # carrega em background; adicionar funciona mesmo sem

        doc_id = f"mem_{int(time.time() * 1000)}"
        meta   = {"fonte": fonte, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        if extra:
            meta.update(extra)

        self._colecao.add(documents=[texto], ids=[doc_id], metadatas=[meta])
        return doc_id

    def remover(self, doc_id: str):
        if self.disponivel:
            self._colecao.delete(ids=[doc_id])

    # ── Leitura com rerank ───────────────────────────────────

    def buscar(self, query: str) -> list[dict]:
        """
        Pipeline completo:
          bi-encoder → top TOP_K_INICIAL candidatos
          cross-encoder → reranqueia
          retorna top TOP_N_FINAL
        Carrega modelos na primeira chamada (lazy).
        """
        if not self.disponivel or self._colecao.count() == 0:
            return []
        if not self._garantir_modelos():
            return []

        n_busca = min(TOP_K_INICIAL, self._colecao.count())

        # Passo 1 — busca inicial por embedding (rápida)
        r = self._colecao.query(query_texts=[query], n_results=n_busca)
        docs  = r.get("documents",  [[]])[0]
        ids   = r.get("ids",        [[]])[0]
        metas = r.get("metadatas",  [[]])[0]

        if not docs:
            return []

        # Passo 2 — reranking com cross-encoder (preciso)
        scores = self._reranker.predict([(query, d) for d in docs])

        ranked = sorted(
            zip(scores, docs, ids, metas),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            {"score": float(s), "texto": d, "id": i, "meta": m}
            for s, d, i, m in ranked[:TOP_N_FINAL]
        ]

    # ── Listagem ─────────────────────────────────────────────

    def listar(self, limite: int = 30) -> list[dict]:
        if not self.disponivel:
            return []
        r = self._colecao.get(limit=limite)
        return [
            {"id": i, "texto": d, "meta": m}
            for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])
        ]

    def total(self) -> int:
        return self._colecao.count() if self.disponivel else 0

    # ── Formatação para injeção no prompt ────────────────────

    def formatar_contexto(self, query: str) -> str:
        """
        Busca, reranqueia e retorna texto formatado para
        ser injetado como contexto no prompt do modelo.
        """
        mems = self.buscar(query)
        if not mems:
            return ""

        linhas = ["MEMORIAS RELEVANTES (reranqueadas):"]
        for i, m in enumerate(mems, 1):
            linhas.append(f"  [{i}] {m['texto']}  (relevância: {m['score']:.2f})")
        return "\n".join(linhas)

    # ── Detecção automática de fatos salváreis ────────────────

    @staticmethod
    def deve_salvar(texto: str) -> bool:
        """Heurística rápida — detecta se a mensagem contém fato que vale salvar."""
        t = texto.lower()
        return any(re.search(p, t) for p in _TRIGGERS)

    # ── Extração de fatos via modelo ─────────────────────────

    @staticmethod
    def prompt_extracao(pergunta_usuario: str, resposta_assistente: str) -> str:
        """
        Retorna o prompt que o motor deve enviar ao modelo para extrair
        fatos salváreis da troca de mensagens.
        """
        return (
            "Analise a troca abaixo e extraia apenas fatos concretos e permanentes "
            "sobre o usuário que valha a pena lembrar em conversas futuras.\n\n"
            "Regras:\n"
            "- Apenas fatos sobre o usuário (nome, preferências, profissão, etc.)\n"
            "- Frase curta e autocontida, máximo 120 caracteres cada\n"
            "- Se não há nada relevante, retorne lista vazia\n"
            "- Responda APENAS com JSON válido, sem explicação\n\n"
            f"Usuário disse: {pergunta_usuario}\n"
            f"Assistente respondeu: {resposta_assistente[:300]}\n\n"
            'Formato: {"fatos": ["fato 1", "fato 2"]}'
        )

    @staticmethod
    def parse_fatos(resposta_json: str) -> list[str]:
        """Extrai lista de fatos do JSON retornado pelo modelo."""
        try:
            # Remove possível markdown ao redor do JSON
            limpo = re.sub(r"```(?:json)?\n?(.*?)```", r"\1", resposta_json, flags=re.DOTALL)
            dados = json.loads(limpo.strip())
            return [f for f in dados.get("fatos", []) if isinstance(f, str) and f.strip()]
        except Exception:
            return []
