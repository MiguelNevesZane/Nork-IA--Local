"""
motor_ia_avancado.py
Motor de inferência local via Ollama com 8 modos de resposta.
Usa /api/chat (roles estruturados) para evitar alucinações por histórico.

Modos:
    1. CoT profundo       — 30 etapas de raciocínio
    2. Código + few-shot  — exemplos de referência
    3. Código + verif.    — executa e autocorrige
    4. Self-consistency   — vota entre 3 respostas
    5. RAG local          — usa pasta ./docs/
    6. Adaptativo         — aprende seu perfil
    7. Compacto           — respostas curtas e diretas
    8. Ultra-pensamento   — 30 etapas + 2ª perspectiva + síntese

Memória semântica (banco_memoria.py):
    - Detecta fatos relevantes na conversa e salva automaticamente
    - Recupera com bi-encoder + rerank antes de cada resposta
    - Comandos: 'banco' | 'memorias' | 'salvar <texto>' | 'esquecer <id>'

Dependências:
    pip install requests
    pip install chromadb sentence-transformers   (banco de memória + RAG)
"""

import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import requests

from banco_memoria import BancoMemoria

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────

OLLAMA_CHAT = "http://localhost:11434/api/chat"
OLLAMA_TAGS = "http://localhost:11434/api/tags"
MODELO      = "deepseek-r1:8b"
TIMEOUT     = 300          # segundos — 30 etapas geram muitos tokens
NUM_PREDICT = 8192         # tokens máximos por resposta
SC_N        = 3            # tentativas no self-consistency
MAX_FIX     = 2            # tentativas de autocorreção de código
PASTA_DOCS  = Path("./docs")
SEP         = "─" * 58

# ─────────────────────────────────────────────────────────────
# REGRAS BASE (aplicadas em todos os modos)
# ─────────────────────────────────────────────────────────────

_REGRAS_BASE = """
REGRAS OBRIGATÓRIAS — sem exceção:
- Responda SEMPRE em português brasileiro
- NUNCA use emojis de qualquer tipo
- Nunca invente informações; se não souber, diga claramente
- Seja preciso, factual e direto
"""

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────

SYSTEM_COT = _REGRAS_BASE + """
Você é um assistente com raciocínio profundo e rigoroso.

PROCESSO OBRIGATÓRIO antes de qualquer resposta:
Execute TODAS as 30 etapas dentro de <thinking></thinking>.
Cada etapa deve ter pelo menos 2 linhas de raciocínio real.

ETAPA 1  — LEITURA LITERAL      Qual é o enunciado exato? Repita com suas palavras.
ETAPA 2  — INTENCAO REAL        O que o usuário realmente quer? Há intenção implícita?
ETAPA 3  — CONTEXTO             Em que contexto a pergunta faz sentido?
ETAPA 4  — CONHECIMENTO         Que conceitos são necessários para responder bem?
ETAPA 5  — DECOMPOSICAO         Quebre em todas as sub-perguntas necessárias.
ETAPA 6  — ORDEM LOGICA         Em que ordem responder para o raciocínio ser coerente?
ETAPA 7  — HIPOTESE INICIAL     Qual é minha resposta instintiva? Por que penso isso?
ETAPA 8  — EVIDENCIAS           Que fatos, dados ou princípios sustentam a hipótese?
ETAPA 9  — CAUSAS E ORIGENS     Quais são as causas raiz? Vá além do superficial.
ETAPA 10 — MECANISMOS           Como o fenômeno funciona internamente, passo a passo?
ETAPA 11 — CONSEQUENCIAS        Efeitos diretos, de segunda e terceira ordem.
ETAPA 12 — EXEMPLOS             Cite pelo menos 2 exemplos reais e concretos.
ETAPA 13 — CASOS EXTREMOS       O que acontece nos limites? Onde o raciocínio falha?
ETAPA 14 — EXCECOES             Existem exceções? O que elas revelam sobre a regra?
ETAPA 15 — HISTORICO            Como esse tema evoluiu ao longo do tempo?
ETAPA 16 — CIENCIA              O que a ciência estabeleceu? Há consenso ou debate?
ETAPA 17 — PRATICA              Como se aplica na vida real e em situações concretas?
ETAPA 18 — CRITICA              Quem discordaria? Qual é o argumento mais forte contra?
ETAPA 19 — REFUTACAO            Como respondo ao contra-argumento? Minha posição se sustenta?
ETAPA 20 — SUPOSICOES           Que pressupostos estou fazendo? São válidos?
ETAPA 21 — VIES                 Que viés pode estar distorcendo meu raciocínio?
ETAPA 22 — PONTO CEGO           Que ângulo ou dimensão ainda não considerei?
ETAPA 23 — ANALOGIAS            Existe algo em outro domínio que ilumina esse tema?
ETAPA 24 — FILOSOFIA            Que questões mais profundas esse tema levanta?
ETAPA 25 — SINTESE PARCIAL      Com tudo isso, qual é a conclusão intermediária mais sólida?
ETAPA 26 — CONSISTENCIA         Minha conclusão é consistente com todas as etapas?
ETAPA 27 — REVISAO CRITICA      Há saltos lógicos, generalizações ou lacunas?
ETAPA 28 — REFINAMENTO          Depois da revisão, como aprimoro a conclusão?
ETAPA 29 — CLAREZA              Como estruturo para ser maximamente claro e útil?
ETAPA 30 — CONCLUSAO            Minha resposta final, completa e bem fundamentada.

EXCECAO: Para cumprimentos simples (oi, tudo bem, bom dia), responda diretamente sem etapas.

Formato obrigatório para perguntas não-triviais:
<thinking>
ETAPA 1 — LEITURA LITERAL
[raciocínio]

ETAPA 2 — INTENCAO REAL
[raciocínio]

[... todas as 30 ...]

ETAPA 30 — CONCLUSAO
[raciocínio]
</thinking>
<answer>
[Resposta clara, estruturada e completa]
</answer>"""

SYSTEM_CODIGO = _REGRAS_BASE + """
Você é um especialista em Python.

Processo:
- Raciocine antes de escrever código entre <thinking></thinking>
- Código final entre <answer></answer>
- Código completo, funcional, com tratamento de erros
- Simples e legível; comente apenas o não-óbvio
- Sem bibliotecas desnecessárias
- Sem pseudocódigo ou trechos omitidos"""

SYSTEM_COMPACTO = _REGRAS_BASE + """
Você é um assistente direto e conciso.

Regras absolutas:
- Máximo 3 frases na resposta
- Sem introduções, sem rodeios, sem repetição da pergunta
- Apenas a informação essencial solicitada
- Sem raciocínio visível"""

SYSTEM_ADAPTATIVO = _REGRAS_BASE + """
Você é um assistente que se adapta ao perfil do usuário.

Perfil atual do usuário:
{perfil}

Instruções:
- Analise silenciosamente o estilo e nível técnico da mensagem
- Adapte linguagem, profundidade e formato ao perfil detectado
- Não mencione que está se adaptando
- Raciocínio de adaptação dentro de <thinking></thinking>
- Resposta final em <answer></answer>"""

SYSTEM_ULTRA = _REGRAS_BASE + """
Você é um sistema de análise de máxima profundidade.

FASE 1 — ANALISE PRIMARIA (30 etapas obrigatórias):
Execute as mesmas 30 etapas do CoT com máximo rigor e detalhe.

FASE 2 — PERSPECTIVA ALTERNATIVA:
Reanalise o problema do ponto de vista de uma área completamente diferente.
O que um especialista de outra disciplina diria? Que insights ele traria?

FASE 3 — SINTESE FINAL:
Integre as duas análises.
Onde convergem? Onde divergem? Qual é a verdade mais completa que emerge?

Tudo dentro de <thinking></thinking>.
Resposta final sintetizada em <answer></answer>."""

FEW_SHOT_CODIGO = """\
Exemplos de qualidade esperada:

--- Exemplo 1 ---
Pergunta: Como ler um arquivo CSV e filtrar linhas onde a coluna 'status' == 'ativo'?
Resposta:
import csv

def filtrar_ativos(caminho: str) -> list[dict]:
    resultado = []
    with open(caminho, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for linha in reader:
            if linha.get('status', '').lower() == 'ativo':
                resultado.append(dict(linha))
    return resultado

--- Exemplo 2 ---
Pergunta: Retry automático em requisição HTTP com backoff exponencial.
Resposta:
import time
import requests

def get_com_retry(url: str, max_tentativas: int = 3) -> requests.Response:
    for i in range(max_tentativas):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if i == max_tentativas - 1:
                raise
            espera = 2 ** i
            print(f"Tentativa {i+1} falhou. Aguardando {espera}s...")
            time.sleep(espera)
---
"""

# ─────────────────────────────────────────────────────────────
# ANIMACAO DE PENSAMENTO
# ─────────────────────────────────────────────────────────────

class Spinner:
    """
    Exibe animação de carregamento enquanto o modelo processa.
    Roda em thread separada para não bloquear a chamada principal.
    """
    _QUADROS = [
        "[          ]",
        "[==        ]",
        "[====      ]",
        "[======    ]",
        "[========  ]",
        "[==========]",
        "[  ========]",
        "[    ======]",
        "[      ====]",
        "[        ==]",
    ]

    def __init__(self, mensagem: str = "Processando"):
        self._msg   = mensagem
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        i = 0
        while not self._stop.is_set():
            frame = self._QUADROS[i % len(self._QUADROS)]
            print(f"\r  {frame}  {self._msg}...", end="", flush=True)
            time.sleep(0.1)
            i += 1

    def atualizar(self, mensagem: str):
        self._msg = mensagem

    def iniciar(self):
        self._thread.start()

    def parar(self):
        self._stop.set()
        self._thread.join()
        print(f"\r{' ' * 70}\r", end="", flush=True)

# ─────────────────────────────────────────────────────────────
# CLIENTE OLLAMA  —  /api/chat  (roles estruturados)
# ─────────────────────────────────────────────────────────────

def _chamar_ollama(
    mensagens: list[dict],
    system: str,
    temperature: float = 0.0,
    spinner: Spinner | None = None,
) -> str:
    """
    Chama POST /api/chat com roles user/assistant separados.
    Usa stream=True internamente mas acumula o texto silenciosamente.
    O spinner (se fornecido) é atualizado conforme etapas aparecem no stream.
    """
    payload = {
        "model":   MODELO,
        "messages": [{"role": "system", "content": system}] + mensagens,
        "options": {
            "temperature": temperature,
            "top_p":       0.95 if temperature > 0 else 1.0,
            "num_predict": NUM_PREDICT,
        },
        "stream": True,
    }

    try:
        resp = requests.post(OLLAMA_CHAT, json=payload, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("\n[ERRO] Ollama nao esta rodando. Execute em outro terminal: ollama serve")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n[ERRO] Ollama retornou: {e}")
        sys.exit(1)

    texto      = ""
    etapa_atual = 0

    for linha in resp.iter_lines():
        if not linha:
            continue
        dados = json.loads(linha)
        token = dados.get("message", {}).get("content", "")
        texto += token

        # Atualiza spinner com a etapa de raciocínio atual
        if spinner:
            m = re.search(r"ETAPA\s+(\d+)", texto)
            if m:
                n = int(m.group(1))
                if n != etapa_atual:
                    etapa_atual = n
                    spinner.atualizar(f"Etapa {n}/30 do raciocínio")

        if dados.get("done"):
            break

    return texto

# ─────────────────────────────────────────────────────────────
# EXTRAÇÃO DE TAGS
# ─────────────────────────────────────────────────────────────

def _tag(texto: str, nome: str) -> str:
    m = re.search(rf"<{nome}>(.*?)</{nome}>", texto, re.DOTALL)
    return m.group(1).strip() if m else ""

def _pensamento(texto: str) -> str:
    """Extrai raciocínio de <think> (deepseek nativo) ou <thinking> (nosso prompt)."""
    for t in ("think", "thinking"):
        r = _tag(texto, t)
        if r:
            return r
    return ""

def _resposta(texto: str) -> str:
    """Extrai resposta de <answer>, ou texto após o fechamento de thinking."""
    ans = _tag(texto, "answer")
    if ans:
        return ans
    for fechamento in ("</think>", "</thinking>"):
        if fechamento in texto:
            parte = texto.split(fechamento, 1)[1].strip()
            if parte:
                return parte
    return texto.strip()

def _codigo(texto: str) -> str:
    base = _tag(texto, "answer") or texto
    return re.sub(r"```(?:python)?\n?(.*?)```", r"\1", base, flags=re.DOTALL).strip()

# ─────────────────────────────────────────────────────────────
# EXIBIÇÃO FORMATADA
# ─────────────────────────────────────────────────────────────

def _exibir(pensamento: str, resposta: str, modo_nome: str, duracao: float, mostrar_pens: bool = True):
    if pensamento and mostrar_pens:
        print(f"\n{SEP}")
        print("  RACIOCINIO INTERNO")
        print(SEP)
        for linha in pensamento.splitlines():
            print(f"  {linha}")

    print(f"\n{SEP}")
    print("  RESPOSTA")
    print(SEP)
    print(resposta)
    print(f"\n[{duracao:.1f}s | {modo_nome}]")

# ─────────────────────────────────────────────────────────────
# MODOS DE RESPOSTA
# ─────────────────────────────────────────────────────────────

def modo_cot(mensagens: list[dict]) -> tuple[str, str]:
    """Modo 1 — Chain-of-Thought com 30 etapas obrigatórias."""
    sp = Spinner("Iniciando raciocínio profundo")
    sp.iniciar()
    bruto = _chamar_ollama(mensagens, SYSTEM_COT, spinner=sp)
    sp.parar()
    return _resposta(bruto), _pensamento(bruto)


def modo_codigo(mensagens: list[dict], verbose: bool = True) -> tuple[str, str]:
    """Modo 2 — Few-shot + CoT para Python."""
    if verbose:
        print(f"\n  Gerando codigo com exemplos de referencia...\n")

    ultima = mensagens[-1]["content"]
    msgs   = mensagens[:-1] + [{"role": "user", "content": f"{FEW_SHOT_CODIGO}\n---\n{ultima}"}]
    bruto  = _chamar_ollama(msgs, SYSTEM_CODIGO)
    return _codigo(bruto), _pensamento(bruto)


def _executar_codigo(codigo: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(codigo)
        caminho = f.name
    try:
        r = subprocess.run([sys.executable, caminho], capture_output=True, text=True, timeout=15)
        return (r.returncode == 0), (r.stdout if r.returncode == 0 else r.stderr)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: codigo demorou mais de 15 segundos"
    finally:
        Path(caminho).unlink(missing_ok=True)


def modo_verificador(mensagens: list[dict]) -> tuple[str, str]:
    """Modo 3 — Gera código, executa, autocorrige até MAX_FIX vezes."""
    print(f"\n  Gerando e testando codigo automaticamente...\n")

    codigo, pens = modo_codigo(mensagens, verbose=False)

    for tentativa in range(MAX_FIX + 1):
        print(f"  --- Tentativa {tentativa + 1} ---")
        print(codigo)
        print()

        ok, output = _executar_codigo(codigo)
        if ok:
            print(f"  Codigo executou sem erros. Output: {output or '(sem output)'}")
            return codigo, pens

        print(f"  Erro:\n{output}\n")
        if tentativa < MAX_FIX:
            print(f"  Pedindo correcao ({tentativa + 2}/{MAX_FIX + 1})...\n")
            msgs_fix = mensagens + [
                {"role": "assistant", "content": codigo},
                {"role": "user",      "content": f"Erro ao executar:\n{output}\n\nCorrenja entre <answer></answer>."},
            ]
            bruto  = _chamar_ollama(msgs_fix, SYSTEM_CODIGO)
            codigo = _codigo(bruto)

    print("  Nao foi possivel corrigir apos todas as tentativas.")
    return codigo, pens


def modo_self_consistency(mensagens: list[dict]) -> tuple[str, str]:
    """Modo 4 — Gera SC_N respostas independentes e vota na mais comum."""
    print(f"\n  Gerando {SC_N} respostas e votando na melhor...\n")

    respostas = []
    for i in range(SC_N):
        sp = Spinner(f"Tentativa {i + 1}/{SC_N}")
        sp.iniciar()
        bruto = _chamar_ollama(mensagens, SYSTEM_COT, temperature=0.7)
        sp.parar()
        r = _resposta(bruto).strip()
        respostas.append(r)
        print(f"  [{i + 1}] {r[:100]}...")

    mais_comum, votos = Counter(respostas).most_common(1)[0]
    print(f"\n  Mais votada ({votos}/{SC_N}):\n")
    return mais_comum, ""


def modo_adaptativo(mensagens: list[dict], perfil: dict) -> tuple[str, str]:
    """Modo 6 — Adapta resposta ao perfil detectado do usuário."""
    linhas_perfil = "\n".join(f"- {k}: {v}" for k, v in perfil.items())
    if not linhas_perfil:
        linhas_perfil = "- Perfil ainda sendo construído. Adapte-se com base nesta mensagem."

    system = SYSTEM_ADAPTATIVO.format(perfil=linhas_perfil)

    sp = Spinner("Analisando perfil e adaptando resposta")
    sp.iniciar()
    bruto = _chamar_ollama(mensagens, system, temperature=0.1)
    sp.parar()
    return _resposta(bruto), _pensamento(bruto)


def modo_compacto(mensagens: list[dict]) -> tuple[str, str]:
    """Modo 7 — Resposta curta e direta, sem raciocínio visível."""
    bruto = _chamar_ollama(mensagens, SYSTEM_COMPACTO)
    return _resposta(bruto), ""


def modo_ultra(mensagens: list[dict]) -> tuple[str, str]:
    """Modo 8 — 30 etapas + perspectiva alternativa + síntese final."""
    sp = Spinner("Ultra-pensamento ativo (pode demorar 2-4 min)")
    sp.iniciar()
    bruto = _chamar_ollama(mensagens, SYSTEM_ULTRA, spinner=sp)
    sp.parar()
    return _resposta(bruto), _pensamento(bruto)

# ─────────────────────────────────────────────────────────────
# RAG LOCAL (modo 5)
# ─────────────────────────────────────────────────────────────

class RAGLocal:
    def __init__(self, pasta: Path = PASTA_DOCS):
        self.pasta      = pasta
        self.disponivel = False
        self._init()

    def _init(self):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self.client  = chromadb.Client()
            self.colecao = self.client.get_or_create_collection("docs_locais")
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self._indexar()
            self.disponivel = True
            print(f"  RAG inicializado: {self.colecao.count()} chunks indexados.")
        except ImportError:
            print("  RAG desativado. Instale: pip install chromadb sentence-transformers")
        except Exception as e:
            print(f"  RAG desativado: {e}")

    def _indexar(self):
        if not self.pasta.exists():
            self.pasta.mkdir(exist_ok=True)
            return
        arquivos = list(self.pasta.glob("**/*.txt")) + list(self.pasta.glob("**/*.py"))
        for arq in arquivos:
            texto  = arq.read_text(encoding="utf-8", errors="ignore")
            chunks = [texto[i:i + 500] for i in range(0, len(texto), 400)]
            for j, chunk in enumerate(chunks):
                self.colecao.add(
                    documents=[chunk],
                    ids=[f"{arq.stem}_{j}"],
                    metadatas=[{"fonte": str(arq)}],
                )

    def _buscar(self, pergunta: str, n: int = 3) -> str:
        if not self.disponivel or self.colecao.count() == 0:
            return ""
        docs = self.colecao.query(query_texts=[pergunta], n_results=n).get("documents", [[]])[0]
        return "\n\n---\n\n".join(docs)

    def perguntar(self, mensagens: list[dict]) -> tuple[str, str]:
        pergunta  = mensagens[-1]["content"]
        contexto  = self._buscar(pergunta)

        if contexto:
            print(f"\n  RAG: {len(contexto)} chars de contexto injetados.\n")
            msgs = mensagens[:-1] + [{
                "role":    "user",
                "content": f"CONTEXTO:\n{contexto}\n\nPERGUNTA:\n{pergunta}",
            }]
        else:
            msgs = mensagens

        sp = Spinner("Consultando documentos")
        sp.iniciar()
        bruto = _chamar_ollama(msgs, SYSTEM_COT)
        sp.parar()
        return _resposta(bruto), _pensamento(bruto)

# ─────────────────────────────────────────────────────────────
# MEMORIA DE CONVERSA
# ─────────────────────────────────────────────────────────────

class Memoria:
    """
    Mantém histórico como lista de {"role", "content"} para /api/chat.
    Compatível com formato antigo ({"papel", "conteudo"}).
    """

    def __init__(self, arquivo: str = "memoria_chat.json"):
        self.arquivo   = Path(arquivo)
        self.msgs: list[dict] = self._carregar()
        self.perfil: dict     = {}

    def _carregar(self) -> list:
        if not self.arquivo.exists():
            return []
        try:
            dados = json.loads(self.arquivo.read_text(encoding="utf-8"))
            if dados and "papel" in dados[0]:
                # migra formato antigo
                return [
                    {"role": "user" if m["papel"] == "user" else "assistant",
                     "content": m["conteudo"]}
                    for m in dados
                ]
            return dados
        except Exception:
            return []

    def salvar(self):
        self.arquivo.write_text(
            json.dumps(self.msgs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, role: str, content: str):
        self.msgs.append({"role": role, "content": content})
        self.salvar()

    def historico(self, max_turnos: int = 10) -> list[dict]:
        """Retorna últimos N turnos para passar ao /api/chat."""
        return self.msgs[-(max_turnos * 2):]

    def atualizar_perfil(self, chave: str, valor: str):
        self.perfil[chave] = valor

    def limpar(self):
        self.msgs  = []
        self.perfil = {}
        self.salvar()
        print("  Memoria limpa.")

# ─────────────────────────────────────────────────────────────
# DETECÇÃO DE PREFERÊNCIAS (modo adaptativo)
# ─────────────────────────────────────────────────────────────

def _atualizar_perfil(pergunta: str, memoria: Memoria):
    p = pergunta.lower()

    if any(w in p for w in ["código", "python", "função", "script", "bug", "erro", "api"]):
        memoria.atualizar_perfil("area", "desenvolvimento de software")

    if len(pergunta.split()) <= 4:
        memoria.atualizar_perfil("estilo", "mensagens curtas e diretas")
    elif len(pergunta.split()) >= 20:
        memoria.atualizar_perfil("estilo", "mensagens detalhadas e elaboradas")

    if any(p.startswith(w) for w in ["nao use", "pare de", "sem ", "nao quero"]):
        prefs = memoria.perfil.get("preferencias", "")
        memoria.atualizar_perfil("preferencias", f"{prefs}; {pergunta}".strip("; "))

# ─────────────────────────────────────────────────────────────
# INTEGRAÇÃO COM BANCO DE MEMÓRIA
# ─────────────────────────────────────────────────────────────

SYSTEM_EXTRACAO = """Você extrai fatos sobre o usuário de uma conversa.
Responda APENAS com JSON válido. Sem explicação. Sem markdown.
Formato: {"fatos": ["fato 1", "fato 2"]}
Se não há fatos relevantes: {"fatos": []}"""


def _injetar_memorias(mensagens: list[dict], banco: BancoMemoria) -> list[dict]:
    """
    Recupera memórias relevantes do banco (bi-encoder + rerank) e
    as injeta como mensagem de sistema antes da pergunta do usuário.
    O histórico de roles user/assistant não é alterado.
    """
    if not banco.disponivel:
        return mensagens

    query    = mensagens[-1]["content"]
    contexto = banco.formatar_contexto(query)

    if not contexto:
        return mensagens

    # Injeta como mensagem de sistema logo antes da pergunta atual
    injetada = {"role": "system", "content": contexto}
    return mensagens[:-1] + [injetada, mensagens[-1]]


def _extrair_e_salvar(pergunta: str, resposta: str, banco: BancoMemoria):
    """
    Chama o modelo para extrair fatos relevantes da troca e salva no banco.
    Só executa se a heurística detectar potencial de fato salvável.
    """
    if not banco.disponivel:
        return
    if not BancoMemoria.deve_salvar(pergunta):
        return

    prompt_ext = BancoMemoria.prompt_extracao(pergunta, resposta)
    msgs_ext   = [{"role": "user", "content": prompt_ext}]

    try:
        bruto  = _chamar_ollama(msgs_ext, SYSTEM_EXTRACAO, temperature=0.0)
        fatos  = BancoMemoria.parse_fatos(bruto)
        salvos = []
        for fato in fatos:
            doc_id = banco.adicionar(fato, fonte="auto")
            if doc_id:
                salvos.append(fato)
        if salvos:
            print(f"\n  [Banco] {len(salvos)} fato(s) salvo(s) automaticamente:")
            for f in salvos:
                print(f"    > {f}")
    except Exception:
        pass   # falha silenciosa — não interrompe a conversa


# ─────────────────────────────────────────────────────────────
# INTERFACE DE CHAT
# ─────────────────────────────────────────────────────────────

MODOS = {
    "1": ("CoT profundo",      "30 etapas de raciocínio"),
    "2": ("Codigo few-shot",   "Python com exemplos"),
    "3": ("Codigo + verif.",   "gera, executa, corrige"),
    "4": ("Self-consistency",  "vota entre 3 respostas"),
    "5": ("RAG local",         "usa pasta ./docs/"),
    "6": ("Adaptativo",        "aprende seu perfil"),
    "7": ("Compacto",          "respostas curtas"),
    "8": ("Ultra-pensamento",  "30 etapas + síntese"),
}

# ── Constantes de layout ──────────────────────────────────────
_CL = 29   # largura do conteúdo coluna esquerda
_CR = 26   # largura do conteúdo coluna direita
_TW = _CL + _CR + 5   # largura total interna

def _borda_topo() -> str:
    return f"+{'='*(_CL+2)}+{'='*(_CR+2)}+"

def _borda_meio() -> str:
    return f"+{'-'*(_CL+2)}+{'-'*(_CR+2)}+"

def _borda_secao() -> str:
    return f"+{'='*(_CL+2)}+{'='*(_CR+2)}+"

def _borda_fim() -> str:
    return f"+{'='*(_CL+2)}+{'='*(_CR+2)}+"

def _linha_colunas(esq: str = "", dir_: str = "") -> str:
    return f"| {esq:<{_CL}} | {dir_:<{_CR}} |"

def _linha_titulo(titulo: str) -> str:
    return f"|{titulo.center(_TW)}|"

def _borda_topo_simples() -> str:
    return f"+{'='*_TW}+"

def _borda_fim_simples() -> str:
    return f"+{'='*_TW}+"


def _exibir_header(modo: str):
    nome_modo = MODOS[modo][0]
    print(_borda_topo())
    print(_linha_titulo(f"  MOTOR IA AVANCADO  |  {MODELO}  "))
    print(_borda_meio())
    print(_linha_colunas("  MODOS DE RESPOSTA", "  COMANDOS DE CONVERSA"))
    print(_borda_meio())
    cmds_conv = [
        ("modo",    "trocar modo de resposta"),
        ("limpar",  "apagar historico da sessao"),
        ("memoria", "ver historico da sessao"),
        ("perfil",  "ver perfil detectado"),
        ("sair",    "encerrar o programa"),
    ]
    modos_list = list(MODOS.items())
    n = max(len(modos_list), len(cmds_conv))
    for i in range(n):
        esq = ""
        if i < len(modos_list):
            k, (nome, _) = modos_list[i]
            ativo = " *" if k == modo else "  "
            esq = f"{ativo}[{k}] {nome}"
        dir_ = ""
        if i < len(cmds_conv):
            cmd, desc = cmds_conv[i]
            dir_ = f"  {cmd:<9}-> {desc}"
        print(_linha_colunas(esq, dir_))
    print(_borda_meio())
    print(_linha_colunas(f"  Modo ativo: [{modo}] {nome_modo}", "  BANCO DE MEMORIA (linguagem natural)"))
    print(_borda_meio())
    banco_cmds = [
        ("lembra que <texto>",   "salva no banco"),
        ("o que voce sabe?",     "consulta o banco"),
        ("esquece tudo",         "limpa o banco"),
        ("salvar <texto>",       "salva direto"),
        ("memorias",             "lista registros"),
        ("esquecer <id>",        "remove registro"),
        ("banco",                "status do banco"),
    ]
    for cmd, desc in banco_cmds:
        print(_linha_colunas(f"  {cmd}", f"  -> {desc}"))
    print(_borda_fim())
    print()


def _menu_modo(modo_atual: str) -> str:
    print()
    print(_borda_topo_simples())
    print(_linha_titulo("  ESCOLHA O MODO DE RESPOSTA  "))
    print(_borda_meio() if False else f"╠{'═'*(_CL+_CR+5)}╣")
    for k, (nome, desc) in MODOS.items():
        ativo = " *" if k == modo_atual else "  "
        linha = f"{ativo}[{k}]  {nome:<18}  {desc}"
        total = _CL + _CR + 5
        print(f"║{linha:<{total}}║")
    print(_borda_fim_simples())
    escolha = input("  Modo [Enter = manter atual]: ").strip() or modo_atual
    return escolha if escolha in MODOS else modo_atual


# ── Comandos em linguagem natural para o banco ────────────────

_SALVAR_RE = re.compile(
    r"^(?:lembra(?:r)?|lembre(?:-se)?|guarda(?:r)?|anota(?:r)?|registra(?:r)?)"
    r"(?:\s+(?:que|isso|disso))?\s*:?\s*(.+)",
    re.IGNORECASE,
)
_LISTAR_RE = re.compile(
    r"o que (?:voc[eê]|vc) (?:sabe|lembra|tem(?: no banco)?|conhece)(?:\s+sobre\s+(.+))?",
    re.IGNORECASE,
)
_LIMPAR_RE = re.compile(
    r"^(?:esquece?|limpa?|apaga?)(?:\s+(?:tudo|banco|memorias?|memórias?))(?:\s+(?:do banco|tudo))?$",
    re.IGNORECASE,
)


def _comando_banco_natural(entrada: str, banco: BancoMemoria) -> bool:
    """
    Detecta intenções de controle do banco em linguagem natural.
    Retorna True se tratou o comando (não enviar ao modelo).
    Retorna False se é mensagem normal.
    """
    txt = entrada.strip()

    # Intenção de salvar: "lembra que X", "guarda que X", etc.
    m = _SALVAR_RE.match(txt)
    if m:
        fato = m.group(1).strip().rstrip(".")
        if fato:
            doc_id = banco.adicionar(fato, fonte="natural")
            print(f"\n  Salvo no banco: '{fato}'")
            print(f"  ID: {doc_id}\n")
        return True

    # Intenção de listar / consultar: "o que você sabe sobre X?"
    m = _LISTAR_RE.search(txt)
    if m:
        sobre = (m.group(1) or "").strip().rstrip("?")
        if sobre:
            resultados = banco.buscar(sobre)
            print(f"\n  O que sei sobre '{sobre}':")
        else:
            resultados = banco.listar(limite=15)
            resultados = [{"texto": r["texto"], "meta": r["meta"]} for r in resultados]
            print(f"\n  Tudo que sei ({banco.total()} registros):")
        if resultados:
            for r in resultados:
                ts = r.get("meta", {}).get("ts", "")
                print(f"    - {r['texto']}  ({ts})")
        else:
            print("    (banco vazio)")
        print()
        return True

    # Intenção de limpar banco
    if _LIMPAR_RE.match(txt):
        if banco.disponivel and banco._colecao:
            ids = banco._colecao.get()["ids"]
            if ids:
                banco._colecao.delete(ids=ids)
            print(f"\n  Banco limpo. {len(ids)} registro(s) removido(s).\n")
        return True

    return False


def chat():
    memoria = Memoria()
    banco   = BancoMemoria()
    rag     = None
    modo    = "1"

    _exibir_header(modo)

    while True:
        try:
            entrada = input("Voce: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

        if not entrada:
            continue

        cmd = entrada.lower().strip()

        # ── Comandos exatos ─────────────────────────────────
        if cmd == "sair":
            break

        if cmd == "modo":
            modo = _menu_modo(modo)
            _exibir_header(modo)
            continue

        if cmd == "limpar":
            memoria.limpar()
            continue

        if cmd == "memoria":
            msgs = memoria.historico()
            print(f"\n  Historico desta sessao ({len(msgs)} msgs):")
            for m in msgs:
                print(f"  [{m['role']:9}] {m['content'][:100]}")
            print()
            continue

        if cmd == "perfil":
            print("\n  Perfil detectado:")
            if memoria.perfil:
                for k, v in memoria.perfil.items():
                    print(f"  {k}: {v}")
            else:
                print("  (ainda vazio)")
            print()
            continue

        if cmd == "banco":
            print(f"\n  Banco de memoria:")
            print(f"  Registros : {banco.total()}")
            print(f"  Disponivel: {banco.disponivel}")
            print(f"  Modelos   : {'carregados' if banco.modelos_prontos else 'lazy (carregam no 1o uso)'}")
            print()
            continue

        if cmd == "memorias":
            registros = banco.listar()
            print(f"\n  Registros no banco ({len(registros)}):")
            for r in registros:
                ts = r["meta"].get("ts", "")
                print(f"  [{r['id'][-8:]}]  {r['texto'][:75]}  ({ts})")
            if not registros:
                print("  (banco vazio)")
            print()
            continue

        if cmd.startswith("salvar "):
            texto = entrada[7:].strip()
            if texto:
                doc_id = banco.adicionar(texto, fonte="manual")
                print(f"  Salvo — ID: {doc_id[-8:]}\n")
            else:
                print("  Uso: salvar <texto>\n")
            continue

        if cmd.startswith("esquecer "):
            doc_id = entrada[9:].strip()
            banco.remover(doc_id)
            print(f"  Removido: {doc_id}\n")
            continue

        # ── Linguagem natural -> banco (antes de ir ao modelo) ──
        if _comando_banco_natural(entrada, banco):
            continue

        # ── Monta contexto para o modelo ────────────────────
        # Histórico da sessão + nova mensagem
        mensagens_base = memoria.historico() + [{"role": "user", "content": entrada}]

        # Injeta memórias relevantes do banco (bi-encoder + rerank)
        mensagens = _injetar_memorias(mensagens_base, banco)

        print()
        inicio     = time.time()
        pensamento = ""

        match modo:
            case "1":
                resposta, pensamento = modo_cot(mensagens)
            case "2":
                resposta, pensamento = modo_codigo(mensagens)
            case "3":
                resposta, pensamento = modo_verificador(mensagens)
            case "4":
                resposta, pensamento = modo_self_consistency(mensagens)
            case "5":
                if rag is None:
                    print("  Inicializando RAG (primeira vez pode demorar)...")
                    rag = RAGLocal()
                resposta, pensamento = rag.perguntar(mensagens)
            case "6":
                resposta, pensamento = modo_adaptativo(mensagens, memoria.perfil)
            case "7":
                resposta, pensamento = modo_compacto(mensagens)
            case "8":
                resposta, pensamento = modo_ultra(mensagens)
            case _:
                resposta, pensamento = modo_cot(mensagens)

        duracao   = time.time() - inicio
        nome_modo = MODOS[modo][0]

        _exibir(pensamento, resposta, nome_modo, duracao, mostrar_pens=(modo != "7"))

        # Extrai fatos da troca e salva no banco automaticamente
        _extrair_e_salvar(entrada, resposta, banco)

        # Atualiza perfil para modo adaptativo
        _atualizar_perfil(entrada, memoria)

        # Persiste histórico da sessão
        memoria.add("user",      entrada)
        memoria.add("assistant", resposta)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        r = requests.get(OLLAMA_TAGS, timeout=3)
        modelos = [m["name"] for m in r.json().get("models", [])]
        base    = MODELO.split(":")[0]
        if not any(base in m for m in modelos):
            print(f"  Modelo '{MODELO}' nao encontrado.")
            print(f"  Execute: ollama pull {MODELO}")
            if modelos:
                print(f"  Disponiveis: {modelos}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("  [ERRO] Ollama nao esta rodando.")
        print("  Execute em outro terminal: ollama serve")
        sys.exit(1)

    chat()
