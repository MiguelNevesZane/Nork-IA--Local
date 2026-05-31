"""
motor_ia_avancado.py
====================
Motor de inferência avançado para modelos locais via Ollama.
Combina: CoT forçado, few-shot, self-consistency, verificador automático e RAG básico.

Uso:
    python motor_ia_avancado.py

Dependências:
    pip install requests
    pip install chromadb sentence-transformers   (opcional — só para RAG)

Modelos recomendados (RTX 3050 6GB):
    ollama pull deepseek-r1:8b      # MELHOR para 6GB — raciocínio nativo (~4.9 GB VRAM) ← PADRÃO
    ollama pull qwen2.5:7b          # alternativa geral, mais leve (~4.1 GB VRAM)
    ollama pull qwen2.5-coder:7b    # alternativa especialista em código (~4.1 GB VRAM)
"""

import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import requests

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
MODELO        = "deepseek-r1:8b"   # melhor que cabe em 6GB VRAM (~4.9GB); raciocínio nativo
TIMEOUT       = 240                # mais tempo para pensamento profundo

SC_TENTATIVAS = 3
MAX_CORRECOES = 2
PASTA_DOCS    = Path("./docs")

# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

SYSTEM_GERAL = """Você é um assistente especialista que raciocina em português brasileiro com profundidade e rigor intelectual máximos.

PROCESSO OBRIGATÓRIO — execute TODAS as 30 etapas dentro de <thinking></thinking> antes de responder.
Cada etapa deve ter ao menos 2-3 linhas de raciocínio real. Não pule nenhuma.

ETAPA 1  — LEITURA LITERAL
Qual é o enunciado exato da pergunta? Repita-o com suas próprias palavras.

ETAPA 2  — INTENÇÃO REAL
O que o usuário realmente quer saber? Pode haver uma intenção implícita além das palavras?

ETAPA 3  — CONTEXTO
Em que contexto essa pergunta faz sentido? Acadêmico, prático, filosófico, técnico?

ETAPA 4  — CONHECIMENTO PRÉVIO NECESSÁRIO
Que conceitos fundamentais preciso dominar para responder bem?

ETAPA 5  — DECOMPOSIÇÃO EM SUB-PERGUNTAS
Quebre a pergunta em todas as partes menores que precisam ser respondidas individualmente.

ETAPA 6  — ORDEM LÓGICA
Em que ordem devo responder as sub-perguntas para o raciocínio ser coerente?

ETAPA 7  — PRIMEIRA HIPÓTESE
Qual é minha resposta instintiva inicial? Por que penso isso?

ETAPA 8  — BASE DE EVIDÊNCIAS
Que fatos, dados, princípios ou leis sustentam minha hipótese inicial?

ETAPA 9  — CAUSAS E ORIGENS
Quais são as causas raiz do fenômeno em questão? Vá além do superficial.

ETAPA 10 — MECANISMOS E PROCESSOS
Como exatamente o fenômeno funciona, passo a passo, internamente?

ETAPA 11 — CONSEQUÊNCIAS E EFEITOS
Quais são os efeitos diretos? E os efeitos de segunda e terceira ordem?

ETAPA 12 — EXEMPLOS CONCRETOS
Cite pelo menos 2 exemplos reais que ilustram os pontos centrais.

ETAPA 13 — CASOS EXTREMOS E LIMITES
O que acontece nos casos limítrofes? Onde o raciocínio começa a falhar?

ETAPA 14 — EXCEÇÕES E ANOMALIAS
Existem exceções conhecidas à regra geral? O que elas revelam?

ETAPA 15 — PERSPECTIVA HISTÓRICA
Como esse tema evoluiu ao longo do tempo? Houve mudanças de paradigma?

ETAPA 16 — PERSPECTIVA CIENTÍFICA
O que a ciência ou a academia estabeleceu sobre isso? Há consenso ou debate?

ETAPA 17 — PERSPECTIVA PRÁTICA
Como isso se aplica na vida real, no dia a dia, em situações concretas?

ETAPA 18 — PERSPECTIVA CRÍTICA
Quem discordaria dessa visão? Qual seria o argumento mais forte do lado oposto?

ETAPA 19 — REFUTAÇÃO DO CONTRA-ARGUMENTO
Como respondo ao argumento oposto? Por que minha posição ainda se sustenta?

ETAPA 20 — SUPOSIÇÕES IMPLÍCITAS
Que pressupostos estou assumindo sem questionar? Eles são válidos?

ETAPA 21 — VIESES POTENCIAIS
Existe algum viés cognitivo, cultural ou ideológico que pode estar distorcendo meu raciocínio?

ETAPA 22 — O QUE ESTOU IGNORANDO
Que ângulo ou dimensão do problema ainda não considerei? Força-se a pensar no ponto cego.

ETAPA 23 — ANALOGIAS E COMPARAÇÕES
Existe algo em outro domínio que funciona de forma semelhante e pode iluminar esse tema?

ETAPA 24 — IMPLICAÇÕES FILOSÓFICAS
Que questões mais profundas esse tema levanta sobre conhecimento, existência ou valores?

ETAPA 25 — SÍNTESE PARCIAL
Com tudo que analisei até aqui, qual é a conclusão intermediária mais sólida?

ETAPA 26 — TESTE DE CONSISTÊNCIA
Minha conclusão parcial é consistente com todas as etapas anteriores? Há contradições?

ETAPA 27 — REVISÃO CRÍTICA DA PRÓPRIA ANÁLISE
Olhando para meu raciocínio como um todo: há saltos lógicos, generalizações indevidas ou lacunas?

ETAPA 28 — REFINAMENTO FINAL
Depois da revisão, como aprimoro ou ajusto minha conclusão?

ETAPA 29 — CLAREZA E UTILIDADE
Como estruturo a resposta para ser maximamente clara e útil para quem perguntou?

ETAPA 30 — CONCLUSÃO DEFINITIVA
Qual é minha resposta final, completa e bem fundamentada?

EXCEÇÃO: Para cumprimentos simples (oi, tudo bem, bom dia), responda diretamente sem as etapas.

Formato obrigatório:
<thinking>
ETAPA 1 — LEITURA LITERAL
[raciocínio]

ETAPA 2 — INTENÇÃO REAL
[raciocínio]

[... todas as 30 etapas ...]

ETAPA 30 — CONCLUSÃO DEFINITIVA
[raciocínio]
</thinking>
<answer>
[Resposta clara, estruturada e completa em português brasileiro]
</answer>"""

SYSTEM_CODIGO = """Você é um especialista em Python que responde em português brasileiro.

REGRAS:
- Raciocine antes de escrever código: use <thinking>...</thinking>
- O código final vai entre <answer> e </answer>
- Código deve ser funcional, com tratamento de erros
- Prefira soluções simples e legíveis
- Adicione comentários nas partes não óbvias"""

# ─────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# ─────────────────────────────────────────────

FEW_SHOT_CODIGO = """
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
Pergunta: Como fazer retry automático em uma requisição HTTP com backoff exponencial?
Resposta:
import time
import requests

def get_com_retry(url: str, max_tentativas: int = 3) -> requests.Response:
    for tentativa in range(max_tentativas):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if tentativa == max_tentativas - 1:
                raise
            espera = 2 ** tentativa
            print(f"Tentativa {tentativa+1} falhou. Aguardando {espera}s...")
            time.sleep(espera)

---
"""

# ─────────────────────────────────────────────
# CLIENTE OLLAMA
# ─────────────────────────────────────────────

def gerar(
    prompt: str,
    system: str = SYSTEM_GERAL,
    temperature: float = 0.0,
    stream: bool = True,
) -> str:
    payload = {
        "model":  MODELO,
        "prompt": prompt,
        "system": system,
        "options": {
            "temperature": temperature,
            "top_p": 0.95 if temperature > 0 else 1.0,
            "num_predict": 8192,   # espaço para 30 etapas de raciocínio
        },
        "stream": stream,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT, stream=stream)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("\n[ERRO] Ollama não está rodando. Execute em outro terminal: ollama serve")
        sys.exit(1)

    texto = ""
    if stream:
        for linha in resp.iter_lines():
            if linha:
                dados = json.loads(linha)
                token = dados.get("response", "")
                texto += token
                print(token, end="", flush=True)
                if dados.get("done"):
                    break
        print()
    else:
        texto = resp.json().get("response", "")

    return texto

# ─────────────────────────────────────────────
# EXTRAÇÃO DE TAGS
# ─────────────────────────────────────────────

def extrair_tag(texto: str, tag: str) -> str:
    padrao = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(padrao, texto, re.DOTALL)
    return match.group(1).strip() if match else ""

def extrair_pensamento(texto: str) -> str:
    """Extrai raciocínio de <think> (deepseek-r1 nativo) ou <thinking> (nosso prompt)."""
    for tag in ("think", "thinking"):
        resultado = extrair_tag(texto, tag)
        if resultado:
            return resultado
    return ""

def extrair_resposta(texto: str) -> str:
    """Extrai resposta de <answer>, ou texto após </think>/</thinking> se não houver tag."""
    answer = extrair_tag(texto, "answer")
    if answer:
        return answer
    # fallback: texto após o fechamento da tag de pensamento
    for fechamento in ("</think>", "</thinking>"):
        if fechamento in texto:
            parte = texto.split(fechamento, 1)[1].strip()
            if parte:
                return parte
    return texto.strip()

def extrair_codigo(texto: str) -> str:
    resposta = extrair_tag(texto, "answer") or texto
    limpo = re.sub(r"```(?:python)?\n?(.*?)```", r"\1", resposta, flags=re.DOTALL)
    return limpo.strip()

SEP = "─" * 56

# ─────────────────────────────────────────────
# TÉCNICA 1: CoT PROFUNDO
# ─────────────────────────────────────────────

def perguntar_cot(pergunta: str, verbose: bool = True) -> str:
    if verbose:
        print(f"\n{SEP}")
        print("  Raciocínio profundo em andamento...")
        print(f"{SEP}\n")

    resposta_bruta = gerar(pergunta, system=SYSTEM_GERAL, temperature=0.0)

    pensamento = extrair_pensamento(resposta_bruta)
    answer     = extrair_resposta(resposta_bruta)

    if verbose and pensamento:
        print(f"\n{SEP}")
        print("  RACIOCÍNIO INTERNO")
        print(SEP)
        # exibe o pensamento formatado, indentado
        for linha in pensamento.splitlines():
            print(f"  {linha}")
        print(f"\n{SEP}")
        print("  RESPOSTA FINAL")
        print(f"{SEP}\n")
        print(answer)
        return answer

    return answer if answer else resposta_bruta

# ─────────────────────────────────────────────
# TÉCNICA 2: FEW-SHOT + CoT PARA CÓDIGO
# ─────────────────────────────────────────────

def gerar_codigo(pergunta: str, verbose: bool = True) -> str:
    if verbose:
        print("\n[Gerando codigo com exemplos de referencia...]\n")

    prompt = f"{FEW_SHOT_CODIGO}\n--- Pergunta atual ---\n{pergunta}\nResposta:"
    resposta = gerar(prompt, system=SYSTEM_CODIGO, temperature=0.0)
    return extrair_codigo(resposta)

# ─────────────────────────────────────────────
# TÉCNICA 3: SELF-CONSISTENCY (VOTAÇÃO)
# ─────────────────────────────────────────────

def perguntar_com_voto(pergunta: str, n: int = SC_TENTATIVAS) -> str:
    print(f"\n[Gerando {n} respostas e votando na melhor...]\n")

    respostas = []
    for i in range(n):
        print(f"  Tentativa {i+1}/{n}...")
        r = gerar(pergunta, system=SYSTEM_GERAL, temperature=0.7, stream=False)
        answer = extrair_tag(r, "answer")
        respostas.append(answer.strip())
        print(f"  -> {answer.strip()[:80]}...")

    mais_comum, votos = Counter(respostas).most_common(1)[0]
    print(f"\nResposta com mais votos ({votos}/{n}):\n{mais_comum}\n")
    return mais_comum

# ─────────────────────────────────────────────
# TÉCNICA 4: VERIFICADOR AUTOMÁTICO DE CÓDIGO
# ─────────────────────────────────────────────

def executar_codigo(codigo: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(codigo)
        caminho = f.name

    try:
        resultado = subprocess.run(
            [sys.executable, caminho],
            capture_output=True, text=True, timeout=15
        )
        if resultado.returncode == 0:
            return True, resultado.stdout
        else:
            return False, resultado.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: codigo demorou mais de 15 segundos"
    finally:
        Path(caminho).unlink(missing_ok=True)

def gerar_e_verificar(pergunta: str) -> str:
    print("\n[Gerando + testando codigo automaticamente...]\n")

    codigo = gerar_codigo(pergunta, verbose=False)

    for tentativa in range(MAX_CORRECOES + 1):
        print(f"--- Codigo (tentativa {tentativa+1}) ---\n{codigo}\n")

        sucesso, output = executar_codigo(codigo)

        if sucesso:
            print(f"Codigo executou sem erros! Output: {output or '(sem output)'}")
            return codigo

        print(f"Erro:\n{output}\n")

        if tentativa < MAX_CORRECOES:
            print(f"Pedindo autocorrecao ({tentativa+2}/{MAX_CORRECOES+1})...\n")
            prompt_correcao = f"""O seguinte código Python tem um erro. Corrija-o.

Pergunta original: {pergunta}

Código com erro:
```python
{codigo}
```

Erro:
{output}

Código corrigido entre <answer> e </answer>."""
            resposta = gerar(prompt_correcao, system=SYSTEM_CODIGO, temperature=0.0)
            codigo = extrair_codigo(resposta)

    print("Nao foi possivel corrigir apos todas as tentativas.")
    return codigo

# ─────────────────────────────────────────────
# TÉCNICA 5: RAG LOCAL (OPCIONAL)
# ─────────────────────────────────────────────

class RAGLocal:
    def __init__(self, pasta: Path = PASTA_DOCS):
        self.pasta = pasta
        self.disponivel = False
        self._inicializar()

    def _inicializar(self):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self.client  = chromadb.Client()
            self.colecao = self.client.get_or_create_collection("docs_locais")
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self._indexar_docs()
            self.disponivel = True
            print(f"RAG inicializado com {self.colecao.count()} chunks.")
        except ImportError:
            print("RAG desativado. Instale: pip install chromadb sentence-transformers")
        except Exception as e:
            print(f"RAG desativado: {e}")

    def _indexar_docs(self):
        if not self.pasta.exists():
            self.pasta.mkdir(exist_ok=True)
            return
        arquivos = list(self.pasta.glob("**/*.txt")) + list(self.pasta.glob("**/*.py"))
        for arquivo in arquivos:
            texto = arquivo.read_text(encoding="utf-8", errors="ignore")
            chunks = [texto[i:i+500] for i in range(0, len(texto), 400)]
            for j, chunk in enumerate(chunks):
                self.colecao.add(
                    documents=[chunk],
                    ids=[f"{arquivo.stem}_{j}"],
                    metadatas=[{"fonte": str(arquivo)}]
                )

    def buscar(self, pergunta: str, n: int = 3) -> str:
        if not self.disponivel or self.colecao.count() == 0:
            return ""
        resultados = self.colecao.query(query_texts=[pergunta], n_results=n)
        docs = resultados.get("documents", [[]])[0]
        return "\n\n---\n\n".join(docs)

    def perguntar(self, pergunta: str) -> str:
        contexto = self.buscar(pergunta)
        if contexto:
            print(f"\n[RAG] {len(contexto)} chars de contexto injetados.\n")
            prompt = f"""Use APENAS o contexto abaixo para responder. Se não tiver a informação, diga que não encontrou.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}"""
        else:
            prompt = pergunta
        return perguntar_cot(prompt)

# ─────────────────────────────────────────────
# MEMÓRIA DE CONVERSA
# ─────────────────────────────────────────────

class MemoriaConversa:
    def __init__(self, arquivo: str = "memoria_chat.json"):
        self.arquivo = Path(arquivo)
        self.historico: list[dict] = self._carregar()

    def _carregar(self) -> list:
        if self.arquivo.exists():
            with open(self.arquivo, encoding="utf-8") as f:
                return json.load(f)
        return []

    def salvar(self):
        with open(self.arquivo, "w", encoding="utf-8") as f:
            json.dump(self.historico, f, ensure_ascii=False, indent=2)

    def adicionar(self, papel: str, conteudo: str):
        self.historico.append({
            "papel": papel,
            "conteudo": conteudo,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        self.salvar()

    def formatar_para_prompt(self, max_turnos: int = 6) -> str:
        recentes = self.historico[-(max_turnos * 2):]
        if not recentes:
            return ""
        linhas = []
        for msg in recentes:
            papel = "Usuario" if msg["papel"] == "user" else "Assistente"
            linhas.append(f"{papel}: {msg['conteudo']}")
        return "\n".join(linhas)

    def limpar(self):
        self.historico = []
        self.salvar()
        print("Memoria limpa.")

# ─────────────────────────────────────────────
# INTERFACE DE CHAT
# ─────────────────────────────────────────────

MODOS = {
    "1": ("Conversa geral (CoT)",    "perguntas gerais, chat, explicacoes"),
    "2": ("Codigo + few-shot",       "geracao de codigo Python"),
    "3": ("Codigo + verificador",    "codigo testado automaticamente"),
    "4": ("Self-consistency",        "respostas objetivas com votacao"),
    "5": ("RAG local",               "usa arquivos da pasta ./docs/"),
}

def menu_modo() -> str:
    print("\n+-- Modo de resposta ---------------------------")
    for k, (nome, desc) in MODOS.items():
        print(f"|  [{k}] {nome:<28} {desc}")
    print("+-----------------------------------------------")
    escolha = input("Modo [1]: ").strip() or "1"
    return escolha if escolha in MODOS else "1"

def chat():
    print("=" * 56)
    print(f"  Motor IA Avancado  |  Modelo: {MODELO}")
    print("=" * 56)
    print("Comandos: 'modo' muda tecnica | 'limpar' reseta memoria")
    print("          'memoria' ve historico | 'sair' encerra\n")

    memoria = MemoriaConversa()
    rag     = None   # inicializado só quando modo 5 for selecionado
    modo    = "1"

    while True:
        try:
            pergunta = input("Voce: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            break
        if pergunta.lower() == "modo":
            modo = menu_modo()
            print(f"Modo: {MODOS[modo][0]}")
            continue
        if pergunta.lower() == "limpar":
            memoria.limpar()
            continue
        if pergunta.lower() == "memoria":
            print("\n-- Historico --")
            print(memoria.formatar_para_prompt(max_turnos=10) or "(vazio)")
            print("---------------")
            continue

        contexto_memoria = memoria.formatar_para_prompt()
        if contexto_memoria:
            prompt_final = f"Historico recente:\n{contexto_memoria}\n\nNova mensagem: {pergunta}"
        else:
            prompt_final = pergunta

        print()
        inicio = time.time()

        if modo == "1":
            resposta = perguntar_cot(prompt_final)
        elif modo == "2":
            resposta = gerar_codigo(prompt_final)
        elif modo == "3":
            resposta = gerar_e_verificar(prompt_final)
        elif modo == "4":
            resposta = perguntar_com_voto(prompt_final)
        elif modo == "5":
            if rag is None:
                print("Inicializando RAG (pode demorar na primeira vez)...")
                rag = RAGLocal()
            resposta = rag.perguntar(prompt_final)

        duracao = time.time() - inicio
        print(f"\n[{duracao:.1f}s | {MODOS[modo][0]}]")

        memoria.adicionar("user", pergunta)
        memoria.adicionar("assistant", resposta)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        modelos_disponiveis = [m["name"] for m in r.json().get("models", [])]
        modelo_base = MODELO.split(":")[0]
        if not any(modelo_base in m for m in modelos_disponiveis):
            print(f"Modelo '{MODELO}' nao encontrado.")
            print(f"Execute: ollama pull {MODELO}")
            if modelos_disponiveis:
                print(f"Modelos disponiveis: {modelos_disponiveis}")
            else:
                print("Nenhum modelo instalado ainda.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("[ERRO] Ollama nao esta rodando.")
        print("   Execute em outro terminal: ollama serve")
        sys.exit(1)

    chat()
