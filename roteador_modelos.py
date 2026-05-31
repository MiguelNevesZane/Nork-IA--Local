"""
roteador_modelos.py
Catálogo de modelos + roteador inteligente para o Nork-IA.

Fluxo:
    rotear_inteligente(mensagem, historico, ...)
        ↓ chama modelo de decisão (menor disponível, ~2-3s)
        ↓ recebe JSON: {"modelo": "...", "motivo": "..."}
        ↓ fallback para regex se o modelo falhar
        → retorna (nome_modelo, motivo_str)

O modelo de decisão lê a mensagem + histórico recente e decide
qual modelo da lista de disponíveis é o mais adequado para a tarefa.
"""

import json
import re

import requests

# ─────────────────────────────────────────────────────────────
# CATÁLOGO DE MODELOS
# ─────────────────────────────────────────────────────────────

CATALOGO: dict[str, dict] = {
    "raciocinio": {
        "nome":     "deepseek-r1:8b",
        "vram_gb":  5.5,
        "descricao": "raciocínio profundo, análise complexa, CoT 30 etapas",
        "temp":     0.6,
        "opcional": False,
    },
    "codigo": {
        "nome":     "qwen2.5-coder:7b",
        "vram_gb":  5.0,
        "descricao": "código Python, debugging, scripts, APIs, refatoração",
        "temp":     0.0,
        "opcional": False,
    },
    "codigo_rapido": {
        "nome":     "qwen2.5-coder:3b",
        "vram_gb":  2.3,
        "descricao": "respostas rápidas, perguntas simples, conversação",
        "temp":     0.0,
        "opcional": False,
    },
    "geral": {
        "nome":     "qwen2.5:7b",
        "vram_gb":  5.0,
        "descricao": "conversação, perguntas gerais, RAG, explicações",
        "temp":     0.3,
        "opcional": False,
    },
    "codigo_grande": {
        "nome":     "qwen2.5-coder:14b",
        "vram_gb":  9.0,
        "descricao": "OPCIONAL — código avançado (requer offload CPU)",
        "temp":     0.0,
        "opcional": True,
    },
}

MODELO_FALLBACK   = "deepseek-r1:8b"
OLLAMA_CHAT_URL   = "http://localhost:11434/api/chat"

# Ordem de preferência para o modelo de decisão (mais rápido primeiro)
_DECISAO_PREFERENCIA = [
    "qwen2.5-coder:3b",
    "qwen2.5:7b",
    "qwen2.5-coder:7b",
    "deepseek-r1:8b",
    "mistral:latest",
]

# Mapeamento modo → categoria (fallback regex)
_MODO_CATEGORIA: dict[str, str] = {
    "1": "raciocinio",
    "2": "codigo",
    "3": "codigo",
    "4": "raciocinio",
    "5": "geral",
    "6": "geral",
    "7": "codigo_rapido",
    "8": "raciocinio",
}

_NOMES_MODOS = {
    "1": "CoT profundo",
    "2": "Codigo few-shot",
    "3": "Codigo + verificacao",
    "4": "Self-consistency",
    "5": "RAG local",
    "6": "Adaptativo",
    "7": "Compacto",
    "8": "Ultra-pensamento",
}

# ─────────────────────────────────────────────────────────────
# ROTEADOR INTELIGENTE (usa modelo de decisão)
# ─────────────────────────────────────────────────────────────

def rotear_inteligente(
    mensagem: str,
    modo: str,
    historico_recente: list[dict],
    modelos_disponiveis: list[str],
    contexto_banco: str = "",
) -> tuple[str, str]:
    """
    Usa um modelo pequeno para decidir qual modelo usar na tarefa.

    Args:
        mensagem:            nova mensagem do usuário
        modo:                modo de resposta ativo ('1'-'8')
        historico_recente:   últimas N mensagens do histórico
        modelos_disponiveis: modelos instalados no Ollama
        contexto_banco:      memórias relevantes já recuperadas

    Returns:
        (nome_modelo, motivo_curto)
        motivo_curto é uma string de até 5 palavras para exibir no terminal.
    """
    modelo_decisao = _modelo_decisao(modelos_disponiveis)
    disponiveis_nao_opcionais = _modelos_nao_opcionais(modelos_disponiveis)

    if not disponiveis_nao_opcionais:
        return MODELO_FALLBACK, "fallback (sem modelos)"

    prompt = _montar_prompt_decisao(
        mensagem, modo, historico_recente, disponiveis_nao_opcionais, contexto_banco
    )

    try:
        resposta_raw = _chamar_decisao(prompt, modelo_decisao)
        modelo, motivo = _parsear_decisao(resposta_raw, disponiveis_nao_opcionais)
        if modelo:
            return modelo, motivo
    except Exception:
        pass

    # Fallback: regex
    modelo, _ = _rotear_regex(mensagem, modo, modelos_disponiveis)
    return modelo, "roteamento por padrao"


def temperatura_para_modelo(nome_modelo: str) -> float:
    """Temperatura recomendada para o modelo."""
    for cfg in CATALOGO.values():
        if cfg["nome"] == nome_modelo:
            return cfg["temp"]
    return 0.3


def listar_catalogo() -> list[dict]:
    return [{"categoria": cat, **cfg} for cat, cfg in CATALOGO.items()]


# ─────────────────────────────────────────────────────────────
# INTERNOS
# ─────────────────────────────────────────────────────────────

def _montar_prompt_decisao(
    mensagem: str,
    modo: str,
    historico: list[dict],
    disponiveis: list[dict],
    contexto_banco: str,
) -> str:
    lista_modelos = "\n".join(
        f'- "{m["nome"]}": {m["descricao"]}'
        for m in disponiveis
    )

    historico_str = _formatar_historico(historico[-6:])  # últimas 3 trocas
    nome_modo     = _NOMES_MODOS.get(modo, f"modo {modo}")

    banco_str = ""
    if contexto_banco:
        banco_str = f"\nMemórias relevantes do banco:\n{contexto_banco[:300]}"

    return f"""Você é um roteador de modelos de IA. Analise a situação e escolha o modelo mais adequado.

Modelos disponíveis:
{lista_modelos}

Modo de resposta ativo: {nome_modo}
{banco_str}

Histórico recente:
{historico_str}

Nova mensagem do usuário: {mensagem}

Regras de decisão:
- Código, script, debug, função, bug → modelo especializado em código
- Análise profunda, filosofia, comparação, "por que", "como funciona" → modelo de raciocínio
- Conversa simples, "oi", "obrigado", pergunta curta → modelo rápido
- Perguntas gerais, explicações, RAG → modelo geral
- Se o histórico mostra código em andamento → continue com o mesmo modelo de código
- Escolha o modelo que melhor combina com o contexto COMPLETO, não só a última mensagem

Responda APENAS com JSON válido, sem explicação, sem markdown:
{{"modelo": "nome_exato_do_modelo_da_lista", "motivo": "razao em ate 5 palavras"}}"""


def _chamar_decisao(prompt: str, modelo: str) -> str:
    """Chamada leve ao modelo de decisão (sem stream, num_predict baixo)."""
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model":   modelo,
            "messages": [
                {
                    "role":    "system",
                    "content": "Roteador de modelos. Responda APENAS com JSON valido. Sem texto adicional.",
                },
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.0,
                "num_predict": 80,
                "num_ctx":     2048,
            },
            "stream": False,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def _parsear_decisao(resposta: str, disponiveis: list[dict]) -> tuple[str, str]:
    """Extrai modelo e motivo do JSON retornado pelo modelo de decisão."""
    nomes_validos = {m["nome"] for m in disponiveis}

    # Tenta extrair JSON da resposta (pode vir com markdown)
    limpo = re.sub(r"```(?:json)?\n?(.*?)```", r"\1", resposta, flags=re.DOTALL).strip()
    m = re.search(r'\{[^{}]+\}', limpo, re.DOTALL)
    if not m:
        return "", ""

    try:
        dados  = json.loads(m.group())
        modelo = dados.get("modelo", "").strip()
        motivo = dados.get("motivo", "").strip()

        # Verifica se o modelo escolhido está realmente disponível
        if modelo in nomes_validos:
            return modelo, motivo

        # Tenta match parcial (ex: "qwen2.5-coder" → "qwen2.5-coder:7b")
        for nome in nomes_validos:
            if modelo.split(":")[0] in nome or nome.split(":")[0] in modelo:
                return nome, motivo

    except (json.JSONDecodeError, KeyError):
        pass

    return "", ""


def _modelo_decisao(disponiveis: list[str]) -> str:
    """Seleciona o modelo mais rápido disponível para fazer a decisão."""
    for preferido in _DECISAO_PREFERENCIA:
        if _disponivel(preferido, disponiveis):
            return preferido
    return disponiveis[0] if disponiveis else MODELO_FALLBACK


def _modelos_nao_opcionais(disponiveis: list[str]) -> list[dict]:
    """Retorna metadados dos modelos não-opcionais que estão instalados."""
    return [
        cfg
        for cfg in CATALOGO.values()
        if not cfg["opcional"] and _disponivel(cfg["nome"], disponiveis)
    ]


def _disponivel(nome: str, disponiveis: list[str]) -> bool:
    base = nome.split(":")[0]
    return any(base in m for m in disponiveis)


def _formatar_historico(msgs: list[dict]) -> str:
    if not msgs:
        return "(sem histórico)"
    linhas = []
    for m in msgs:
        role    = "Usuario" if m["role"] == "user" else "Assistente"
        conteudo = m["content"][:120].replace("\n", " ")
        linhas.append(f"  {role}: {conteudo}")
    return "\n".join(linhas)


_RE_CODIGO_FALLBACK = re.compile(
    r"\b(python|def |class |import |codigo|codigo|script|funcao|"
    r"bug|erro|fix|refatora|algoritmo|api|json|csv|sql|escreve?|implementa?)\b",
    re.IGNORECASE,
)


def _rotear_regex(texto: str, modo: str, disponiveis: list[str]) -> tuple[str, str]:
    """Roteamento simples por regex (fallback quando LLM falha)."""
    cat = _MODO_CATEGORIA.get(modo, "geral")

    # Modos adaptativos/gerais: refina pelo conteúdo
    if modo in ("5", "6") and _RE_CODIGO_FALLBACK.search(texto):
        cat = "codigo"

    cfg = CATALOGO.get(cat, CATALOGO["geral"])
    if _disponivel(cfg["nome"], disponiveis):
        return cfg["nome"], cat
    for cat_fb in ["raciocinio", "codigo", "geral", "codigo_rapido"]:
        nome_fb = CATALOGO[cat_fb]["nome"]
        if _disponivel(nome_fb, disponiveis):
            return nome_fb, cat_fb
    return MODELO_FALLBACK, "raciocinio"


# Alias mantido para compatibilidade com testes existentes
def rotear(texto: str, modo: str, modelos_disponiveis: list[str]) -> tuple[str, str]:
    return _rotear_regex(texto, modo, modelos_disponiveis)


def temperatura_para_modo(modo: str) -> float:
    cat = _MODO_CATEGORIA.get(modo, "geral")
    return CATALOGO.get(cat, CATALOGO["geral"])["temp"]
