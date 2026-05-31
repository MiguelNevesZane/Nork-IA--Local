"""
roteador_modelos.py
Catálogo de modelos e roteador de tarefas para o Nork-IA.

Todos os modelos padrão usam Q4_K_M e cabem em 6-8 GB de VRAM.
Modelos opcionais (14B+) requerem offload parcial CPU/RAM.

Fontes consultadas: model cards em hub.ollama.com (qwen2.5-coder, deepseek-r1)
DeepSeek R1 temperatura recomendada: 0.5-0.7 (documentação DeepSeek, jan/2025)
"""

import re

# ─────────────────────────────────────────────────────────────
# CATÁLOGO DE MODELOS
# ─────────────────────────────────────────────────────────────
# vram_gb: estimativa para Q4_K_M incluindo overhead KV cache (~1 GB)
# Confirme disponibilidade com: ollama list

CATALOGO: dict[str, dict] = {
    "raciocinio": {
        "nome":     "deepseek-r1:8b",
        "vram_gb":  5.5,
        "descricao": "Raciocínio profundo (<think> nativo, CoT 30 etapas)",
        "temp":     0.6,   # DeepSeek R1 docs: range recomendado 0.5-0.7
        "opcional": False,
    },
    "codigo": {
        "nome":     "qwen2.5-coder:7b",
        "vram_gb":  5.0,
        "descricao": "Código Python especializado (principal)",
        "temp":     0.0,
        "opcional": False,
    },
    "codigo_rapido": {
        "nome":     "qwen2.5-coder:3b",
        "vram_gb":  2.3,
        "descricao": "Código rápido / baixa latência",
        "temp":     0.0,
        "opcional": False,
    },
    "geral": {
        "nome":     "qwen2.5:7b",
        "vram_gb":  5.0,
        "descricao": "Conversação e perguntas gerais",
        "temp":     0.3,
        "opcional": False,
    },
    # ── Opcional: offload CPU/RAM, mais lento ────────────────
    "codigo_grande": {
        "nome":     "qwen2.5-coder:14b",
        "vram_gb":  9.0,
        "descricao": "OPCIONAL — código avançado (requer >8 GB ou offload CPU)",
        "temp":     0.0,
        "opcional": True,
    },
}

MODELO_FALLBACK = "deepseek-r1:8b"

# ─────────────────────────────────────────────────────────────
# MAPEAMENTO: MODO -> CATEGORIA PREFERIDA
# ─────────────────────────────────────────────────────────────

_MODO_CATEGORIA: dict[str, str] = {
    "1": "raciocinio",    # CoT profundo
    "2": "codigo",        # Código few-shot
    "3": "codigo",        # Código + verificação
    "4": "raciocinio",    # Self-consistency
    "5": "geral",         # RAG local
    "6": "geral",         # Adaptativo
    "7": "codigo_rapido", # Compacto
    "8": "raciocinio",    # Ultra-pensamento
}

# ─────────────────────────────────────────────────────────────
# PADRÕES DE CONTEÚDO (refinamento intra-modo)
# ─────────────────────────────────────────────────────────────

_RE_CODIGO = re.compile(
    r"\b(python|def |class |import |código|codigo|script|função|funcao|"
    r"bug|erro|fix|refatora|algoritmo|api|json|csv|sql|\.py|"
    r"fazer funcionar|não funciona|corrige?|corrija|implementa?)\b",
    re.IGNORECASE,
)

_RE_RACIOCINIO = re.compile(
    r"\b(analisa?|analise|explica?|explique|por que|porque|como funciona|"
    r"diferença|compare|avalia?|avalie|estratégia|decisão|argumento|"
    r"vantagens?|desvantagens?|pros e contras)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────

def rotear(
    texto: str,
    modo: str,
    modelos_disponiveis: list[str],
) -> tuple[str, str]:
    """
    Seleciona o modelo mais adequado para (texto, modo).

    Returns:
        (nome_modelo, categoria_selecionada)
    """
    categoria = _MODO_CATEGORIA.get(modo, "geral")

    # Para modos adaptativos/gerais, refina pela análise do conteúdo
    if modo in ("5", "6"):
        if _RE_CODIGO.search(texto):
            categoria = "codigo"
        elif _RE_RACIOCINIO.search(texto):
            categoria = "raciocinio"

    return _escolher(categoria, modelos_disponiveis)


def temperatura_para_modo(modo: str) -> float:
    """Temperatura recomendada para o modo de resposta."""
    cat = _MODO_CATEGORIA.get(modo, "geral")
    return CATALOGO.get(cat, CATALOGO["geral"])["temp"]


def listar_catalogo() -> list[dict]:
    return [
        {"categoria": cat, **cfg}
        for cat, cfg in CATALOGO.items()
    ]


# ─────────────────────────────────────────────────────────────
# INTERNO
# ─────────────────────────────────────────────────────────────

def _escolher(categoria: str, disponiveis: list[str]) -> tuple[str, str]:
    cfg = CATALOGO.get(categoria, CATALOGO["geral"])
    if _disponivel(cfg["nome"], disponiveis):
        return cfg["nome"], categoria

    for cat_fb in _fallbacks(categoria):
        nome_fb = CATALOGO.get(cat_fb, {}).get("nome", "")
        if nome_fb and _disponivel(nome_fb, disponiveis):
            return nome_fb, cat_fb

    if disponiveis:
        return disponiveis[0], categoria
    return MODELO_FALLBACK, categoria


def _disponivel(nome: str, disponiveis: list[str]) -> bool:
    base = nome.split(":")[0]
    return any(base in m for m in disponiveis)


def _fallbacks(categoria: str) -> list[str]:
    MAPA = {
        "raciocinio":    ["geral", "codigo", "codigo_rapido"],
        "codigo":        ["codigo_rapido", "raciocinio", "geral"],
        "codigo_rapido": ["codigo", "geral", "raciocinio"],
        "geral":         ["raciocinio", "codigo", "codigo_rapido"],
    }
    return MAPA.get(categoria, ["raciocinio"])
