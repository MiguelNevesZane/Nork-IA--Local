"""
grafo.py
Orquestração LangGraph do Nork-IA — grafo de estados com transições explícitas.

Fluxo:
    roteador -> contexto -> inferencia -> [execucao -> reflexao]* -> saida

Requer: pip install langgraph langchain-core
Fallback automático quando não instalado (retorna disponivel=False).

Uso no motor:
    from grafo import construir_grafo
    app, ok = construir_grafo(banco, modelos_disponiveis)
    if ok:
        resultado = app.invoke(estado_inicial)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

logger = logging.getLogger("nork.grafo")

try:
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    _LANGGRAPH_OK = True
except ImportError:
    _LANGGRAPH_OK = False
    TypedDict = dict  # type: ignore


# ─────────────────────────────────────────────────────────────
# ESTADO DO GRAFO
# ─────────────────────────────────────────────────────────────

if _LANGGRAPH_OK:
    class EstadoNork(TypedDict, total=False):
        # Entrada
        entrada:        str
        modo:           str
        mensagens:      list[dict]
        perfil:         dict

        # Roteador
        modelo:         str
        categoria:      str

        # Contexto
        ctx_banco:      str

        # Inferência
        pensamento:     str
        resposta:       str
        codigo_gerado:  str

        # Execução de código
        exec_ok:        bool
        exec_output:    str
        tentativas:     int

        # Metadados
        trace:          list[str]
        duracao_s:      float
else:
    EstadoNork = dict  # type: ignore


# ─────────────────────────────────────────────────────────────
# CONSTRUTOR DO GRAFO
# ─────────────────────────────────────────────────────────────

def construir_grafo(
    banco,                          # BancoMemoria — injetado via closure
    modelos_disponiveis: list[str],
    max_fix: int = 2,
) -> tuple[Any, bool]:
    """
    Monta e compila o grafo LangGraph.

    Args:
        banco:               instância de BancoMemoria
        modelos_disponiveis: lista de modelos instalados no Ollama
        max_fix:             tentativas máximas de correção de código

    Returns:
        (app, disponivel)
        app.invoke(estado) executa o fluxo completo.
        disponivel=False quando LangGraph não está instalado.
    """
    if not _LANGGRAPH_OK:
        logger.warning(
            "LangGraph nao instalado. "
            "Execute: pip install langgraph langchain-core"
        )
        return None, False

    # Imports do motor — feitos aqui para evitar importação circular no topo
    from motor_ia_avancado import (
        SYSTEM_COT, SYSTEM_CODIGO, SYSTEM_COMPACTO, SYSTEM_ULTRA,
        SYSTEM_ADAPTATIVO, FEW_SHOT_CODIGO, MAX_FIX as _MAX_FIX,
        _chamar_ollama, _resposta, _pensamento, _codigo,
    )
    from roteador_modelos import rotear
    from sandbox import executar as sandbox_exec

    _max = max(max_fix, _MAX_FIX)

    # ── Nós ─────────────────────────────────────────────────

    def no_roteador(estado: EstadoNork) -> dict:
        modelo, cat = rotear(
            estado.get("entrada", ""),
            estado.get("modo", "1"),
            modelos_disponiveis,
        )
        _log(estado, f"roteador | modo={estado.get('modo')} cat={cat} modelo={modelo}")
        return {"modelo": modelo, "categoria": cat}

    def no_contexto(estado: EstadoNork) -> dict:
        ctx = ""
        if banco.disponivel:
            ctx = banco.formatar_contexto(estado.get("entrada", ""))
        _log(estado, f"contexto | banco={len(ctx)} chars")
        return {"ctx_banco": ctx}

    def no_inferencia(estado: EstadoNork) -> dict:
        modo     = estado.get("modo", "1")
        modelo   = estado.get("modelo", "deepseek-r1:8b")
        msgs     = list(state.get("mensagens", []) for state in [estado])[0]

        # Injeta contexto do banco se disponível
        ctx = estado.get("ctx_banco", "")
        if ctx and msgs:
            msgs = msgs[:-1] + [
                {"role": "system", "content": ctx},
                msgs[-1],
            ]

        # Seleciona prompt de sistema pelo modo
        sistema, temperatura, eh_codigo = _sistema_para_modo(
            modo,
            estado.get("perfil", {}),
            SYSTEM_COT, SYSTEM_CODIGO, SYSTEM_COMPACTO, SYSTEM_ULTRA, SYSTEM_ADAPTATIVO,
        )

        if eh_codigo:
            ultima = msgs[-1]["content"] if msgs else ""
            msgs = msgs[:-1] + [{"role": "user", "content": f"{FEW_SHOT_CODIGO}\n---\n{ultima}"}]

        bruto = _chamar_ollama(msgs, sistema, temperature=temperatura, modelo=modelo)
        pens  = _pensamento(bruto)

        if modo in ("2", "3"):
            cod = _codigo(bruto)
            _log(estado, f"inferencia | codigo={len(cod)} chars")
            return {"codigo_gerado": cod, "pensamento": pens, "resposta": cod}

        resp = _resposta(bruto)
        _log(estado, f"inferencia | resposta={len(resp)} chars")
        return {"resposta": resp, "pensamento": pens}

    def no_execucao(estado: EstadoNork) -> dict:
        cod = estado.get("codigo_gerado", "")
        if not cod:
            return {"exec_ok": False, "exec_output": "Nenhum código gerado"}
        ok, out = sandbox_exec(cod)
        _log(estado, f"execucao | ok={ok} output={out[:60]!r}")
        return {"exec_ok": ok, "exec_output": out}

    def no_reflexao(estado: EstadoNork) -> dict:
        tentativas = estado.get("tentativas", 0) + 1
        _log(estado, f"reflexao | tentativa={tentativas}/{_max}")

        msgs_fix = list(estado.get("mensagens", [])) + [
            {"role": "assistant", "content": estado.get("codigo_gerado", "")},
            {"role": "user",      "content": (
                f"Erro ao executar:\n{estado.get('exec_output', '')}\n\n"
                "Corrija o código. Coloque o código corrigido entre <answer></answer>."
            )},
        ]
        modelo = estado.get("modelo", "deepseek-r1:8b")
        bruto  = _chamar_ollama(msgs_fix, SYSTEM_CODIGO, temperature=0.0, modelo=modelo)
        novo   = _codigo(bruto)
        return {"codigo_gerado": novo, "resposta": novo, "tentativas": tentativas}

    def no_saida(estado: EstadoNork) -> dict:
        _log(estado, "saida")
        return {}

    # ── Condicionais ─────────────────────────────────────────

    def _para_execucao(estado: EstadoNork) -> Literal["execucao", "saida"]:
        return "execucao" if estado.get("modo") == "3" else "saida"

    def _apos_execucao(estado: EstadoNork) -> Literal["reflexao", "saida"]:
        if estado.get("exec_ok"):
            return "saida"
        if estado.get("tentativas", 0) >= _max:
            return "saida"
        return "reflexao"

    def _apos_reflexao(estado: EstadoNork) -> Literal["execucao", "saida"]:
        if estado.get("tentativas", 0) >= _max:
            return "saida"
        return "execucao"

    # ── Montagem ─────────────────────────────────────────────

    g = StateGraph(EstadoNork)
    g.add_node("roteador",   no_roteador)
    g.add_node("contexto",   no_contexto)
    g.add_node("inferencia", no_inferencia)
    g.add_node("execucao",   no_execucao)
    g.add_node("reflexao",   no_reflexao)
    g.add_node("saida",      no_saida)

    g.set_entry_point("roteador")
    g.add_edge("roteador",   "contexto")
    g.add_edge("contexto",   "inferencia")
    g.add_conditional_edges("inferencia", _para_execucao)
    g.add_conditional_edges("execucao",   _apos_execucao)
    g.add_conditional_edges("reflexao",   _apos_reflexao)
    g.add_edge("saida", END)

    app = g.compile()
    logger.info("Grafo LangGraph compilado com sucesso.")
    return app, True


# ─────────────────────────────────────────────────────────────
# AUXILIARES
# ─────────────────────────────────────────────────────────────

def _log(estado: EstadoNork, msg: str):
    trace = list(estado.get("trace", []))
    trace.append(msg)
    estado["trace"] = trace
    logger.debug(msg)


def _sistema_para_modo(modo, perfil, COT, CODIGO, COMPACTO, ULTRA, ADAPTATIVO):
    """Retorna (system_prompt, temperatura, eh_codigo)."""
    if modo == "1":
        return COT, 0.6, False
    if modo == "2":
        return CODIGO, 0.0, True
    if modo == "3":
        return CODIGO, 0.0, True
    if modo == "7":
        return COMPACTO, 0.0, False
    if modo == "8":
        return ULTRA, 0.6, False
    if modo == "6":
        linhas = "\n".join(f"- {k}: {v}" for k, v in perfil.items()) or "- Perfil sendo construído."
        return ADAPTATIVO.format(perfil=linhas), 0.1, False
    return COT, 0.6, False


def disponivel() -> bool:
    """Retorna True se LangGraph está instalado."""
    return _LANGGRAPH_OK
