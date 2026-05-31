"""
sandbox.py
Execução segura de código Python gerado pelo modelo.

Proteções implementadas:
  1. Análise estática: bloqueia imports de rede e sistema perigosos
  2. Subprocess isolado: timeout, captura de I/O, sem herança de proxies
  3. Limite de output capturado (evita DoS por output excessivo)

Nota: proteção é heurística, não garantia de sandbox de SO.
Adequada para código gerado por modelos locais em ambiente de desenvolvimento.
"""

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_PADRAO = 15     # segundos antes de cancelar execução
LIMITE_OUTPUT  = 4096   # chars máximos de stdout/stderr capturados

# Módulos que implicam acesso de rede ou operações potencialmente destrutivas
_IMPORTS_BLOQUEADOS = frozenset({
    "socket", "urllib", "urllib3", "httpx", "http",
    "ftplib", "smtplib", "imaplib", "poplib", "xmlrpc",
    "paramiko", "fabric", "aiohttp", "httplib2",
    "subprocess",     # evita escape de sandbox via subprocesso
    "multiprocessing",
    "ctypes", "cffi", "mmap",
})


def validar(codigo: str) -> tuple[bool, str]:
    """
    Análise estática antes de executar.

    Returns:
        (valido, motivo_rejeicao) — motivo é "" se valido=True
    """
    try:
        arvore = ast.parse(codigo)
    except SyntaxError as e:
        return False, f"Sintaxe inválida: {e}"

    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            for alias in no.names:
                base = alias.name.split(".")[0]
                if base in _IMPORTS_BLOQUEADOS:
                    return False, f"Import bloqueado por segurança: {alias.name}"
        elif isinstance(no, ast.ImportFrom):
            if no.module:
                base = no.module.split(".")[0]
                if base in _IMPORTS_BLOQUEADOS:
                    return False, f"Import bloqueado por segurança: {no.module}"

    return True, ""


def executar(codigo: str, timeout: int = TIMEOUT_PADRAO) -> tuple[bool, str]:
    """
    Executa código Python em subprocess isolado com timeout.

    Args:
        codigo:  código Python a executar
        timeout: segundos máximos de execução

    Returns:
        (sucesso, output)
        sucesso=True  → output é stdout (truncado em LIMITE_OUTPUT)
        sucesso=False → output é stderr ou mensagem de erro
    """
    valido, motivo = validar(codigo)
    if not valido:
        return False, f"[BLOQUEADO] {motivo}"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(codigo)
        caminho = f.name

    env = dict(os.environ)
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(var, None)

    try:
        proc = subprocess.run(
            [sys.executable, caminho],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if proc.returncode == 0:
            return True, (proc.stdout or "(sem output)")[:LIMITE_OUTPUT]
        erro = proc.stderr or f"Processo encerrou com código {proc.returncode}"
        return False, erro[:LIMITE_OUTPUT]

    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: código excedeu {timeout}s sem concluir"
    except Exception as e:
        return False, f"Erro ao executar: {e}"
    finally:
        Path(caminho).unlink(missing_ok=True)
