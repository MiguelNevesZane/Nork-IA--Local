@echo off
cd /d "%~dp0"

rem ── Otimizações do Ollama ─────────────────────────────────
rem Mantém o modelo na VRAM entre chamadas (sem recarregar a cada sessão)
set OLLAMA_KEEP_ALIVE=10m

rem Ativa Flash Attention se suportado pela sua GPU (RTX série 20+)
rem Remove o rem abaixo para ativar:
rem set OLLAMA_FLASH_ATTENTION=1

rem KV Cache em quantização q8_0 — economiza ~20%% de VRAM
rem Verifique suporte: ollama --version >= 0.3.0
rem set OLLAMA_KV_CACHE_TYPE=q8_0

rem ── Encoding e ambiente virtual ─────────────────────────
set PYTHONIOENCODING=utf-8
call venv_cuda\Scripts\activate.bat
python main.py
pause
