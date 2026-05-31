# Nork-IA — Motor de Inteligência Artificial Local

> Assistente conversacional avançado com roteador de modelos, 8 modos de resposta, memória semântica, sandbox de código e orquestração LangGraph. Inferência 100% local via Ollama.

---

## Visão Geral

O **Nork-IA** combina múltiplos modelos Ollama (Q4_K_M, 6-8 GB VRAM) com um roteador automático de tarefas, memória semântica bi-encoder + cross-encoder, execução segura de código Python e orquestração via grafo de estados LangGraph.

```
Entrada do Usuário
       |
  [Roteador] -> seleciona modelo ideal por tarefa
       |
  [Contexto] -> histórico + memória semântica (bi-encoder + rerank)
       |
  [Inferência] -> Ollama /api/chat com modelo selecionado
       |
  [Sandbox]? -> executa Python, captura output (Modo 3)
       |
  [Reflexão]? -> autocorrige até MAX_FIX tentativas
       |
  [Resposta + Auto-save de Fatos]
```

---

## Arquitetura

### Camada 1 — Runtime (Ollama)
- Mantém modelo na VRAM com `KEEP_ALIVE=10m`
- `num_ctx=8192` equilibra contexto e VRAM
- Flash Attention e KV cache quantizado: configuráveis em `iniciar.bat`

### Camada 2 — Roteador de Modelos (`roteador_modelos.py`)

| Categoria | Modelo (Q4_K_M) | VRAM est. | Quando usar |
|---|---|---|---|
| raciocinio | deepseek-r1:8b | ~5.5 GB | CoT 30 etapas, modos 1/4/8 |
| codigo | qwen2.5-coder:7b | ~5.0 GB | Código Python (modos 2/3) |
| codigo_rapido | qwen2.5-coder:3b | ~2.3 GB | Respostas rápidas (modo 7) |
| geral | qwen2.5:7b | ~5.0 GB | Conversação, RAG (modos 5/6) |
| *(opcional)* | qwen2.5-coder:14b | ~9.0 GB | Código avançado (offload CPU) |

Todos os modelos padrão cabem em 6-8 GB de VRAM. Confirme com `ollama list`.

### Camada 3 — Modos de Raciocínio
- DeepSeek R1 com temperatura 0.6 (faixa recomendada: 0.5-0.7)
- 30 etapas CoT nomeadas, self-consistency (3 votos), ultra-pensamento

### Camada 4 — Memória + RAG
- Banco: ChromaDB + all-MiniLM-L6-v2 + cross-encoder rerank (diferencial)
- RAG (Modo 5): splitting code-aware para .py (por função/classe) + overlap para .txt

### Camada 5 — Orquestração LangGraph (`grafo.py`)
- Grafo de estados: roteador → contexto → inferência → [execução → reflexão] → saída
- Ativo para Modo 3 quando `langgraph` instalado
- Traces em log `nork.grafo` (DEBUG)

### Camada 6 — Sandbox (`sandbox.py`)
- Execução segura: análise estática (AST), timeout, sem herança de proxies
- Imports bloqueados: socket, urllib, subprocess, ctypes, etc.
- Saída capturada e truncada em 4 KB

---

## Requisitos

| Componente | Versão Mínima |
|---|---|
| Python | 3.10+ |
| Ollama | qualquer |
| RAM | 8 GB |
| VRAM | 6 GB (modelos padrão em Q4_K_M) |
| Disco | ~10 GB (modelos + dependências) |

---

## Instalação

### 1. Instalar o Ollama

Baixe em [ollama.com](https://ollama.com). Depois baixe os modelos desejados:

```bash
# Modelo padrão (raciocínio, já suficiente para tudo)
ollama pull deepseek-r1:8b

# Opcional: modelo especializado em código (melhora modos 2/3)
ollama pull qwen2.5-coder:7b

# Opcional: modelo rápido (modo 7 compacto)
ollama pull qwen2.5-coder:3b

# Opcional: modelo geral (modos 5/6)
ollama pull qwen2.5:7b
```

O roteador usa automaticamente o melhor modelo disponível. Se só o `deepseek-r1:8b` estiver instalado, o sistema funciona normalmente.

### 2. Instalar Dependências Python

```bash
python -m venv venv_cuda
venv_cuda\Scripts\activate   # Windows

# Obrigatório
pip install requests chromadb sentence-transformers torch

# Opcional: grafo de estados com traces
pip install langgraph langchain-core

# Opcional: fine-tuning QLoRA
pip install unsloth
```

---

## Como Executar

**Terminal 1 — Ollama:**
```bash
ollama serve
```

**Terminal 2 — Nork-IA:**
```bash
iniciar.bat          # Windows (ativa venv + otimizações Ollama)
# ou
python main.py
```

---

## Modos de Resposta

Use `modo` no chat para alternar.

| # | Nome | Modelo preferido | Quando usar |
|---|---|---|---|
| 1 | CoT profundo | deepseek-r1:8b | Análise, perguntas complexas |
| 2 | Código few-shot | qwen2.5-coder:7b | Geração de Python |
| 3 | Código + verificação | qwen2.5-coder:7b | Código que precisa rodar |
| 4 | Self-consistency | deepseek-r1:8b | Decisões críticas (3 votos) |
| 5 | RAG local | qwen2.5:7b | Consultar ./docs/ |
| 6 | Adaptativo | qwen2.5:7b | Uso diário personalizado |
| 7 | Compacto | qwen2.5-coder:3b | Respostas rápidas |
| 8 | Ultra-pensamento | deepseek-r1:8b | Análise máxima (2-4 min) |

---

## Comandos do Chat

### Sistema

| Comando | Ação |
|---|---|
| `modo` | Trocar modo de resposta |
| `limpar` | Apagar histórico da sessão |
| `memoria` | Exibir histórico de conversa |
| `perfil` | Mostrar perfil detectado |
| `banco` | Status do banco de memória |
| `memorias` | Listar registros do banco |
| `salvar <texto>` | Salvar diretamente no banco |
| `esquecer <id>` | Remover registro por ID |
| `sair` | Encerrar |

### Memória Semântica (linguagem natural)

| Comando | Ação |
|---|---|
| `lembra que <texto>` | Salva fato no banco |
| `guarda que <texto>` | Alias para salvar |
| `o que você sabe?` | Lista todas as memórias |
| `o que você sabe sobre <tema>?` | Busca por tema (bi-encoder + rerank) |
| `esquece tudo` | Limpa o banco inteiro |

---

## Estrutura de Arquivos

```
RPA_IA/
+-- main.py                    # Ponto de entrada
+-- motor_ia_avancado.py       # Motor: 8 modos, chat, memória de sessão
+-- banco_memoria.py           # Memória semântica: ChromaDB + bi-encoder + rerank
+-- roteador_modelos.py        # Catálogo de modelos + roteador automático de tarefas
+-- sandbox.py                 # Execução segura de código Python
+-- grafo.py                   # Orquestração LangGraph (opcional)
+-- testes.py                  # Suite de 46 testes
+-- requirements.txt           # Dependências com justificativas
+-- iniciar.bat                # Inicialização Windows (venv + env vars Ollama)
+-- memoria_db/                # ChromaDB (gerado automaticamente)
+-- docs/                      # Pasta para RAG local (crie manualmente)
+-- memoria_chat.json          # Histórico de conversa (gerado automaticamente)
```

---

## Configuração

### Parâmetros do Motor (`motor_ia_avancado.py`)

```python
MODELO      = "deepseek-r1:8b"  # modelo fallback
TIMEOUT     = 300               # segundos (aumentar para GPUs lentas)
NUM_PREDICT = 8192              # tokens máximos por resposta
NUM_CTX     = 8192              # janela de contexto
KEEP_ALIVE  = "10m"             # tempo de retenção na VRAM
SC_N        = 3                 # tentativas self-consistency
MAX_FIX     = 2                 # tentativas de autocorreção de código
USO_GRAFO   = True              # habilita LangGraph quando instalado
```

### Otimizações do Ollama (`iniciar.bat`)

```bat
set OLLAMA_KEEP_ALIVE=10m         # já habilitado por padrão
rem set OLLAMA_FLASH_ATTENTION=1  # descomente para RTX série 20+
rem set OLLAMA_KV_CACHE_TYPE=q8_0 # descomente para economizar ~20% VRAM
```

---

## Testes

```bash
# Todos os testes (46 casos)
python testes.py

# Com encoding correto no Windows
$env:PYTHONIOENCODING="utf-8"; python testes.py
```

Resultado esperado: `46 OK | 0 FALHA | 0 PULADOS`

---

## Pipeline de Memória Semântica

```
Pergunta do Usuário
       |
 [Bi-Encoder: all-MiniLM-L6-v2]
   embeddings rapidos -> top-20 candidatos
       |
 [Cross-Encoder: ms-marco-MiniLM-L-6-v2]
   reranking preciso -> top-5 relevantes
       |
 Contexto injetado antes da inferencia
```

---

## Limitações Conhecidas

- **Ollama obrigatório** em terminal separado (`ollama serve`)
- **6-8 GB VRAM** limita a modelos Q4_K_M ≤ 8B por vez (troca automática pelo roteador)
- **Modos 1 e 8** podem levar 1-4 min (30 etapas de CoT)
- **Sandbox** não é garantia de segurança de SO — proteção é heurística via AST
- **Trocar embedding** invalida vetores existentes em `memoria_db/`

---

## Autor

**Miguel Neves Zane**
GitHub: [@MiguelNevesZane](https://github.com/MiguelNevesZane)
Email: ayatechbr@gmail.com
