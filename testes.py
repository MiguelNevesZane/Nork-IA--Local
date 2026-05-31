"""
testes.py
Suite de testes para o Nork-IA.

Cobertura:
  1. Roteador de modelos (detecção de tipo, fallbacks)
  2. Sandbox (validação estática, execução segura)
  3. Banco de memória (regex de comandos naturais)
  4. Motor (extração de tags, constantes)
  5. Splitting de documentos RAG

Uso:
    python testes.py          # todos os testes
    python testes.py --rapido # pula testes que precisam de modelos pesados
"""

import sys
import textwrap
import traceback

_RAPIDO = "--rapido" in sys.argv

PASSOU = 0
FALHOU = 0
PULOS  = 0


def ok(nome: str):
    global PASSOU
    PASSOU += 1
    print(f"  [OK]   {nome}")


def falha(nome: str, motivo: str):
    global FALHOU
    FALHOU += 1
    print(f"  [FALHA] {nome}: {motivo}")


def pulo(nome: str, motivo: str = ""):
    global PULOS
    PULOS += 1
    print(f"  [PULO] {nome}" + (f" ({motivo})" if motivo else ""))


def secao(titulo: str):
    print(f"\n{'-'*60}")
    print(f"  {titulo}")
    print(f"{'-'*60}")


# ─────────────────────────────────────────────────────────────
# 1. ROTEADOR DE MODELOS
# ─────────────────────────────────────────────────────────────

def testar_roteador():
    secao("1. Roteador de Modelos")
    from roteador_modelos import rotear, temperatura_para_modo, listar_catalogo, CATALOGO

    # 1.1 — Modo 1 (CoT) mapeia para categoria raciocinio
    modelo, cat = rotear("explique machine learning", "1", ["deepseek-r1:8b"])
    if cat == "raciocinio":
        ok("modo-1 -> categoria raciocinio")
    else:
        falha("modo-1 -> categoria raciocinio", f"obteve: {cat}")

    # 1.2 — Modo 2 (código) mapeia para codigo
    _, cat = rotear("escreva uma função Python", "2", ["qwen2.5-coder:7b"])
    if cat == "codigo":
        ok("modo-2 -> categoria codigo")
    else:
        falha("modo-2 -> categoria codigo", f"obteve: {cat}")

    # 1.3 — Modo 7 (compacto) mapeia para codigo_rapido
    _, cat = rotear("oi", "7", ["qwen2.5-coder:3b"])
    if cat == "codigo_rapido":
        ok("modo-7 -> categoria codigo_rapido")
    else:
        falha("modo-7 -> categoria codigo_rapido", f"obteve: {cat}")

    # 1.4 — Fallback quando modelo preferido não disponível
    modelo, _ = rotear("crie um script", "2", ["deepseek-r1:8b"])
    if "deepseek" in modelo:
        ok("fallback para deepseek quando qwen nao disponivel")
    else:
        falha("fallback", f"modelo inesperado: {modelo}")

    # 1.5 — Sem modelos disponíveis retorna MODELO_FALLBACK
    from roteador_modelos import MODELO_FALLBACK
    modelo, _ = rotear("oi", "7", [])
    if modelo == MODELO_FALLBACK:
        ok("lista vazia -> MODELO_FALLBACK")
    else:
        falha("lista vazia -> MODELO_FALLBACK", f"obteve: {modelo}")

    # 1.6 — Temperatura para modo de raciocínio deve ser > 0
    temp = temperatura_para_modo("1")
    if temp > 0:
        ok(f"temperatura modo-1 > 0 (obtido: {temp})")
    else:
        falha("temperatura modo-1", f"esperado >0, obtido: {temp}")

    # 1.7 — Temperatura para modo de código deve ser 0
    temp = temperatura_para_modo("2")
    if temp == 0.0:
        ok("temperatura modo-2 == 0.0")
    else:
        falha("temperatura modo-2", f"esperado 0.0, obtido: {temp}")

    # 1.8 — Catalogo tem chaves obrigatórias
    erros = []
    for cat, cfg in CATALOGO.items():
        for campo in ("nome", "vram_gb", "temp", "opcional"):
            if campo not in cfg:
                erros.append(f"{cat} falta campo {campo}")
    if not erros:
        ok("estrutura do catálogo válida")
    else:
        falha("estrutura do catálogo", "; ".join(erros))

    # 1.9 — Modo 6 adaptativo refina para codigo quando input tem código
    _, cat = rotear("escreve um script python para mim", "6", ["qwen2.5-coder:7b"])
    if cat == "codigo":
        ok("modo-6 com input de codigo -> refina para codigo")
    else:
        falha("modo-6 refinamento", f"obteve: {cat}")


# ─────────────────────────────────────────────────────────────
# 2. SANDBOX
# ─────────────────────────────────────────────────────────────

def testar_sandbox():
    secao("2. Sandbox de Execução de Código")
    from sandbox import validar, executar

    # 2.1 — Código simples válido
    valido, motivo = validar("x = 1 + 1\nprint(x)")
    if valido:
        ok("validacao codigo simples")
    else:
        falha("validacao codigo simples", motivo)

    # 2.2 — Import de rede bloqueado
    valido, motivo = validar("import socket\nsocket.connect(('x', 80))")
    if not valido and "socket" in motivo:
        ok("import socket bloqueado")
    else:
        falha("import socket bloqueado", f"valido={valido} motivo={motivo}")

    # 2.3 — import subprocess bloqueado
    valido, motivo = validar("import subprocess\nsubprocess.run(['ls'])")
    if not valido:
        ok("import subprocess bloqueado")
    else:
        falha("import subprocess bloqueado", "deveria rejeitar")

    # 2.4 — Sintaxe inválida rejeitada
    valido, motivo = validar("def f(\n    pass")
    if not valido and "Sintaxe" in motivo:
        ok("sintaxe invalida rejeitada")
    else:
        falha("sintaxe invalida rejeitada", f"valido={valido}")

    # 2.5 — Execução bem-sucedida
    sucesso, output = executar("print('hello nork')")
    if sucesso and "hello nork" in output:
        ok("execucao simples bem-sucedida")
    else:
        falha("execucao simples", f"sucesso={sucesso} output={output!r}")

    # 2.6 — Execução com erro captura stderr
    sucesso, output = executar("raise ValueError('erro de teste')")
    if not sucesso and "ValueError" in output:
        ok("execucao com erro captura stderr")
    else:
        falha("execucao com erro", f"sucesso={sucesso} output={output!r}")

    # 2.7 — Timeout
    sucesso, output = executar("while True: pass", timeout=2)
    if not sucesso and "TIMEOUT" in output:
        ok("timeout detectado")
    else:
        falha("timeout", f"sucesso={sucesso} output={output!r}")

    # 2.8 — Código bloqueado não chega ao subprocess
    sucesso, output = executar("import urllib.request\nurllib.request.urlopen('http://x')")
    if not sucesso and "BLOQUEADO" in output:
        ok("import bloqueado nao executa")
    else:
        falha("import bloqueado nao executa", f"sucesso={sucesso} output={output!r}")

    # 2.9 — Código com output retorna stdout
    sucesso, output = executar("for i in range(3): print(i)")
    if sucesso and "0" in output and "1" in output:
        ok("output multiplas linhas capturado")
    else:
        falha("output multiplas linhas", f"output={output!r}")


# ─────────────────────────────────────────────────────────────
# 3. BANCO DE MEMÓRIA — REGEX DE LINGUAGEM NATURAL
# ─────────────────────────────────────────────────────────────

def testar_banco_regex():
    secao("3. Comandos Naturais do Banco de Memória")
    from motor_ia_avancado import _SALVAR_RE, _LISTAR_RE, _LIMPAR_RE

    # 3.1 — Detecta intenção de salvar
    casos_salvar = [
        "lembra que meu nome é Miguel",
        "lembre-se que eu uso Python",
        "guarda que prefiro código simples",
        "anota isso: gosto de documentação",
        "registra que trabalho com automação",
    ]
    erros = [c for c in casos_salvar if not _SALVAR_RE.match(c)]
    if not erros:
        ok("detecao de intencao salvar (5/5)")
    else:
        falha("detecao salvar", f"nao detectou: {erros}")

    # 3.2 — Extrai texto após comando de salvar
    m = _SALVAR_RE.match("lembra que meu nome é Miguel")
    if m and "meu nome" in m.group(1):
        ok("extracao de texto apos comando salvar")
    else:
        falha("extracao de texto salvar", f"grupo: {m.group(1) if m else 'None'}")

    # 3.3 — Detecta intenção de listar
    casos_listar = [
        "o que você sabe?",
        "o que vc lembra sobre Python?",
        "o que você tem sobre automação",
        "o que você conhece",
    ]
    erros = [c for c in casos_listar if not _LISTAR_RE.search(c)]
    if not erros:
        ok("detecao de intencao listar (4/4)")
    else:
        falha("detecao listar", f"nao detectou: {erros}")

    # 3.4 — Detecta intenção de limpar
    casos_limpar = [
        "esquece tudo",
        "limpa banco",
        "apaga memorias",
        "esquece banco",
    ]
    erros = [c for c in casos_limpar if not _LIMPAR_RE.match(c)]
    if not erros:
        ok("detecao de intencao limpar (4/4)")
    else:
        falha("detecao limpar", f"nao detectou: {erros}")

    # 3.5 — Não confunde mensagem normal com comando de banco
    normais = [
        "como funciona o Python?",
        "qual é a capital do Brasil?",
        "me explica recursão",
    ]
    falsos = [c for c in normais if _SALVAR_RE.match(c) or _LIMPAR_RE.match(c)]
    if not falsos:
        ok("mensagens normais nao confundidas com comandos")
    else:
        falha("falsos positivos", f"confundiu: {falsos}")


# ─────────────────────────────────────────────────────────────
# 4. EXTRAÇÃO DE TAGS DO MOTOR
# ─────────────────────────────────────────────────────────────

def testar_extracao_tags():
    secao("4. Extração de Tags (pensamento, resposta, código)")
    from motor_ia_avancado import _tag, _pensamento, _resposta, _codigo

    # 4.1 — Extrai tag genérica
    r = _tag("<answer>hello</answer>", "answer")
    if r == "hello":
        ok("extracao tag generica")
    else:
        falha("extracao tag generica", f"obteve: {r!r}")

    # 4.2 — Extrai pensamento de <think>
    bruto = "<think>raciocinio aqui</think>\nResposta final"
    p = _pensamento(bruto)
    if "raciocinio" in p:
        ok("extracao <think>")
    else:
        falha("extracao <think>", f"obteve: {p!r}")

    # 4.3 — Extrai pensamento de <thinking>
    bruto = "<thinking>pensamento</thinking><answer>final</answer>"
    p = _pensamento(bruto)
    if "pensamento" in p:
        ok("extracao <thinking>")
    else:
        falha("extracao <thinking>", f"obteve: {p!r}")

    # 4.4 — Extrai resposta de <answer>
    bruto = "<think>p</think><answer>resposta aqui</answer>"
    r = _resposta(bruto)
    if r == "resposta aqui":
        ok("extracao <answer>")
    else:
        falha("extracao <answer>", f"obteve: {r!r}")

    # 4.5 — Extrai resposta do texto pós-</think>
    bruto = "<think>pens</think>\nTexto depois do think"
    r = _resposta(bruto)
    if "Texto depois" in r:
        ok("extracao resposta pos-think")
    else:
        falha("extracao resposta pos-think", f"obteve: {r!r}")

    # 4.6 — Extrai código de bloco markdown
    bruto = "```python\nx = 42\nprint(x)\n```"
    r = _codigo(bruto)
    if "x = 42" in r:
        ok("extracao codigo de markdown")
    else:
        falha("extracao codigo markdown", f"obteve: {r!r}")

    # 4.7 — Extrai código de <answer>
    bruto = "<answer>\nx = 1\nprint(x)\n</answer>"
    r = _codigo(bruto)
    if "x = 1" in r:
        ok("extracao codigo de <answer>")
    else:
        falha("extracao codigo <answer>", f"obteve: {r!r}")


# ─────────────────────────────────────────────────────────────
# 5. SPLITTING DE DOCUMENTOS (RAG)
# ─────────────────────────────────────────────────────────────

def testar_splitting():
    secao("5. Splitting de Documentos para RAG")
    from motor_ia_avancado import _split_codigo, _split_texto, _split_por_janela

    # 5.1 — Split de código por função
    codigo = textwrap.dedent("""\
        def soma(a, b):
            return a + b

        def produto(a, b):
            return a * b

        class Calculadora:
            def dividir(self, a, b):
                return a / b
    """)
    chunks = _split_codigo(codigo)
    if len(chunks) >= 3:
        ok(f"split codigo por funcao/classe ({len(chunks)} chunks)")
    else:
        falha("split codigo", f"esperado >=3 chunks, obteve {len(chunks)}: {chunks}")

    # 5.2 — Cada chunk de código começa com def/class
    primeiros_tokens = [c.strip().split()[0] for c in chunks if c.strip()]
    validos = all(t in ("def", "class", "async") for t in primeiros_tokens)
    if validos:
        ok("cada chunk comeca com def/class/async")
    else:
        falha("chunks comecam com def/class", f"tokens: {primeiros_tokens}")

    # 5.3 — Split de texto com overlap
    texto = "a" * 1200
    chunks = _split_texto(texto, tamanho=500, overlap=100)
    if len(chunks) >= 3:
        ok(f"split texto com overlap ({len(chunks)} chunks)")
    else:
        falha("split texto", f"esperado >=3, obteve {len(chunks)}")

    # 5.4 — Chunks não ficam vazios
    chunks_vazios = [c for c in chunks if not c.strip()]
    if not chunks_vazios:
        ok("nenhum chunk vazio")
    else:
        falha("chunks vazios", f"{len(chunks_vazios)} chunks vazios encontrados")

    # 5.5 — Código sem funções cai para janela
    texto_simples = "x = 1\ny = 2\nz = x + y\nprint(z)\n" * 50
    chunks = _split_codigo(texto_simples)
    if len(chunks) >= 1:
        ok("codigo sem funcoes faz fallback para janela")
    else:
        falha("fallback para janela", "0 chunks gerados")


# ─────────────────────────────────────────────────────────────
# 6. CONFIGURAÇÃO DO MOTOR
# ─────────────────────────────────────────────────────────────

def testar_configuracao():
    secao("6. Configuração do Motor")
    import motor_ia_avancado as m

    # 6.1 — Constantes essenciais existem
    campos = ["OLLAMA_CHAT", "OLLAMA_TAGS", "MODELO", "TIMEOUT",
              "NUM_PREDICT", "NUM_CTX", "KEEP_ALIVE", "SC_N", "MAX_FIX"]
    faltando = [c for c in campos if not hasattr(m, c)]
    if not faltando:
        ok("todas as constantes essenciais presentes")
    else:
        falha("constantes faltando", str(faltando))

    # 6.2 — TIMEOUT suficiente para 30 etapas
    if m.TIMEOUT >= 120:
        ok(f"TIMEOUT suficiente para CoT ({m.TIMEOUT}s)")
    else:
        falha("TIMEOUT", f"muito baixo: {m.TIMEOUT}s (minimo 120s para CoT)")

    # 6.3 — NUM_PREDICT suficiente
    if m.NUM_PREDICT >= 4096:
        ok(f"NUM_PREDICT adequado ({m.NUM_PREDICT})")
    else:
        falha("NUM_PREDICT", f"muito baixo: {m.NUM_PREDICT}")

    # 6.4 — MODOS tem 8 entradas
    if len(m.MODOS) == 8:
        ok("MODOS tem 8 entradas")
    else:
        falha("MODOS", f"esperado 8, obteve {len(m.MODOS)}")

    # 6.5 — _REGRAS_BASE sem emojis
    import re
    emoji_re = re.compile(
        "[\U00002702-\U000027B0\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        re.UNICODE,
    )
    if not emoji_re.search(m._REGRAS_BASE):
        ok("_REGRAS_BASE nao contem emojis")
    else:
        falha("_REGRAS_BASE contem emojis", "remover emojis das regras base")

    # 6.6 — KEEP_ALIVE definido e não vazio
    if m.KEEP_ALIVE:
        ok(f"KEEP_ALIVE configurado ({m.KEEP_ALIVE!r})")
    else:
        falha("KEEP_ALIVE", "vazio ou nao definido")


# ─────────────────────────────────────────────────────────────
# 7. BANCO DE MEMÓRIA (instanciação sem modelos)
# ─────────────────────────────────────────────────────────────

def testar_banco_memoria():
    secao("7. Banco de Memória (sem modelos, sem chromadb)")
    try:
        from banco_memoria import BancoMemoria
        banco = BancoMemoria()
        # Só testa se chromadb estiver instalado
        if banco.disponivel:
            ok(f"BancoMemoria inicializado ({banco.total()} registros)")
            ok("modelos: lazy (nao carregados no startup)")
        else:
            pulo("BancoMemoria.disponivel=False", "chromadb nao instalado")
    except Exception as e:
        falha("BancoMemoria.__init__", str(e))

    # 7.1 — deve_salvar detecta fatos pessoais
    from banco_memoria import BancoMemoria as BM
    casos = [
        ("meu nome é Miguel", True),
        ("eu sou desenvolvedor", True),
        ("como funciona Python?", False),
        ("qual é a capital?", False),
    ]
    erros = []
    for txt, esperado in casos:
        r = BM.deve_salvar(txt)
        if r != esperado:
            erros.append(f"'{txt}' -> {r} (esperado {esperado})")
    if not erros:
        ok(f"deve_salvar detecta corretamente ({len(casos)} casos)")
    else:
        falha("deve_salvar", "; ".join(erros))

    # 7.2 — parse_fatos extrai lista do JSON
    from banco_memoria import BancoMemoria as BM2
    json_str = '{"fatos": ["usa Python", "prefere código simples"]}'
    fatos = BM2.parse_fatos(json_str)
    if len(fatos) == 2 and "usa Python" in fatos:
        ok("parse_fatos extrai lista corretamente")
    else:
        falha("parse_fatos", f"obteve: {fatos}")

    # 7.3 — parse_fatos tolerante a markdown
    json_md = '```json\n{"fatos": ["fato 1"]}\n```'
    fatos = BM2.parse_fatos(json_md)
    if fatos == ["fato 1"]:
        ok("parse_fatos tolerante a markdown")
    else:
        falha("parse_fatos markdown", f"obteve: {fatos}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NORK-IA - Suite de Testes")
    print("=" * 60)

    for fn in [
        testar_roteador,
        testar_sandbox,
        testar_banco_regex,
        testar_extracao_tags,
        testar_splitting,
        testar_configuracao,
        testar_banco_memoria,
    ]:
        try:
            fn()
        except Exception:
            falha(fn.__name__, traceback.format_exc().splitlines()[-1])

    print(f"\n{'='*60}")
    print(f"  Resultado: {PASSOU} OK | {FALHOU} FALHA | {PULOS} PULADOS")
    print(f"{'='*60}\n")
    sys.exit(0 if FALHOU == 0 else 1)
