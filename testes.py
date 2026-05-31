"""
Script de validacao automatica — LLM local (GPT-2 PT + fine-tuning).
Cobre todas as funcoes do main.py e dados_treino.py.

Uso:
    python testes.py           (completo)
    python testes.py --rapido  (pula downloads e treino)

Resultado esperado: todos os testes com [OK].
"""

import os, sys, time, argparse, tempfile, json

# ── Relanca no venv CUDA se necessario ───────────────────────────────────────
_VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "venv_cuda", "Scripts", "python.exe")
if os.path.exists(_VENV) and os.path.abspath(sys.executable) != os.path.abspath(_VENV):
    os.execv(_VENV, [_VENV] + sys.argv)

import torch

# ── Argumentos ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rapido", action="store_true",
                    help="Pula testes que baixam modelos ou datasets")
ARGS = parser.parse_args()

# ── Helpers de resultado ──────────────────────────────────────────────────────
_resultados = []

def _ok(nome):
    print(f"  [OK]     {nome}")
    _resultados.append((nome, True, ""))

def _falhou(nome, motivo):
    print(f"  [FALHOU] {nome}")
    print(f"           {motivo}")
    _resultados.append((nome, False, motivo))

def executar(nome, fn):
    try:
        fn()
        _ok(nome)
    except AssertionError as e:
        _falhou(nome, f"Assertion: {e}")
    except Exception as e:
        _falhou(nome, f"{type(e).__name__}: {e}")

# ── Importa modulos ───────────────────────────────────────────────────────────
print("\n=== Importando modulos ===")
try:
    from main import (
        _fmt_t, _bar,
        gerar, pre_validar,
        _sanitizar, _extrair_resposta, _montar_prompt,
        _salvar_sessao, _carregar_sessao,
        CorpusDataset,
        DEVICE, BASE_MODEL, FINE_TUNE_EP, LR,
        MAX_TURNOS, SESSAO_F,
    )
    from dados_treino import (
        CORPUS, CORPUS_POR_FASE,
        CORPUS_TEXTO, CORPUS_QA, CORPUS_THINK,
        carregar_datasets_externos, _fmt_conversa,
    )
    print("  [OK] Importacao de main.py e dados_treino.py\n")
except Exception as e:
    print(f"  [FALHOU] Importacao: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — Utilitarios de display
# ══════════════════════════════════════════════════════════════════════════════
print("=== BLOCO 1: Utilitarios ===")

def t_fmt_segundos(): assert _fmt_t(45)   == "45s"
def t_fmt_minutos():  assert _fmt_t(125)  == "2m05s"
def t_fmt_horas():    assert _fmt_t(3661) == "1h01m01s"

def t_bar_inicio():
    assert _bar(0, 10, w=10) == "░" * 10
def t_bar_meio():
    assert _bar(5, 10, w=10) == "█" * 5 + "░" * 5
def t_bar_completo():
    assert _bar(10, 10, w=10) == "█" * 10
def t_bar_zero():
    b = _bar(0, 0, w=10)
    assert len(b) == 10

def t_cuda_disponivel():
    assert torch.cuda.is_available(), "CUDA deve estar disponivel"
    mb = torch.cuda.memory_allocated() // 1024**2
    assert isinstance(mb, int)

executar("_fmt_t: segundos",       t_fmt_segundos)
executar("_fmt_t: minutos",        t_fmt_minutos)
executar("_fmt_t: horas",          t_fmt_horas)
executar("_bar: 0%",               t_bar_inicio)
executar("_bar: 50%",              t_bar_meio)
executar("_bar: 100%",             t_bar_completo)
executar("_bar: divisao por zero", t_bar_zero)
executar("CUDA: disponivel",       t_cuda_disponivel)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — Corpus local
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BLOCO 2: Corpus local ===")

def t_corpus_tamanho():
    assert len(CORPUS) > 50_000, f"CORPUS pequeno: {len(CORPUS)}"
def t_corpus_fases():
    assert len(CORPUS_POR_FASE) == 3
def t_corpus_texto():
    assert len(CORPUS_TEXTO) > 10_000
def t_corpus_qa():
    assert len(CORPUS_QA) > 5_000
    assert "Pergunta:" in CORPUS_QA and "Resposta:" in CORPUS_QA
def t_corpus_think():
    assert len(CORPUS_THINK) > 5_000
    assert "[Pensando]" in CORPUS_THINK and "[/Pensando]" in CORPUS_THINK
def t_corpus_fases_nao_vazias():
    for i, c in enumerate(CORPUS_POR_FASE):
        assert len(c) > 100, f"fase {i+1} vazia"

executar("CORPUS: > 50k chars",       t_corpus_tamanho)
executar("CORPUS_POR_FASE: 3 fases",  t_corpus_fases)
executar("CORPUS_TEXTO: > 10k",       t_corpus_texto)
executar("CORPUS_QA: tem Q&A",        t_corpus_qa)
executar("CORPUS_THINK: tem think",   t_corpus_think)
executar("Todas as fases nao vazias", t_corpus_fases_nao_vazias)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — Datasets externos
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BLOCO 3: Datasets externos ===")

def t_ds_importavel():
    from dados_treino import carregar_datasets_externos
    assert callable(carregar_datasets_externos)

def t_ds_retorna_str():
    r = carregar_datasets_externos(max_exemplos_por_ds=0, verbose=False)
    assert isinstance(r, str)

def t_ds_sem_crash():
    try:
        r = carregar_datasets_externos(max_exemplos_por_ds=1, verbose=False)
        assert isinstance(r, str)
    except SystemExit:
        raise AssertionError("chamou sys.exit()")

def t_fmt_conversa_basico():
    turnos = [
        {"role": "user",      "content": "Ola"},
        {"role": "assistant", "content": "Oi, posso ajudar?"},
    ]
    r = _fmt_conversa(turnos)
    assert "Pergunta: Ola" in r
    assert "Resposta: Oi" in r

def t_fmt_conversa_roles_alternativos():
    # Testa roles variacoes (prompter, gpt, human)
    turnos = [
        {"role": "prompter", "content": "Q"},
        {"role": "gpt",      "content": "A"},
    ]
    r = _fmt_conversa(turnos)
    assert "Pergunta:" in r and "Resposta:" in r

def t_fmt_conversa_vazia():
    assert _fmt_conversa([]) == ""

def t_fmt_conversa_sem_content():
    # Turnos sem content nao devem gerar linha
    turnos = [{"role": "user", "content": ""}]
    r = _fmt_conversa(turnos)
    assert r.strip() == ""

executar("carregar_datasets_externos: importavel", t_ds_importavel)
executar("carregar_datasets_externos: retorna str",t_ds_retorna_str)
executar("carregar_datasets_externos: sem crash",  t_ds_sem_crash)
executar("_fmt_conversa: basico",                  t_fmt_conversa_basico)
executar("_fmt_conversa: roles alternativos",      t_fmt_conversa_roles_alternativos)
executar("_fmt_conversa: lista vazia",             t_fmt_conversa_vazia)
executar("_fmt_conversa: sem content",             t_fmt_conversa_sem_content)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — CorpusDataset (dataset de fine-tuning)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BLOCO 4: CorpusDataset ===")

def t_corpus_dataset_basico():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ds  = CorpusDataset("Hello world! " * 60, tok, max_len=32)
    assert len(ds) > 0, "dataset vazio"
    item = ds[0]
    assert "input_ids" in item and "labels" in item

def t_corpus_dataset_len_tensor():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ds  = CorpusDataset("Test text. " * 60, tok, max_len=32)
    x   = ds[0]["input_ids"]
    assert x.shape == (32,), f"shape errado: {x.shape}"
    assert x.dtype == torch.long

def t_corpus_dataset_labels_iguais_ids():
    from transformers import AutoTokenizer
    tok  = AutoTokenizer.from_pretrained("gpt2")
    ds   = CorpusDataset("Sample text. " * 60, tok, max_len=32)
    item = ds[0]
    assert torch.equal(item["input_ids"], item["labels"]), \
        "labels deve ser igual a input_ids para LM causal"

def t_corpus_dataset_texto_curto():
    # Texto curto demais (< max_len) resulta em dataset vazio sem crash
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ds  = CorpusDataset("curto", tok, max_len=512)
    assert len(ds) == 0, "texto curto deve resultar em dataset vazio"

if ARGS.rapido:
    print("  [PULADO] --rapido ativo (evita download do tokenizer gpt2)")
else:
    executar("CorpusDataset: estrutura basica",        t_corpus_dataset_basico)
    executar("CorpusDataset: shape e dtype",           t_corpus_dataset_len_tensor)
    executar("CorpusDataset: labels == input_ids",     t_corpus_dataset_labels_iguais_ids)
    executar("CorpusDataset: texto curto sem crash",   t_corpus_dataset_texto_curto)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — Sanitizacao e historico
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BLOCO 5: Sanitizacao e historico ===")

def t_sanitizar_normal():
    assert _sanitizar("Ola, tudo bem?") == "Ola, tudo bem?"
def t_sanitizar_limite():
    assert len(_sanitizar("a" * 600)) == 500
def t_sanitizar_ctrl():
    r = _sanitizar("texto\x00\x01normal")
    assert "\x00" not in r and "\x01" not in r

def t_extrair_com_resposta():
    r = _extrair_resposta("[Pensando] algo [/Pensando]\nResposta: Sim.")
    assert "Sim." in r
def t_extrair_com_think():
    r = _extrair_resposta("[Pensando] p [/Pensando]\nResultado X.")
    assert "Resultado X." in r
def t_extrair_sem_marcadores():
    r = _extrair_resposta("Resposta direta.")
    assert r == "Resposta direta."

def t_montar_prompt_vazio():
    p = _montar_prompt("O que e IA?", [])
    assert "Pergunta: O que e IA?" in p
def t_montar_prompt_historico():
    hist = [{"pergunta": "Ola", "resposta": "Oi"}]
    p    = _montar_prompt("Como vai?", hist)
    assert "Ola" in p and "Como vai?" in p
def t_montar_prompt_limite_turnos():
    # Historico com 10 turnos — deve incluir no maximo MAX_TURNOS
    hist = [{"pergunta": f"p{i}", "resposta": f"r{i}"} for i in range(10)]
    p    = _montar_prompt("nova", hist)
    # A pergunta mais antiga (p0) nao deve aparecer se MAX_TURNOS < 10
    if MAX_TURNOS < 10:
        assert "p0" not in p, "historico excedeu MAX_TURNOS"

def t_sessao_save_load():
    hist_orig = [{"pergunta": "teste", "resposta": "ok"}]
    arq = "_teste_sessao_temp_2.json"
    import main as _m
    orig_f = _m.SESSAO_F
    _m.SESSAO_F = arq
    try:
        _salvar_sessao(hist_orig)
        carregado = _carregar_sessao()
        assert carregado == hist_orig
    finally:
        _m.SESSAO_F = orig_f
        if os.path.exists(arq):
            os.unlink(arq)

executar("_sanitizar: normal",               t_sanitizar_normal)
executar("_sanitizar: limite 500",           t_sanitizar_limite)
executar("_sanitizar: ctrl chars",           t_sanitizar_ctrl)
executar("_extrair_resposta: com Resposta:", t_extrair_com_resposta)
executar("_extrair_resposta: com /Pensando", t_extrair_com_think)
executar("_extrair_resposta: sem marcador",  t_extrair_sem_marcadores)
executar("_montar_prompt: sem historico",    t_montar_prompt_vazio)
executar("_montar_prompt: com historico",    t_montar_prompt_historico)
executar("_montar_prompt: limite turnos",    t_montar_prompt_limite_turnos)
executar("sessao: save e load",              t_sessao_save_load)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — pre_validar
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BLOCO 6: pre_validar ===")

def t_pv_valida():
    assert pre_validar("Inteligencia artificial e a capacidade de maquinas aprenderem.", "q")
def t_pv_curta():
    assert not pre_validar("Ok.", "q")
def t_pv_vazia():
    assert not pre_validar("", "q")
def t_pv_so_pensamento():
    assert not pre_validar("[Pensando] algo [/Pensando]", "q")

executar("pre_validar: resposta valida",   t_pv_valida)
executar("pre_validar: resposta curta",    t_pv_curta)
executar("pre_validar: vazia",             t_pv_vazia)
executar("pre_validar: so pensamento",     t_pv_so_pensamento)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 7 — Configuracoes e ambiente
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BLOCO 7: Configuracoes ===")

def t_device():
    assert DEVICE in ("cuda", "cpu")
def t_cuda():
    assert torch.cuda.is_available(), "CUDA deve estar disponivel"
def t_base_model_str():
    assert isinstance(BASE_MODEL, str) and len(BASE_MODEL) > 0
def t_lr_positivo():
    assert 0 < LR < 1, f"LR fora do range: {LR}"
def t_epochs_positivo():
    assert FINE_TUNE_EP > 0
def t_max_turnos():
    assert MAX_TURNOS > 0

executar("DEVICE: valido",        t_device)
executar("CUDA: disponivel",      t_cuda)
executar("BASE_MODEL: string",    t_base_model_str)
executar("LR: positivo e < 1",   t_lr_positivo)
executar("FINE_TUNE_EP: > 0",    t_epochs_positivo)
executar("MAX_TURNOS: > 0",      t_max_turnos)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 8 — Modelo fine-tunado (se existir checkpoint)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BLOCO 8: Modelo fine-tunado ===")

_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_finetuned")

if not os.path.isdir(_CHECKPOINT) or ARGS.rapido:
    print("  [PULADO] checkpoint nao encontrado ou --rapido ativo")
else:
    def t_checkpoint_carrega():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok   = AutoTokenizer.from_pretrained(_CHECKPOINT)
        model = AutoModelForCausalLM.from_pretrained(_CHECKPOINT)
        assert tok is not None and model is not None

    def t_modelo_gera_tokens():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok   = AutoTokenizer.from_pretrained(_CHECKPOINT)
        model = AutoModelForCausalLM.from_pretrained(_CHECKPOINT).to(DEVICE)
        model.eval()
        ids = tok.encode("Pergunta: O que e IA?", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=20, do_sample=False)
        assert out.shape[1] > ids.shape[1], "modelo nao gerou tokens novos"

    def t_gerar_retorna_string():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok   = AutoTokenizer.from_pretrained(_CHECKPOINT)
        model = AutoModelForCausalLM.from_pretrained(_CHECKPOINT).to(DEVICE)
        r = gerar(tok, model, "Pergunta: O que e Python?", mostrar_pensamento=False)
        assert isinstance(r, str), f"gerar deve retornar str, got {type(r)}"

    executar("Checkpoint: carrega sem erro",    t_checkpoint_carrega)
    executar("Modelo: gera tokens novos",       t_modelo_gera_tokens)
    executar("gerar(): retorna string",         t_gerar_retorna_string)

# ══════════════════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ══════════════════════════════════════════════════════════════════════════════
total  = len(_resultados)
ok     = sum(1 for _, p, _ in _resultados if p)
falhou = total - ok

print("\n" + "=" * 50)
print(f"  RESULTADO: {ok}/{total} testes passaram")
if falhou:
    print(f"\n  FALHAS ({falhou}):")
    for nome, passou, motivo in _resultados:
        if not passou:
            print(f"    - {nome}: {motivo}")
print("=" * 50 + "\n")

sys.exit(0 if falhou == 0 else 1)
