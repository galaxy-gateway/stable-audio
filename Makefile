# Makefile for stable-audio-3-small-music generator.
# Run `make` or `make help` to list targets.

# --- Config (override on the command line, e.g. `make gen PROMPT="jazz 90 BPM")
PYBIN    ?= python3.10
VENV     := .venv
PY       := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
PROMPT     ?= deep house warm pads 124 BPM
PROMPT_FILE ?=
DURATION   ?= 30
STEPS      ?= 8
CFG        ?= 1.0
OUT        ?= outputs/output.wav
SEED       ?=

# Pass --seed only when SEED is set.
SEED_ARG := $(if $(SEED),--seed $(SEED),)
# Prefer a file when PROMPT_FILE is set (best for multiline lyric sheets);
# otherwise pass PROMPT inline. generate.py sanitizes either way.
PROMPT_ARG := $(if $(PROMPT_FILE),--prompt-file "$(PROMPT_FILE)","$(PROMPT)")

.DEFAULT_GOAL := help

.PHONY: help setup env check gen play clean clean-cache reinstall

help: ## Show this help
	@echo "stable-audio generator"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Vars: PROMPT DURATION STEPS CFG OUT SEED  (e.g. make gen PROMPT=\"lofi 85 BPM\" DURATION=60)"

setup: $(VENV) env ## Create venv, install deps, scaffold .env

$(VENV): requirements.txt ## (internal) build venv when requirements change
	@command -v $(PYBIN) >/dev/null || { echo "ERROR: $(PYBIN) not found. brew install python@3.10"; exit 1; }
	$(PYBIN) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ">> overriding stable-audio-tools' PyWavelets==1.4.1 pin (numpy 2 ABI fix)"
	$(PIP) install --upgrade "PyWavelets>=1.7"
	@touch $(VENV)

env: ## Create .env from template if missing
	@test -f .env || { cp .env.example .env; echo "Created .env - paste your HF_TOKEN into it."; }

check: ## Verify env: token set, torch + MPS, deps importable
	@test -f .env || { echo "MISSING .env (run: make env)"; exit 1; }
	@grep -q '^HF_TOKEN=hf_' .env || { echo "HF_TOKEN not set in .env"; exit 1; }
	@$(PY) -c "import torch, stable_audio_tools, soundfile; \
print('torch', torch.__version__, '| mps', torch.backends.mps.is_available())"
	@echo "OK - ready to generate."

gen: ## Generate audio (uses PROMPT or PROMPT_FILE, DURATION STEPS CFG OUT SEED)
	@test -d $(VENV) || { echo "Run 'make setup' first."; exit 1; }
	$(PY) generate.py $(PROMPT_ARG) --duration $(DURATION) --steps $(STEPS) \
		--cfg-scale $(CFG) --out "$(OUT)" $(SEED_ARG)

play: ## Play the last output (macOS afplay)
	@test -f "$(OUT)" || { echo "No file at $(OUT). Run 'make gen' first."; exit 1; }
	afplay "$(OUT)"

clean: ## Remove generated audio
	rm -rf outputs

clean-cache: ## Remove downloaded model weights from HF cache
	rm -rf ~/.cache/huggingface/hub/models--stabilityai--stable-audio-3-small-music

reinstall: ## Nuke venv and rebuild from scratch
	rm -rf $(VENV)
	$(MAKE) setup
