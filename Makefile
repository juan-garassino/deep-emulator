# deepEmulator — Make targets
#
# `make help` prints the full workflow grouped by stage.
# All cartridge-specific verbs default to Pokemon Coral but accept env-var overrides:
#   make train_pixel CART="POKEMON RED" ROM=roms/PokemonRed.gb INIT_STATE=states/init.state

ROM           ?= roms/PokemonCoral.gbc
CART          ?= POKEMON CORAL
INIT_STATE    ?= states/coral_init.state
STEPS         ?= 5000
EPISODE_STEPS ?= 2048
SAVE_EVERY    ?= 1000
DRIVE         ?= $(HOME)/Google\ Drive/My\ Drive/deepEmulator
MODE          ?= train

# DINO defaults
CORPUS        ?= data/frames/coral_local
DINO_STEPS    ?= 1000
DINO_BATCH    ?= 32
DINO_SAVE     ?= 250
N_FRAMES      ?= 5000

# Cartridge slug used by checkpoints/<slug>/latest.txt
SLUG          := $(shell echo "$(CART)" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
LATEST_PTR    := checkpoints/$(SLUG)/latest.txt
LATEST_ENC    := encoders/dino/latest.txt

PY            := PYTHONPATH=. python

.DEFAULT_GOAL := help

# ============================================================================
# help — list every target grouped by stage
# ============================================================================
help:
	@printf "deepEmulator — workflow\n\n"
	@printf "  CONFIG (override via env or 'make X=Y target'):\n"
	@printf "    ROM=%s\n" '$(ROM)'
	@printf "    CART=%s\n" '$(CART)'
	@printf "    INIT_STATE=%s\n" '$(INIT_STATE)'
	@printf "    STEPS=%s   EPISODE_STEPS=%s   SAVE_EVERY=%s\n" '$(STEPS)' '$(EPISODE_STEPS)' '$(SAVE_EVERY)'
	@printf "    CORPUS=%s   DINO_STEPS=%s   N_FRAMES=%s\n" '$(CORPUS)' '$(DINO_STEPS)' '$(N_FRAMES)'
	@printf "    DRIVE=%s\n\n" '$(DRIVE)'
	@awk 'BEGIN{FS=":.*?## "} /^## ===/{printf "\n\033[1m%s\033[0m\n", substr($$0, 8)} /^[a-zA-Z_-]+:.*?## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n  TYPICAL FLOW:\n"
	@printf "    1. make install_dev\n"
	@printf "    2. make verify_ram             # sanity-check RAM addresses on real ROM\n"
	@printf "    3. make smoke_rom              # 200-step random-play smoke\n"
	@printf "    4. make play                   # record states/coral_init.state manually (~5 min)\n"
	@printf "    5a. PIXEL BASELINE:  make train_pixel STEPS=100000\n"
	@printf "    5b. DINO PIPELINE:   make collect_frames + make pretrain_dino + make train_encoder\n"
	@printf "    6. make eval / make visualize / make attention\n"
	@printf "    7. make sync_to_drive DRIVE=...   # for Colab\n\n"

# ============================================================================
## === INSTALL ===
# ============================================================================
install: ## minimal install: pyboy + torch + numpy
	@pip install -e . -U

install_dev: ## + dev (pytest/black/flake8) + play (pygame) + atari
	@pip install -e ".[dev,play,atari]" -U

install_cloud: ## + cloud (google-cloud-storage + wandb) — for RunPod images
	@pip install -e ".[cloud]" -U

lock: ## generate uv.lock from pyproject.toml (run after editing deps)
	@uv lock

install_atari: ## + ale-py for Atari ROMs
	@pip install -e ".[atari]" -U

install_sega: ## + stable-retro for SEGA Genesis (planned)
	@pip install -e ".[sega]" -U

install_all: ## everything (dev + play + atari + sega + cloud)
	@pip install -e ".[dev,play,atari,sega,cloud]" -U

# ============================================================================
## === QUALITY ===
# ============================================================================
smoke: ## one-line import check (no ROM)
	@$(PY) -c "import deepEmulator; print('deepEmulator', deepEmulator.__version__, 'OK')"

test: ## full pytest with coverage
	@coverage run -m pytest tests/
	@coverage report -m --omit="$$VIRTUAL_ENV/lib/python*"

test_fast: ## fast subset — excludes slow ROM smokes + nano e2e
	@$(PY) -m pytest tests/ -q --timeout 600 --ignore=tests/test_generic_gb.py --ignore=tests/test_nano_e2e.py

black: ## auto-format deepEmulator/ + tests/
	@black deepEmulator tests

check_code: ## flake8 lint
	@flake8 deepEmulator

# ============================================================================
## === CARTRIDGE WORKFLOW (defaults: Coral) ===
# ============================================================================
verify_ram: ## load $(CART) + print dump_state — sanity-check # VERIFY RAM addresses
	@$(PY) -c "from deepEmulator.cartridges import pokemon_coral, pokemon_crystal, pokemon_red; \
		from deepEmulator.cartridges.pokemon_crystal import dump_state; \
		from deepEmulator.core import registry; \
		from deepEmulator.platforms.gameboy import PyBoyEnv; \
		a = registry.get('$(CART)')(); \
		env = PyBoyEnv(a, rom_path='$(ROM)', headless=True, max_steps=20); \
		env.reset(); print(dump_state(env.pyboy)); env.close()"

smoke_rom: ## 200 random steps against $(ROM) — verifies env + reward end-to-end (strict reward mode)
	@$(PY) -c "import numpy as np; \
		from deepEmulator.cartridges import load_all; load_all(); \
		from deepEmulator.core import registry; \
		from deepEmulator.platforms.gameboy import PyBoyEnv; \
		a = registry.get('$(CART)')(reward_strict=True); \
		env = PyBoyEnv(a, rom_path='$(ROM)', headless=True, max_steps=300); \
		env.reset(); rng = np.random.default_rng(0); rewards = []; \
		[rewards.append(env.step(int(rng.integers(0, env.action_space.n)))[1]) for _ in range(200)]; \
		print(f'sum={sum(rewards):.3f} max={max(rewards):.4f} nonzero={sum(1 for r in rewards if r!=0)}/200'); \
		print(f'phase: {a.current_phase()}'); env.close()"

play: ## open visible PyBoy window for recording an init.state (close window to exit)
	@$(PY) -c "from pyboy import PyBoy; \
		p = PyBoy('$(ROM)', window='SDL2'); p.set_emulation_speed(1); \
		print('controls: arrows + Z(A) + A(B) + Enter(Start) + Space(Select)'); \
		print('to save state: in a python REPL, open(\"$(INIT_STATE)\",\"wb\").write(p.save_state(...))'); \
		[p.tick(1, True) for _ in range(99999999)]"

# ============================================================================
## === DDQN TRAINING ===
# ============================================================================
train_pixel: ## headless DDQN training on raw pixels (Phase I baseline)
	@mkdir -p states checkpoints
	@if [ -f "$(INIT_STATE)" ]; then \
		$(PY) -m deepEmulator.training.train --cartridge "$(CART)" --rom $(ROM) \
			--init-state $(INIT_STATE) --steps $(STEPS) --max-episode-steps $(EPISODE_STEPS) \
			--save-every $(SAVE_EVERY) --headless; \
	else \
		echo "[train_pixel] no $(INIT_STATE) — running boot-only (plumbing-validation regime)"; \
		$(PY) -m deepEmulator.training.train --cartridge "$(CART)" --rom $(ROM) \
			--steps $(STEPS) --max-episode-steps $(EPISODE_STEPS) \
			--save-every $(SAVE_EVERY) --headless; \
	fi

train_encoder: ## headless DDQN training on frozen DINO latents (Phase II)
	@if [ ! -f $(LATEST_ENC) ]; then echo "no $(LATEST_ENC) — run 'make pretrain_dino' first"; exit 1; fi
	@ENC=$$(cat $(LATEST_ENC)); echo "[train_encoder] using encoder: $$ENC"; \
		if [ -f "$(INIT_STATE)" ]; then \
			$(PY) -m deepEmulator.training.train --cartridge "$(CART)" --rom $(ROM) \
				--init-state $(INIT_STATE) --steps $(STEPS) --max-episode-steps $(EPISODE_STEPS) \
				--save-every $(SAVE_EVERY) --encoder $$ENC --headless; \
		else \
			$(PY) -m deepEmulator.training.train --cartridge "$(CART)" --rom $(ROM) \
				--steps $(STEPS) --max-episode-steps $(EPISODE_STEPS) \
				--save-every $(SAVE_EVERY) --encoder $$ENC --headless; \
		fi

# ============================================================================
## === DINO / SSL PIPELINE (Phase II) ===
# ============================================================================
collect_frames: ## build a frame corpus from $(CART) ($(N_FRAMES) frames → $(CORPUS))
	@$(PY) -m deepEmulator.training.collect_frames \
		--cartridges "$(CART)" --rom $(ROM) --frames $(N_FRAMES) \
		--out $(CORPUS)

collect_multi: ## multi-cartridge corpus (Coral + Crystal + Red if ROMs present)
	@ROMS=""; CARTS=""; \
		for pair in "roms/PokemonCoral.gbc:POKEMON CORAL" "roms/PokemonCrystal.gbc:POKEMON CRYSTAL" "roms/PokemonRed.gb:POKEMON RED"; do \
			r=$${pair%:*}; c=$${pair#*:}; \
			if [ -f $$r ]; then ROMS="$$ROMS $$r"; CARTS="$$CARTS \"$$c\""; fi; \
		done; \
		if [ -z "$$ROMS" ]; then echo "no ROMs found in roms/"; exit 1; fi; \
		echo "[collect_multi] envs: $$CARTS"; \
		bash -c "$(PY) -m deepEmulator.training.collect_frames \
			--cartridges $$CARTS --rom $$ROMS \
			--frames $(N_FRAMES) --out data/frames/multi"

pretrain_dino: ## SSL-pretrain ViT-tiny on $(CORPUS) ($(DINO_STEPS) iterations)
	@if [ ! -d "$(CORPUS)" ]; then echo "no corpus at $(CORPUS) — run 'make collect_frames' first"; exit 1; fi
	@$(PY) -m deepEmulator.training.pretrain_dino \
		--corpus $(CORPUS) --steps $(DINO_STEPS) --batch-size $(DINO_BATCH) \
		--save-every $(DINO_SAVE) --log-every 50

latest_encoder: ## print path to the latest DINO encoder bundle
	@cat $(LATEST_ENC) 2>/dev/null || echo "no $(LATEST_ENC) yet — run 'make pretrain_dino'"

# ============================================================================
## === VISUALIZATION + EVAL ===
# ============================================================================
visualize: ## render arrow trajectory PNG from latest training run
	@if [ ! -f $(LATEST_PTR) ]; then echo "no $(LATEST_PTR) yet"; exit 1; fi
	@RUN=$$(cat $(LATEST_PTR)); \
		$(PY) -m deepEmulator.visualization.arrows \
			--trajectories $$RUN/trajectories --cartridge "POKEMON CRYSTAL" \
			--out arrows_$(SLUG).png && echo "wrote arrows_$(SLUG).png"

attention: ## render attention-rollout GIF for $(CART) using latest encoder
	@if [ ! -f $(LATEST_ENC) ]; then echo "no encoder — run 'make pretrain_dino' first"; exit 1; fi
	@ENC=$$(cat $(LATEST_ENC)); \
		$(PY) -m deepEmulator.visualization.attention \
			--encoder $$ENC --cartridge "$(CART)" --rom $(ROM) \
			$(if $(wildcard $(INIT_STATE)),--init-state $(INIT_STATE),) \
			--steps 200 --out attention_$(SLUG).gif && echo "wrote attention_$(SLUG).gif"

eval: ## head-to-head: PIXEL=<pixel_bundle> TREAT=<treatment_bundle> make eval
	@if [ -z "$(PIXEL)" ] || [ -z "$(TREAT)" ]; then \
		echo "usage: make eval PIXEL=checkpoints/$(SLUG)/<pixel_run> TREAT=checkpoints/$(SLUG)/<dino_run>"; \
		exit 1; \
	fi
	@$(PY) -m deepEmulator.training.eval \
		--baseline $(PIXEL) --treatment $(TREAT) \
		--cartridge "$(CART)" --rom $(ROM) \
		$(if $(wildcard $(INIT_STATE)),--init-state $(INIT_STATE),) \
		--episodes 10 --out eval_report.html && echo "wrote eval_report.html"

# ============================================================================
## === INSPECTION ===
# ============================================================================
metrics: ## tail the latest DDQN run's metrics.tsv
	@if [ -f $(LATEST_PTR) ]; then \
		RUN=$$(cat $(LATEST_PTR)); echo "run: $$RUN"; tail -20 $$RUN/metrics.tsv; \
	else echo "no $(LATEST_PTR) — run 'make train_pixel' or 'make train_encoder' first"; fi

dino_metrics: ## tail the latest DINO pretraining metrics.tsv
	@if [ -f $(LATEST_ENC) ]; then \
		RUN=$$(cat $(LATEST_ENC)); echo "encoder: $$RUN"; tail -20 $$RUN/metrics.tsv; \
	else echo "no $(LATEST_ENC) yet"; fi

latest_bundle: ## print path to the latest DDQN bundle
	@cat $(LATEST_PTR) 2>/dev/null || echo "no $(LATEST_PTR) yet"

# ============================================================================
## === DEMO + DRIVE ===
# ============================================================================
nano: ## synthetic-env full pipeline (no ROM): collect → DINO → DDQN → eval → GIF
	@$(PY) scripts/nano_e2e.py --out /tmp/deepemu_nano_e2e

sync_to_drive: ## rsync repo + ROMs + states to $(DRIVE) (for Colab pip install)
	@mkdir -p $(DRIVE)
	@echo "syncing repo to $(DRIVE) ..."
	@rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' \
		--exclude='checkpoints/*/' --exclude='encoders/*/' --exclude='data/frames/*/' \
		--exclude='*.egg-info' --exclude='.coverage' \
		./ $(DRIVE)/
	@echo "done. on Colab: !pip install /content/drive/MyDrive/deepEmulator"

# ============================================================================
## === HOUSEKEEPING ===
# ============================================================================
clean: ## remove build artifacts, __pycache__, .coverage
	@rm -f .coverage
	@find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -fr build dist *.egg-info

all: clean install_dev smoke test_fast ## full local-dev refresh

count_lines: ## LOC by file under deepEmulator/
	@find ./deepEmulator -name '*.py' -exec wc -l {} \; | sort -n

# ============================================================================
## === RUNPOD + GCS ===
# ============================================================================
# Image rebuilds are pinned to uv.lock (committed). After editing pyproject.toml,
# run `make lock` and commit the resulting uv.lock so the Dockerfile's
# `uv export --frozen` step resolves consistently.
IMAGE_NAME ?= deepemulator
GH_USER    ?= juan-garassino
REGISTRY   ?= ghcr.io/$(GH_USER)
TAG        ?= latest

# GCP target (canonical layout — see workspace root CLAUDE.md § "GCP architecture"):
#   project:  garassino-ml
#   region:   europe-west1
#   bucket:   gs://garassino-ml-artifacts/deepemulator/   (prefix in shared bucket)
GCP_PROJECT ?= garassino-ml
GCP_REGION  ?= europe-west1
GCS_BUCKET  ?= gs://garassino-ml-artifacts
GCS_PREFIX  ?= deepemulator/default

runpod_build: ## docker build the training image locally
	@docker build -t $(IMAGE_NAME):$(TAG) .

runpod_push: ## docker push to ghcr.io (requires GITHUB_TOKEN env var with write:packages)
	@if [ -z "$$GITHUB_TOKEN" ]; then \
		echo "GITHUB_TOKEN is required. Get a PAT with write:packages scope and:"; \
		echo "  export GITHUB_TOKEN=ghp_..."; \
		echo "  echo \$$GITHUB_TOKEN | docker login ghcr.io -u $(GH_USER) --password-stdin"; \
		exit 1; \
	fi
	@echo "$$GITHUB_TOKEN" | docker login ghcr.io -u $(GH_USER) --password-stdin
	@docker tag $(IMAGE_NAME):$(TAG) $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	@docker push $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	@echo "image: $(REGISTRY)/$(IMAGE_NAME):$(TAG)"
	@echo "RunPod setup: GPU pod, container image above, env GCS_BUCKET + GCS_PREFIX + MODE."

runpod_run_local: ## docker run locally with GPU + a file:// GCS surrogate (smoke test)
	@mkdir -p /tmp/deepemu
	@docker run --rm --gpus all \
		-e GCS_BUCKET=file:///tmp/deepemu \
		-e GCS_PREFIX=local-smoke \
		-e MODE=$(MODE) \
		-e CARTRIDGE='$(CART)' \
		-e STEPS=$(STEPS) \
		-v /tmp/deepemu:/tmp/deepemu \
		$(IMAGE_NAME):$(TAG)

runpod_run_self_improve: ## DANGER — autonomous Claude Code loop in the container
	@if [ -z "$$ANTHROPIC_API_KEY" ]; then echo "ANTHROPIC_API_KEY required"; exit 1; fi
	@if [ -z "$$GITHUB_TOKEN" ]; then echo "GITHUB_TOKEN required (for branch push)"; exit 1; fi
	@mkdir -p /tmp/deepemu
	@docker run --rm --gpus all \
		-e GCS_BUCKET=$(GCS_BUCKET) \
		-e GCS_PREFIX=$(GCS_PREFIX) \
		-e MODE=self_improve \
		-e CLAUDE_CODE_ENABLED=1 \
		-e ANTHROPIC_API_KEY=$$ANTHROPIC_API_KEY \
		-e GITHUB_TOKEN=$$GITHUB_TOKEN \
		-e MAX_ITERATIONS=$(or $(MAX_ITERATIONS),10) \
		-e MAX_WALLCLOCK_HOURS=$(or $(MAX_WALLCLOCK_HOURS),12) \
		$(IMAGE_NAME):$(TAG)

runpod_logs: ## follow container logs (running container only)
	@docker logs -f $$(docker ps -q --filter ancestor=$(IMAGE_NAME):$(TAG) | head -1)

tf_init: ## terraform init (GCS backend at gs://garassino-op-tf-state/deepemulator/)
	@cd infra && terraform init -input=false

tf_validate: ## terraform fmt + validate
	@cd infra && terraform fmt -check && terraform validate

tf_plan: tf_init ## terraform plan for $(GCP_PROJECT)
	@cd infra && terraform plan -input=false

tf_apply: tf_init ## terraform apply — creates the SA + IAM bindings + key
	@cd infra && terraform apply -input=false

tf_destroy: tf_init ## terraform destroy — tears down SA + IAM (artifacts in bucket persist)
	@cd infra && terraform destroy -input=false

tf_output_sa_key: ## write SA key JSON to /tmp/gcp-sa-deepemu.json — upload to RunPod as Pod-secret 'gcp-sa-deepemu'
	@cd infra && terraform output -raw sa_key_json > /tmp/gcp-sa-deepemu.json
	@echo "wrote /tmp/gcp-sa-deepemu.json (gitignored)"
	@echo "next: RunPod UI -> Secrets -> Add Secret 'gcp-sa-deepemu' = contents of that file"
	@echo "      then mount in pod at /secrets/gcp-sa.json (entrypoint expects GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-sa.json)"

gcs_pull_latest: ## pull the latest run bundle from GCS to ./checkpoints (BUCKET=... PREFIX=...)
	@if [ -z "$(BUCKET)" ] || [ -z "$(PREFIX)" ]; then \
		echo "usage: make gcs_pull_latest BUCKET=gs://garassino-ml-artifacts PREFIX=deepemulator/coral/run-001"; \
		exit 1; \
	fi
	@gsutil -m rsync -r $(BUCKET)/$(PREFIX)/ ./checkpoints/$(PREFIX)/
	@echo "pulled to ./checkpoints/$(PREFIX)/"

.PHONY: help install install_dev install_cloud install_atari install_sega install_all lock \
	smoke test test_fast black check_code \
	verify_ram smoke_rom play \
	train_pixel train_encoder \
	collect_frames collect_multi pretrain_dino latest_encoder \
	visualize attention eval \
	metrics dino_metrics latest_bundle \
	nano sync_to_drive clean all count_lines \
	runpod_build runpod_push runpod_run_local runpod_run_self_improve runpod_logs gcs_pull_latest \
	tf_init tf_validate tf_plan tf_apply tf_destroy tf_output_sa_key
