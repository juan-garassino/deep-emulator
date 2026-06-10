# deepEmulator — RunPod GPU training container
#
# Ports the autoresearch container pattern at
#   /Users/juan-garassino/Code/005-products/020-autoresearch/Dockerfile
# Differences vs. upstream:
#   - Installs deepEmulator's [atari,cloud] extras (no [viz], headless).
#   - Includes @anthropic-ai/claude-code but only invokes it when
#     MODE=self_improve AND CLAUDE_CODE_ENABLED=1 (Phase 0.5 toggle).
#
# Base is `base` not `devel`: the locked torch wheels bundle their own CUDA
# runtime libs — the devel toolkit was ~8 GB of dead weight per pod pull.
FROM nvidia/cuda:12.8.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# python-is-python3 is load-bearing: entrypoint.sh and the watchdog call
# `python`, which ubuntu's python3.10 packages do not provide.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip python-is-python3 \
        curl git ca-certificates rsync \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Claude Code is only invoked when MODE=self_improve AND CLAUDE_CODE_ENABLED=1.
# Always installed for one-image simplicity; presence alone has no effect.
RUN npm install -g @anthropic-ai/claude-code

# uv for fast reproducible installs (autoresearch parity). Build-time only.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Prime the dependency cache from the lockfile — invalidated only when
# pyproject.toml or uv.lock change, not when source changes.
# --no-emit-project keeps the project itself out of the layer (source is
# copied + installed below), preserving the cache claim.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --extra atari --extra cloud --format requirements-txt > /tmp/requirements.txt \
    && uv pip install --system --no-deps -r /tmp/requirements.txt

COPY deepEmulator ./deepEmulator
COPY scripts ./scripts
COPY program.md ./program.md
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh scripts/iteration_watchdog.sh 2>/dev/null || true

# Install the repo itself in editable mode on top of the locked deps.
RUN uv pip install --system --no-deps -e .

# Headless display vars — PyBoy/torch must never reach for an X server.
ENV PYBOY_HEADLESS=1
ENV MPLBACKEND=Agg

# Container working directories. /runs is where bundles are written before
# being synced to GCS; /data is where ROMs / init.states get staged;
# /secrets is where entrypoint.sh materializes the GCP SA key.
RUN mkdir -p /runs /data /baseline /secrets /workspace

# Non-root runtime (workspace convention) — especially important with
# --dangerously-skip-permissions in self-improve mode.
RUN useradd -m -u 10001 runner \
    && chown -R runner:runner /app /runs /data /baseline /secrets /workspace
USER runner
# git identity for the self-improve clone commits
RUN git config --global user.name "deepemu-self-improve" \
    && git config --global user.email "deepemu-self-improve@users.noreply.github.com"

ENTRYPOINT ["./entrypoint.sh"]
