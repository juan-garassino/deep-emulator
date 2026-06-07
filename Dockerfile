# deepEmulator — RunPod GPU training container
#
# Ports the autoresearch container pattern at
#   /Users/juan-garassino/Code/005-products/020-autoresearch/Dockerfile
# Differences vs. upstream:
#   - Installs deepEmulator's [atari,cloud] extras (no [viz], headless).
#   - Includes @anthropic-ai/claude-code but only invokes it when
#     MODE=self_improve AND CLAUDE_CODE_ENABLED=1 (Phase 0.5 toggle).
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        curl git ca-certificates rsync \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Claude Code is only invoked when MODE=self_improve AND CLAUDE_CODE_ENABLED=1.
# Always installed for one-image simplicity; presence alone has no effect.
RUN npm install -g @anthropic-ai/claude-code

# uv for fast reproducible installs (autoresearch parity).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Google Cloud SDK (gsutil) for entrypoint-level rsync to GCS.
RUN curl -sSL https://sdk.cloud.google.com | bash > /dev/null
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

WORKDIR /app

# Prime the dependency cache from the lockfile — invalidated only when
# pyproject.toml or uv.lock change, not when source changes.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --extra atari --extra cloud --format requirements-txt > /tmp/requirements.txt \
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
# being rsynced to GCS; /data is where ROMs / init.states get staged.
RUN mkdir -p /runs /data /baseline

ENTRYPOINT ["./entrypoint.sh"]
