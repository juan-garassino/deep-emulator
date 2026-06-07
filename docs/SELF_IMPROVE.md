# Self-improvement mode — reviewer guide

`MODE=self_improve` runs Claude Code autonomously inside a RunPod container. It edits `deepEmulator/`, scores changes against `deepemu-eval`, commits to a `claude-self-improve-<RUN_ID>` branch, and pushes the branch to GitHub for **human review**. **Never auto-merges to `master`.**

This doc is the reviewer's checklist for evaluating one of those branches.

---

## When you see a `claude-self-improve-*` branch land

`.github/workflows/review_self_improve.yml` fires automatically on push:

1. Runs the full `pytest` suite on the branch.
2. Validates the schema of `improvement_log.tsv`.
3. Sends a Telegram notification (`🤖 Self-improve branch needs review`) if the secrets are configured.

This is **just CI plumbing** — the workflow does not validate the *correctness* of any improvement. That's your job.

---

## The reviewer checklist

Open the PR (or `gh pr create` if the branch is not yet a PR).

### 1. Read the branch shape first

```bash
git fetch origin
git checkout claude-self-improve-<RUN_ID>
git log master..HEAD --oneline
cat improvement_log.tsv
```

Each commit should be prefixed `[self-improve <N>]` and named after the hypothesis. The log row should align with the commit.

**Red flag:** commits without the prefix, commits that touch `improvement_log.tsv` itself, commits that touch `scripts/eval_signed.py`, commits that touch `program.md`. Any of these means Claude went off-script — discard the branch.

### 2. Recompute the SHA256 signatures

The signed-eval guarantee is the only reason this mode is safe. Each row in `improvement_log.tsv` looks like:

```
ts                          baseline                       treatment                       score      sig
2026-06-07T12:34:56Z   /baseline                      /runs/train/<ts>               42.500000  abc123...
```

The `sig` column is `SHA256(sha256_dir(baseline) || sha256_dir(treatment) || score || ts)`. To verify:

```bash
python -c "
import hashlib
from pathlib import Path
from deepEmulator.utils import gcs

baseline = Path('/path/to/baseline')      # the read-only mount
treatment = Path('/path/to/treatment')     # the bundle the row references
score = 42.500000                          # from the row
ts = '2026-06-07T12:34:56Z'                # from the row

h = hashlib.sha256()
h.update(gcs.sha256_dir(baseline).encode())
h.update(gcs.sha256_dir(treatment).encode())
h.update(f'{score:.6f}'.encode())
h.update(ts.encode())
print(h.hexdigest())
"
```

The result should match the `sig` column. **If any row's signature does not recompute, treat the branch as untrusted and discard.** Claude either tampered with the eval, edited the bundle after scoring, or swapped the baseline.

### 3. Sanity-check the eval methodology

Even with valid signatures, Claude can game the eval by reducing rigor:

- **Episodes per run:** should be ≥ 5. Fewer = noisy.
- **Seeds:** should be ≥ 3. Single seed = single-anecdote, not improvement.
- **Steps:** matches the baseline's training horizon (5 k DDQN or whatever the baseline uses).
- **Cartridge + init.state:** identical to the baseline.

Open `eval_episodes.tsv` in the treatment bundle and confirm the run shape. If the treatment ran 3 episodes on 1 seed vs. the baseline's 15×3, the "improvement" is noise.

### 4. Read the diff with skepticism

`git diff master...HEAD` for each commit. Watch for:

- **Reward-function edits** — Claude may have made the reward easier to satisfy. If `core/reward.py` or `cartridges/pokemon_*.py` reward fields changed, scrutinize whether the change is genuine or shifts the goal posts.
- **Eval-harness edits** — `deepEmulator/training/eval.py` should not change. If it did, ask why.
- **Test deletions** — any deleted test is a red flag. New tests are fine.
- **Hyperparameter changes only** — fine; the cheapest legitimate improvement axis.
- **Architecture changes** (new layers, encoders, loss components) — fine if narrowly scoped; suspicious if sweeping.

### 5. Decide

Three outcomes:

| Outcome | Action |
| --- | --- |
| Genuine improvement, methodology sound | Merge the branch into `master` (or cherry-pick the best commits). |
| Signature mismatch / methodology cut corners / sweeping change | Close the PR without merging. Document the failure mode in this file's "Known failure patterns" section so we learn. |
| Methodology sound but improvement marginal (< 1 σ) | Leave the branch; do not merge. The next self-improve run gets a clean baseline. |

---

## Cost + budget guarantees (what the guardrails do)

| Guardrail | What it protects against | Where it lives |
| --- | --- | --- |
| `MAX_ITERATIONS` (default 10) | Runaway loops | `scripts/iteration_watchdog.sh` + `entrypoint.sh` |
| `MAX_WALLCLOCK_HOURS` (default 12) | Forgotten pod burning GPU | Background timer in `iteration_watchdog.sh` |
| Anthropic project-level spend cap | API token cost overrun | **Set in the Anthropic console — not in code.** Verify before enabling. |
| Read-only baseline mount (`-v $baseline:/baseline:ro`) | Goodhart by editing comparator | `Makefile` `runpod_run_self_improve` target |
| Branch isolation (`claude-self-improve-<RUN_ID>`) | Damage to `master` | `entrypoint.sh` self-improve branch |
| SHA256-signed eval rows | Tampered metrics | `scripts/eval_signed.py` |
| `appleboy/telegram-action` notify | Branch lands without a reviewer noticing | `.github/workflows/review_self_improve.yml` |

These are **audit, not sandbox.** A motivated adversarial prompt can technically bypass any of them — Claude has bash inside the pod. The signatures + branch isolation + spend caps make tampering costly + visible.

---

## How to actually invoke it

```bash
# Generate a fresh program.md if you want a different improvement target;
# otherwise the default at repo root is fine.

make runpod_run_self_improve \
    API_KEY=$ANTHROPIC_API_KEY \
    GCS_BUCKET=gs://garassino-ml-artifacts \
    GCS_PREFIX=deepemulator/self_improve/2026-06-07 \
    MAX_ITERATIONS=5 \
    MAX_WALLCLOCK_HOURS=4
```

In a real RunPod pod (not local Docker), add the env vars to the pod template instead of `make`.

---

## Known failure patterns

(Empty — populated as we observe them. Each entry: date, what happened, what the guardrails caught, what to change.)

---

## How to turn it off completely

Leave `CLAUDE_CODE_ENABLED` unset (the default). The `claude` binary is in the image but `entrypoint.sh`'s `MODE=self_improve` branch refuses to start without all three flags. The deterministic training path is unaffected.
