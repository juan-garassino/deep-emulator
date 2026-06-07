# deepEmulator — autonomous self-improvement program

You are running inside a RunPod GPU container, with full bash access. Your job
is to iteratively improve the deepEmulator codebase against the head-to-head
eval harness `deepemu-eval`, producing a sequence of small, individually-scored
git commits on the branch `claude-self-improve-${RUN_ID}` (already created).

## Ground rules

1. **Never touch `master`.** Stay on your self-improve branch. No `git push origin master`. No merges.
2. **Never edit the baseline bundle.** It is mounted at `/baseline:ro`. The eval comparator is fixed.
3. **Score with `scripts/eval_signed.py`, not bare `deepemu-eval`.** Each accepted improvement appends a SHA256-signed row to `improvement_log.tsv`. A reviewer will reject any row whose signature does not recompute from the on-disk bundles.
4. **Stop at `MAX_ITERATIONS` (default 10).** A background watchdog also enforces `MAX_WALLCLOCK_HOURS` (default 12). Do not try to defeat either.
5. **Commit message format:** `[self-improve <N>] <one-line summary> | score=<float> | sig=<short>`.
6. **No backwards-compat shims, no broad refactors.** One focused hypothesis per iteration. Tests must pass after each commit (`make test`).

## Each iteration

```
1. Read CLAUDE.md, last 5 commits, current improvement_log.tsv.
2. Form one hypothesis ("changing X improves the metric because Y").
3. Implement the smallest code change that tests the hypothesis.
4. Run: make test                           — must pass
5. Run: make pretrain_vjepa STEPS=2000      (or train_pixel STEPS=2000 if pre-Phase-1)
6. Run: python scripts/eval_signed.py \
        --baseline /baseline \
        --treatment <your candidate run_dir> \
        --cartridge "POKEMON CORAL" \
        --rom /data/PokemonCoral.gbc \
        --init-state /data/coral_init.state \
        --episodes 5 \
        --log improvement_log.tsv
7. Read the score. If better than the running best: commit. If worse: revert the change, log the negative result in improvement_log.tsv (still signed), do NOT commit code.
8. iter_increment (sources the watchdog helper if you want the counter file).
```

## When you finish

Push the branch (the entrypoint handles this on exit; you can also `git push origin HEAD` mid-loop).
Print a final summary to stdout: best score, baseline delta, top 3 commits.

## What you should NOT do

- Do not run `git push --force` anywhere.
- Do not delete `improvement_log.tsv` rows.
- Do not edit `scripts/eval_signed.py` to weaken the signature.
- Do not network-call anywhere except `api.anthropic.com`, `github.com`, GCS, ghcr.io. The container may have egress rules; treat unexpected DNS failures as intentional.

Go.
