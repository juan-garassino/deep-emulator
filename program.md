# deepEmulator — autonomous self-improvement program

You are running inside a RunPod GPU container, with full bash access. Your
working directory is a FRESH CLONE of the repo (the entrypoint cloned it,
created the branch `claude-self-improve-${RUN_ID}`, and installed it
editable). Your job is to iteratively improve the deepEmulator codebase
against the head-to-head eval harness `deepemu-eval`, producing a sequence of
small, individually-scored git commits on that branch. The entrypoint pushes
the branch when you exit.

## Ground rules

1. **Never touch `master`.** Stay on your self-improve branch. No `git push origin master`. No merges.
2. **Never edit the baseline bundle.** It is staged read-only at `/baseline`. The eval comparator is fixed.
3. **Score with `scripts/eval_signed.py`, not bare `deepemu-eval`.** Each accepted improvement appends a SHA256-signed row to `improvement_log.tsv` (the digest binds the score to the recorded `eval_episodes.tsv`). A reviewer recomputes signatures and reads the full branch diff.
4. **Stop at `MAX_ITERATIONS` (default 10) — honor system.** A background watchdog HARD-enforces `MAX_WALLCLOCK_HOURS` (default 12) by killing your process group. Do not try to defeat either.
5. **Commit message format:** `[self-improve <N>] <one-line summary> | score=<float> | sig=<short>`.
6. **No backwards-compat shims, no broad refactors.** One focused hypothesis per iteration. Tests must pass after each commit (`make test`).
7. **Protected files — never edit:** `scripts/eval_signed.py`, `program.md`, `entrypoint.sh`, `Dockerfile`, anything under `.github/`. The review workflow hard-fails the branch if these change.

## Each iteration

```
1. Read CLAUDE.md, last 5 commits, current improvement_log.tsv.
2. Form one hypothesis ("changing X improves the metric because Y").
3. Implement the smallest code change that tests the hypothesis.
4. Run: make test                              — must pass
5. Run: make train_pixel STEPS=2000 ROM=/data/PokemonCoral.gbc \
            INIT_STATE=/data/coral_init.state  — produce a candidate bundle
        (make pretrain_dino DINO_STEPS=... for encoder-track hypotheses)
6. Run: python scripts/eval_signed.py \
        --baseline /baseline \
        --treatment <your candidate run_dir> \
        --cartridge "POKEMON CORAL" \
        --rom /data/PokemonCoral.gbc \
        --init-state /data/coral_init.state \
        --episodes 5 \
        --log improvement_log.tsv
7. Read the score (eval also writes a machine-readable eval_report.json
   next to the HTML). If better than the running best: commit. If worse:
   revert the change, log the negative result in improvement_log.tsv
   (still signed), do NOT commit code.
8. Append your iteration number to the commit subject; the wallclock
   watchdog is the hard cap.
```

## When you finish

`git push origin HEAD` if you want a mid-loop checkpoint; the entrypoint
pushes the branch on exit regardless.
Print a final summary to stdout: best score, baseline delta, top 3 commits.

## What you should NOT do

- Do not run `git push --force` anywhere.
- Do not delete `improvement_log.tsv` rows.
- Do not edit the protected files (rule 7).
- Do not network-call anywhere except `api.anthropic.com`, `github.com`, GCS, ghcr.io. The container may have egress rules; treat unexpected DNS failures as intentional.

Go.
