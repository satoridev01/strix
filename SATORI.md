# Satori integration

This fork starts from upstream Strix commit `a3c31236bd6305784a501a3102dab0bda83f2317`
(2026-10-21). The `local` runtime uses the OpenAI Agents SDK's Unix local
sandbox inside an already isolated Satori container. It does not start Docker.

Set `STRIX_RUNTIME_BACKEND=local` before invoking the normal headless CLI.
Provide `STRIX_LLM`, `LLM_API_KEY`, and, for OpenRouter, `OPENROUTER_API_KEY`.
Enable `STRIX_RUN_RECORD_TIMINGS=1` to save model request and tool timings in
`run.json`. It also writes `diagnostics.jsonl` and flushes a redacted
`[strix-diag]` line for each completed model request and tool call to stdout.
These events contain timing, status and token counts, never prompts, command
arguments, tool output or credentials. A typical invocation is:

```sh
STRIX_RUNTIME_BACKEND=local \
STRIX_RUN_RECORD_TIMINGS=1 \
strix -n --target "$HOST" --scan-mode quick --max-budget 5
```

The Satori container must supply Strix's pentest binaries and a writable
workspace. Local mode deliberately does not start Caido, set Caido proxy
variables, or offer the six Caido-only agent tools. Other sandbox tools run
normally. Oversized tool results use upstream Strix's workspace spill mechanism
instead of the old prompt trimming patch. Local mode buffers each model
response until it completes, letting the SDK retry a provider stream that
cuts off partway through. Set `STRIX_ATOMIC_STREAM=0` only when immediate
partial model output is needed more than retry safety.

Read `strix_runs/*/run.json` after every execution. Its `status` must be
`completed` and `scan_results.scan_completed` true to mark a scan complete;
the CLI exit code must also be 0 or 2. Its `finding_count` must match the number
of entries in `vulnerabilities.json`; a missing findings file is clean only when the count is
zero. Preserve `vulnerabilities.json` when present, even if the scan stopped.
Treat a missing run record or `status` of `stopped`, `failed`, or `interrupted`
as incomplete. A budget stop can otherwise look like a clean exit.

Upstream Strix captures OpenRouter's final streamed `usage.cost` for its budget
ledger. This fork pairs provider charges with SDK estimates by response id and
retains estimates for responses lacking a provider charge. Where an id is
missing on one side, upstream's rule still holds: a provider charge replaces
the estimates it cannot be matched to and never adds to them. Missing provider
cost and missing LiteLLM price still produce an unknown cost. Verify a nonzero
cost on a charged smoke run against OpenRouter before relying on `--max-budget`.
A dedicated OpenRouter key with a spend limit provides an independent ceiling.
Keep the exact fork commit, SDK
dependencies, and container image pinned for every production run.
