# strix — local runtime backend for Satori

Run [Strix](https://github.com/usestrix/strix) (the AI-powered automated
pentesting tool) **inside an unprivileged container** — such as a
[Satori CI](https://satori.ci) execution sandbox — **without Docker-in-Docker**.

## The problem

Out of the box, the `strix` CLI runs on a host and spawns a **sibling Docker
sandbox container** (`ghcr.io/usestrix/strix-sandbox`) to execute its pentest
tools. It hard-requires a Docker daemon:

- it calls `check_docker_installed()` + `pull_docker_image()` at startup, and
- every scan calls `session_manager.create_or_reuse(...)` → the `docker`
  runtime backend (`docker.from_env()`).

Satori's ephemeral containers are **unprivileged** (no `CAP_SYS_ADMIN`, no
overlay mounts, no netfilter) and don't expose the host Docker socket, so a
nested Docker daemon can't start. Strix therefore can't run there unmodified.

## The fix

`strix_local_backend.py` makes Strix run its tools **directly in the current
container** by plugging into Strix's own `register_backend()` extension hook and
the openai-agents SDK's built-in **`unix_local`** sandbox (subprocess-based, no
Docker). It:

1. registers a `local` runtime backend built on
   `agents.sandbox.sandboxes.unix_local.UnixLocalSandboxClient`,
2. monkeypatches `check_docker_installed()` / `pull_docker_image()` to no-ops,
3. stubs the in-container Caido proxy sidecar bootstrap (it isn't running
   locally) and strips the `http(s)_proxy` env Strix injects for it,
4. sets `STRIX_RUNTIME_BACKEND=local` and invokes the normal `strix` CLI
   headless (`-n`).

Because it reuses the SDK's real `unix_local` session, all SDK-internal
machinery (materialization, concurrency limits, workspace handling) is
satisfied — no fragile re-implementation.

## Usage

```bash
pip install strix-agent
export STRIX_LLM="openrouter/google/gemini-2.5-flash"
export LLM_API_KEY="$OPENROUTER"        # OpenRouter key (LiteLLM generic var)
export STRIX_TARGET="https://target.com"
export STRIX_SCAN_MODE="quick"          # quick | standard | deep (optional)
python3 strix_local_backend.py
```

The pentest tools the agent drives (nmap, nuclei, httpx, etc.) must be present
in the container — install the subset you need alongside `strix-agent`.

## Environment variables

| Var | Meaning |
|-----|---------|
| `STRIX_LLM`        | LiteLLM model, e.g. `openrouter/google/gemini-2.5-flash` |
| `LLM_API_KEY`      | Provider key (OpenRouter `sk-or-...`) |
| `STRIX_TARGET`     | URL / domain / IP to assess |
| `STRIX_SCAN_MODE`  | `quick` (default), `standard`, or `deep` |
| `STRIX_EXTRA_ARGS` | Extra raw `strix` flags, space-separated (optional) |

## Satori playbook

`strix.yml` is a ready-to-run Satori playbook that clones this repo and
`strix-agent` into the execution container and applies the fix at run time:

```bash
satori run satori://path/strix.yml \
  -d HOST="https://target.com" \
  -d MODEL="openrouter/google/gemini-2.5-flash" \
  -d OPENROUTER=$OPENROUTER --report --output
```

> ⚠️ Caveat: tools like `agent-browser` (dynamic web analysis) and the Caido
> proxy live only in the upstream Kali sandbox image. In the local backend you
> get the tools you install; browser-driven testing is reduced unless you add
> `agent-browser` yourself.
