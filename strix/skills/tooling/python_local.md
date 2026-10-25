---
name: python_local
description: Run Python through exec_command in the local sandbox. Use direct HTTP clients and save concise evidence.
---

# Python In The Local Sandbox

Use `exec_command` to run Python. There is no separate Python executor.
For reusable probes, write a task-specific `.py` file and run it with
`python3 <name>.py`. For short one-off transformations, use `python3 -c`.
The `shell` parameter selects a POSIX shell; it does not select the
Python interpreter.

Use `curl` or Python HTTP clients for direct requests. Record the URL,
method, status, relevant headers, and a bounded response excerpt for
decisive evidence. Avoid printing whole responses or large payloads into
the agent context. Prefer a file artifact plus a concise summary when
output is large.

There is no Caido service, `caido_api` module, or captured-request replay
in this backend. If a third-party Python package is needed, check that it
is available before importing it. Prefer the standard library for small
scripts.
