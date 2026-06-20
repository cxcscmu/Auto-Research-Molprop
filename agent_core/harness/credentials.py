"""Head-node credential loading.

Order (first match wins):

  B. $MAGENT_ENV_FILE → ./.env → <multi_agent pkg root>/.env → <local root>/.env
  A. ANTHROPIC_API_KEY already exported in os.environ
  → otherwise raise RuntimeError with both remediation paths

No `python-dotenv` dependency — we parse simple KEY=VALUE lines
ourselves. Values may be wrapped in single or double quotes; shell
interpolation (`$FOO`) is intentionally NOT performed so keys with
literal `$` survive verbatim.

Only the laptop supervisor + single-specialist runner need this.
GPU pods never call the SDK, so `run_trial.sh` must not import from
here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import config

_API_KEY_VAR = "ANTHROPIC_API_KEY"


_PKG_ROOT = Path(__file__).resolve().parent.parent   # .../multi_agent/


def _candidate_env_files() -> list[Path]:
    """Ordered list of .env paths to try. Existence is checked by the caller."""
    paths: list[Path] = []
    override = os.environ.get("MAGENT_ENV_FILE")
    if override:
        paths.append(Path(override))
    paths.append(Path.cwd() / ".env")
    # Active task package's .env (e.g. multi_agent_pg/.env). Lazy: only
    # consult adapter if it's already registered, otherwise skip.
    try:
        from agent_core import current_adapter
        paths.append(current_adapter().pkg_root / ".env")
    except RuntimeError:
        pass
    paths.append(_PKG_ROOT / ".env")
    paths.append(config.LOCAL_ROOT / ".env")
    # dedupe while preserving order (cwd == pkg_root is the common dev case)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE file. Comments (#) and blank lines are skipped."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        out[k] = v
    return out


def load_env_file() -> Optional[Path]:
    """Load the first existing candidate .env into os.environ.

    Uses `setdefault` — an already-exported var is NOT overwritten, so
    method A (plain export) takes precedence over a conflicting .env
    entry. Returns the path actually loaded, or None if no candidate
    exists.
    """
    for path in _candidate_env_files():
        if path.is_file():
            for k, v in _parse_env_file(path).items():
                os.environ.setdefault(k, v)
            return path
    return None


def ensure_api_key() -> str:
    """Guarantee ANTHROPIC_API_KEY is set. Raise with remediation text if not.

    Idempotent: safe to call multiple times. The .env scan re-runs each
    call but `setdefault` makes that harmless.
    """
    load_env_file()
    key = os.environ.get(_API_KEY_VAR, "").strip()
    if not key:
        tried = "\n".join(f"    - {p}" for p in _candidate_env_files())
        raise RuntimeError(
            f"{_API_KEY_VAR} is required but not set. Either:\n"
            f"  A. export {_API_KEY_VAR}=<your-key> before launch, OR\n"
            f"  B. write `{_API_KEY_VAR}=<your-key>` into an .env file at\n"
            f"     one of these paths (first match wins):\n{tried}\n"
            f"     Override the search with MAGENT_ENV_FILE=/abs/path/to/.env"
        )
    return key
