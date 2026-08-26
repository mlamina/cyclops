"""``cyclops-admin``: the status page, always on, on port 80.

gunicorn is embedded rather than driven from a second command line, so the console script is the
whole story and ``uv run cyclops-admin --port=8080`` on a laptop is the same server that runs on
the Pi. Two workers of two threads: enough that a wedged ``/proc`` read cannot block the page,
small enough to stay out of the way of x264 and the realtime audio thread.
"""

from __future__ import annotations

import os
import sys

WORKERS = 2
THREADS = 2
TIMEOUT_S = 30  # a worker stuck on a kernel read gets recycled rather than staying stuck


def _parse_args(argv: list[str], host: str, port: int) -> tuple[str, int]:
    """``--host=`` / ``--port=`` overrides, in the same argv style as ``cyclops-ui``."""
    for arg in argv:
        if arg.startswith("--host="):
            host = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError:
                raise SystemExit(f"error: bad --port, expected a number, got {arg!r}") from None
    return host, port


def _application(bind: str):
    """gunicorn's programmatic entrypoint, configured in code so there is no second config file."""
    from gunicorn.app.base import BaseApplication

    class CyclopsAdmin(BaseApplication):
        def load_config(self) -> None:
            options = {
                "bind": bind,
                "workers": WORKERS,
                "threads": THREADS,
                "timeout": TIMEOUT_S,
                "accesslog": None,  # per-request lines would be the only writes this makes
                "errorlog": "-",
                "loglevel": "warning",
                "proc_name": "cyclops-admin",
            }
            for key, value in options.items():
                self.cfg.set(key, value)

        def load(self):
            from .wsgi import application

            return application

    return CyclopsAdmin()


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cyclops.admin.settings")

    from ..config import ConfigError, load_settings

    try:
        settings = load_settings(require_api_key=False)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    host, port = _parse_args(sys.argv[1:], settings.admin_host, settings.admin_port)
    # The resolved sessions path is printed because "0 sessions" is otherwise indistinguishable
    # from "started in the wrong directory" - see WorkingDirectory in deploy/cyclops-admin.service.
    print(
        f"· cyclops admin on http://{host}:{port}/  (Ctrl+C to quit)\n"
        f"  sessions: {settings.sessions_dir.expanduser().resolve()}",
        flush=True,
    )
    _application(f"{host}:{port}").run()


if __name__ == "__main__":
    main()
