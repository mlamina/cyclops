"""Django settings for the admin page - no database, no apps, one template, one middleware.

Nothing here reads or writes a model, so ``INSTALLED_APPS`` and ``DATABASES`` are both empty and
Django's checks pass anyway. That keeps the service a single process with nothing to migrate,
nothing to back up, and nothing to go stale on the card.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Nothing is signed, sessioned or authenticated here; Django simply insists on a key existing.
SECRET_KEY = os.environ.get("CYCLOPS_ADMIN_SECRET_KEY", "cyclops-admin-nothing-here-is-signed")
DEBUG = False
# Reached by IP, by .local name, and from the kiosk on loopback. There is no host-based routing
# to protect, and on a private LAN nothing to protect it from.
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS: list[str] = []
DATABASES: dict[str, dict] = {}

# CommonMiddleware only. No CSRF: the single mutating endpoint refuses anything that is not
# loopback, which is a stronger guarantee than a token on a page that has no login to steal.
MIDDLEWARE = ["django.middleware.common.CommonMiddleware"]

ROOT_URLCONF = "cyclops.admin.urls"
WSGI_APPLICATION = "cyclops.admin.wsgi.application"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": False,  # there are no apps to look inside; templates come from DIRS
        "OPTIONS": {"context_processors": []},
    }
]

USE_I18N = False  # nothing is translated, so skip loading the translation machinery
USE_TZ = True

# Django's default logging routes request errors to mail_admins, which is configured nowhere -
# so with DEBUG=False a broken view would 500 in complete silence. Put tracebacks on stderr,
# where journald keeps them for ``journalctl -u cyclops-admin``.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"stderr": {"class": "logging.StreamHandler"}},
    "loggers": {"django.request": {"handlers": ["stderr"], "level": "ERROR"}},
}
