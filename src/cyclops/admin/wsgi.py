"""The WSGI entrypoint. WSGI, not ASGI: every view here is synchronous, and Django runs sync
views under ASGI on a single thread-sensitive executor - so ASGI would serialise the very
requests a thread pool handles for free."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cyclops.admin.settings")

application = get_wsgi_application()
