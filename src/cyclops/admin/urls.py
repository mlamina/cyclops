"""The page, the numbers behind it, the volume, the way back to the camera - and the diagrams.

The diagram routes are on this service rather than a second one because the panel already has a
browser pointed here and kept warm; giving it somewhere else to be would mean starting a second
one, and starting one is the thing the whole design avoids (see ``kiosk.py:110-113``).
"""

from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/status", views.status, name="status"),
    path("api/panel", views.panel, name="panel"),
    path("api/diagram/<str:ident>", views.diagram, name="diagram"),
    path("diagram/shown", views.diagram_shown, name="diagram-shown"),
    path("static/<str:name>", views.static_file, name="static"),
    path("volume", views.set_volume, name="volume"),
    path("close", views.close_browser, name="close"),
]
