"""Four routes: the page, the numbers behind it, the volume, and the way back to the camera."""

from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/status", views.status, name="status"),
    path("volume", views.set_volume, name="volume"),
    path("close", views.close_browser, name="close"),
]
