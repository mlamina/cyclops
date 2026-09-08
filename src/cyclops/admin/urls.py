"""The page, the numbers behind it, the volume, the way back to the camera - and the panel.

The panel routes are on this service rather than a second one because the panel already has a
browser pointed here and kept warm; giving it somewhere else to be would mean starting a second
one, and starting one is the thing the whole design avoids (see ``kiosk.py:110-113``).
"""

from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/status", views.status, name="status"),
    # The card, read-only and LAN-visible like the four numbers are. Fetched when a view opens
    # rather than polled: none of it changes while you are looking at it, and /api/status is
    # already walking the sessions directory every five seconds on the page's behalf.
    path("api/sessions", views.sessions, name="sessions"),
    path("api/session/<str:name>", views.session, name="session"),
    path("api/session/<str:name>/records", views.session_records, name="session-records"),
    # The button, and the reel that plays what it made. Both answer the LAN as well as the
    # panel - see views.find_clips. The finished clips need no route of their own: they sit in
    # clips/ inside the session folder, which views.MEDIA_DIRS allows, and
    # /media/<name>/clips/1.mp4 is already served by the byte-range file server below.
    path("api/session/<str:name>/clips", views.find_clips, name="find-clips"),
    path("api/videos", views.videos, name="videos"),
    # The conversation as it happens, for a phone or an iPad open beside the bench. The one route
    # here that is polled while somebody watches it, which is why it takes `name` and `since` and
    # usually answers with nothing at all.
    path("api/live", views.live, name="live"),
    path("api/media", views.media_stream, name="media-stream"),
    # <path:> and not <str:>, so "photos/14-33-12_you.jpg" arrives in one piece. What makes that
    # safe is not the converter - it is the two resolutions in views._media_file.
    path("media/<str:name>/<path:relative>", views.media, name="media"),
    # The other half of the card, and the durable half: one folder per project, browsed a
    # directory at a time. Same shape as the session routes above.
    path("api/projects", views.projects, name="projects"),
    path("api/project/<str:name>/files", views.project_files, name="project-files"),
    path("api/project/<str:name>/file", views.project_file, name="project-file"),
    # The gallery, which is the whole folder and not just Photos/ - see shelf.pictures. A route
    # of its own rather than a flag on /files, because it answers a different question: /files
    # says what is in one directory, this says what pictures the project has.
    path("api/project/<str:name>/pictures", views.project_pictures, name="project-pictures"),
    # And the two that write, which are the only routes here that do. Reading a project on a
    # laptop and then having to scp a datasheet onto the Pi is a browser that stops halfway, so
    # these exist and - unlike every other POST below - they answer to the LAN rather than to
    # loopback. They still go through store.py, which remains the only module in the repo that
    # writes into projects/; see the note above them in views.py for what holds the line instead.
    path("api/project/<str:name>/folder", views.project_mkdir, name="project-mkdir"),
    path("api/project/<str:name>/upload", views.project_upload, name="project-upload"),
    path(
        "project-media/<str:name>/<path:relative>",
        views.project_media,
        name="project-media",
    ),
    path("api/panel", views.panel, name="panel"),
    path("api/picture/<str:ident>", views.picture, name="picture"),
    path("panel/painted", views.picture_painted, name="panel-painted"),
    path("static/<str:name>", views.static_file, name="static"),
    path("volume", views.set_volume, name="volume"),
    path("barge-in", views.set_barge_in, name="barge-in"),
    path("record-source", views.set_record_source, name="record-source"),
    path("voice", views.set_voice, name="voice"),
    path("close", views.close_browser, name="close"),
]
