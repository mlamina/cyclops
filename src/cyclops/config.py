"""Settings loaded from the environment (and a local ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = "gpt-realtime-2.1"  # newest GA speech-to-speech model (Jul 2026)
    voice: str = "marin"
    volume: float = 1.0  # output gain 0.0-1.0 (CYCLOPS_VOLUME is a percent 0-100)
    reasoning_effort: str | None = None  # None: "low" on reasoning models, omitted otherwise
    camera_index: int | None = None  # None: probe cameras and remember the one that works
    half_duplex: bool | None = None  # None: auto-detect from the output device
    barge_in_db: float | None = 8.0  # how much louder than the echo you must be; None = off
    transcribe_lang: str | None = "en"  # ISO-639-1 hint for the input transcriber; None = auto
    input_device: str | None = None  # sounddevice mic: index or name substring; None = default
    output_device: str | None = None  # sounddevice speaker: index or name substring; None = default
    captures_dir: Path = Path("captures")


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _tristate(value: str | None) -> bool | None:
    """'1/true/yes/on' -> True, '0/false/no/off' -> False, unset/'auto' -> None."""
    normalized = (value or "").lower()
    if normalized in {"", "auto"}:
        return None
    return normalized in {"1", "true", "yes", "on"}


def _int(name: str) -> int | None:
    raw = _env(name)
    if raw is None or raw.lower() == "auto":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer or 'auto', got {raw!r}") from exc


def _barge_in_db(name: str) -> float | None:
    raw = _env(name)
    if raw is None:
        return Settings.barge_in_db
    if raw.lower() in {"off", "0", "false", "no"}:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number of dB or 'off', got {raw!r}") from exc


def _volume(name: str) -> float:
    raw = _env(name)
    if raw is None:
        return Settings.volume
    try:
        percent = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number 0-100, got {raw!r}") from exc
    return max(0.0, min(1.0, percent / 100.0))


def _lang(name: str) -> str | None:
    raw = _env(name)
    if raw is None:
        return Settings.transcribe_lang
    return None if raw.lower() == "auto" else raw.lower()


def load_settings() -> Settings:
    """Load the nearest ``.env`` (searching upward from the CWD) and build Settings."""
    if dotenv_path := find_dotenv(usecwd=True):
        load_dotenv(dotenv_path)

    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Put it in a .env file next to pyproject.toml "
            "(see .env.example) or export it in your shell."
        )
    return Settings(
        api_key=api_key,
        model=_env("CYCLOPS_MODEL") or Settings.model,
        voice=_env("CYCLOPS_VOICE") or Settings.voice,
        volume=_volume("CYCLOPS_VOLUME"),
        reasoning_effort=_env("CYCLOPS_REASONING_EFFORT"),
        camera_index=_int("CYCLOPS_CAMERA_INDEX"),
        half_duplex=_tristate(_env("CYCLOPS_HALF_DUPLEX")),
        barge_in_db=_barge_in_db("CYCLOPS_BARGE_IN_DB"),
        transcribe_lang=_lang("CYCLOPS_LANG"),
        input_device=_env("CYCLOPS_INPUT_DEVICE"),
        output_device=_env("CYCLOPS_OUTPUT_DEVICE"),
        captures_dir=Path(_env("CYCLOPS_CAPTURES_DIR") or Settings.captures_dir).expanduser(),
    )
