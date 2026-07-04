"""Centralized runtime and settings path resolution for Chemometric Studio."""

import os
import platform
import tempfile
from pathlib import Path

APP_DIR_NAME = "ChemometricStudio"


def get_runtime_root_dir() -> Path:
    """Return writable runtime directory for app-managed transient state."""
    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        return Path(xdg_runtime_dir) / APP_DIR_NAME

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / APP_DIR_NAME

    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME / "Runtime"

    if os.name == "posix":
        return Path.home() / ".local" / "state" / APP_DIR_NAME

    return Path(tempfile.gettempdir()) / APP_DIR_NAME


def get_tempfiles_dir() -> Path:
    """Return runtime tempfiles directory used by packaged workflows."""
    return get_runtime_root_dir() / "tempfiles"


def get_runtime_model_json_path() -> Path:
    """Return model.json runtime location."""
    return get_runtime_root_dir() / "model.json"


def get_runtime_model_log_path() -> Path:
    """Return model execution log runtime location."""
    return get_runtime_root_dir() / "model_log.txt"


def get_settings_dir() -> Path:
    """Return user-writable persistent settings directory."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / APP_DIR_NAME

        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            return Path(localappdata) / APP_DIR_NAME

        return Path.home() / "AppData" / "Roaming" / APP_DIR_NAME

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / APP_DIR_NAME

    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME

    return Path.home() / ".config" / APP_DIR_NAME
