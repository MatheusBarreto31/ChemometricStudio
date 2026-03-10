"""Add-on discovery and function-spec merging for Chemometric Studio."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SPEC_SECTIONS = ("return_specs", "input_specs", "import_map", "gui_listing", "parameter_types")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _is_specs_shape_valid(specs: Dict[str, Any]) -> bool:
    for section in SPEC_SECTIONS:
        if section not in specs or not isinstance(specs[section], dict):
            return False
    return True


def _resolve_addon_config_path(addon_root: Path, raw_config_path: str, language: str) -> Optional[Path]:
    raw = str(raw_config_path or "").strip().replace("\\", "/")
    if not raw:
        return None

    raw_path = Path(raw)
    filename = raw_path.name

    candidate_paths: List[Path] = []
    candidate_paths.append(addon_root / raw_path)
    if raw.startswith("gui_configs/") and filename:
        candidate_paths.append(addon_root / "gui_configs" / language / filename)
        candidate_paths.append(addon_root / "gui_configs" / "en" / filename)

    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def _module_file_exists_for_addon(addon_root: Path, module_name: str) -> bool:
    if not module_name.startswith("chemometrics"):
        return True

    parts = module_name.split(".")
    if not parts:
        return False

    if len(parts) == 1:
        return (addon_root / "chemometrics" / "__init__.py").exists()

    rel_parts = parts[1:]
    module_file = addon_root / "chemometrics" / Path(*rel_parts).with_suffix(".py")
    if module_file.exists():
        return True

    module_pkg_init = addon_root / "chemometrics" / Path(*rel_parts) / "__init__.py"
    return module_pkg_init.exists()


def _collect_required_addon_aliases(addon_specs: Dict[str, Any]) -> Optional[List[str]]:
    gui_aliases = set(addon_specs.get("gui_listing", {}).keys())
    if not gui_aliases:
        return None

    for section in ("return_specs", "input_specs", "import_map", "parameter_types"):
        section_aliases = set(addon_specs.get(section, {}).keys())
        if section_aliases != gui_aliases:
            return None

    return sorted(gui_aliases)


def load_combined_function_specs(base_dir: Path, language: str = "en") -> Dict[str, Any]:
    """Load core + add-on function specs without mutating core files."""
    core_specs_path = base_dir / "function_specs.json"
    core_specs = _load_json(core_specs_path)
    if not core_specs or not _is_specs_shape_valid(core_specs):
        raise RuntimeError(f"Invalid or missing core function specs: {core_specs_path}")

    merged_specs = copy.deepcopy(core_specs)
    addon_registry: Dict[str, Any] = {
        "available_addons": {},
        "function_to_addon": {},
        "warnings": []
    }

    addons_dir = base_dir / "Add-ons"
    if not addons_dir.exists() or not addons_dir.is_dir():
        return {"specs": merged_specs, "addon_registry": addon_registry}

    for addon_root in sorted([p for p in addons_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        addon_id = addon_root.name
        specs_path = addon_root / "function_specs.json"
        gui_root = addon_root / "gui_configs"
        chem_root = addon_root / "chemometrics"

        if not specs_path.exists() or not gui_root.exists() or not chem_root.exists():
            addon_registry["warnings"].append(
                f"Skipped add-on '{addon_id}': expected function_specs.json, gui_configs/, and chemometrics/."
            )
            continue

        addon_specs = _load_json(specs_path)
        if not addon_specs or not _is_specs_shape_valid(addon_specs):
            addon_registry["warnings"].append(f"Skipped add-on '{addon_id}': invalid function_specs.json.")
            continue

        aliases = _collect_required_addon_aliases(addon_specs)
        if not aliases:
            addon_registry["warnings"].append(
                f"Skipped add-on '{addon_id}': inconsistent aliases across spec sections."
            )
            continue

        has_conflict = False
        for alias in aliases:
            if alias in merged_specs["gui_listing"]:
                addon_registry["warnings"].append(
                    f"Skipped add-on '{addon_id}': alias '{alias}' already exists."
                )
                has_conflict = True
                break
        if has_conflict:
            continue

        resolved_gui_listing: Dict[str, Dict[str, str]] = {}
        addon_valid = True

        for alias in aliases:
            gui_meta = addon_specs["gui_listing"].get(alias, {})
            config_path = _resolve_addon_config_path(addon_root, gui_meta.get("config_path", ""), language)
            if not config_path:
                addon_registry["warnings"].append(
                    f"Skipped add-on '{addon_id}': config path for '{alias}' was not found."
                )
                addon_valid = False
                break

            import_info = addon_specs["import_map"].get(alias, [])
            if not isinstance(import_info, list) or len(import_info) != 2:
                addon_registry["warnings"].append(
                    f"Skipped add-on '{addon_id}': invalid import_map entry for '{alias}'."
                )
                addon_valid = False
                break

            module_name = str(import_info[0])
            if not _module_file_exists_for_addon(addon_root, module_name):
                addon_registry["warnings"].append(
                    f"Skipped add-on '{addon_id}': module '{module_name}' for '{alias}' was not found in add-on chemometrics/."
                )
                addon_valid = False
                break

            resolved_gui_listing[alias] = {"config_path": str(config_path)}

        if not addon_valid:
            continue

        for alias in aliases:
            merged_specs["return_specs"][alias] = addon_specs["return_specs"][alias]
            merged_specs["input_specs"][alias] = addon_specs["input_specs"][alias]
            merged_specs["import_map"][alias] = addon_specs["import_map"][alias]
            merged_specs["parameter_types"][alias] = addon_specs["parameter_types"][alias]
            merged_specs["gui_listing"][alias] = resolved_gui_listing[alias]
            addon_registry["function_to_addon"][alias] = addon_id

        addon_registry["available_addons"][addon_id] = {
            "id": addon_id,
            "name": addon_id,
            "root_path": str(addon_root.resolve()),
            "chemometrics_path": str((addon_root / "chemometrics").resolve())
        }

    return {"specs": merged_specs, "addon_registry": addon_registry}


def normalize_required_addons(required_addons: Any) -> List[str]:
    normalized: List[str] = []
    if not isinstance(required_addons, list):
        return normalized

    for item in required_addons:
        if isinstance(item, str):
            value = item.strip()
        elif isinstance(item, dict):
            value = str(item.get("id", "")).strip()
        else:
            value = ""

        if value and value not in normalized:
            normalized.append(value)

    return normalized
