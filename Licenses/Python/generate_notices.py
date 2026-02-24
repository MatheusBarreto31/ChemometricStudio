from __future__ import annotations

import importlib.metadata as md
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
REQ_FILE = ROOT / "requirements.txt"
OUT_FILE = ROOT / "Licenses" / "Python" / "THIRD-PARTY-NOTICES.md"


def _parse_requirements(path: Path) -> List[Tuple[str, str]]:
    deps: List[Tuple[str, str]] = []
    if not path.exists():
        return deps

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, version = line.split("==", 1)
            deps.append((name.strip(), version.strip()))
        else:
            deps.append((line, ""))
    return deps


def _extract_license_info(pkg_name: str) -> Dict[str, str]:
    result = {
        "license": "UNKNOWN",
        "classifiers": "",
        "home_page": "",
    }
    try:
        meta = md.metadata(pkg_name)
    except md.PackageNotFoundError:
        result["license"] = "NOT INSTALLED IN CURRENT ENV"
        return result

    lic = (meta.get("License") or "").strip()
    if lic:
        result["license"] = lic

    classifiers = [
        c.replace("License ::", "").strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    if classifiers:
        result["classifiers"] = " | ".join(classifiers)

    result["home_page"] = (meta.get("Home-page") or meta.get("Project-URL") or "").strip()
    return result


def build_notice() -> str:
    deps = _parse_requirements(REQ_FILE)

    lines: List[str] = []
    lines.append("# Python Third-Party Notices")
    lines.append("")
    lines.append("This file is generated from `requirements.txt` and installed package metadata.")
    lines.append("")
    lines.append("| Package | Pinned Version | Installed Metadata License | License Classifiers | Home Page |")
    lines.append("|---|---:|---|---|---|")

    for pkg, pinned_version in deps:
        info = _extract_license_info(pkg)
        lic = info["license"].replace("|", "\\|")
        classifiers = info["classifiers"].replace("|", "\\|")
        home_page = info["home_page"].replace("|", "\\|")
        lines.append(f"| {pkg} | {pinned_version or '-'} | {lic or '-'} | {classifiers or '-'} | {home_page or '-'} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Review any `UNKNOWN` or `NOT INSTALLED IN CURRENT ENV` entries before release.")
    lines.append("- For binary redistribution, include upstream license files as required by each dependency.")

    return "\n".join(lines) + "\n"


def main() -> None:
    content = build_notice()
    OUT_FILE.write_text(content, encoding="utf-8")
    print(f"Updated: {OUT_FILE}")


if __name__ == "__main__":
    main()
