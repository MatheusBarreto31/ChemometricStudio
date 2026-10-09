# Third-Party Licensing Documentation

This folder contains third-party licensing material that must be included when redistributing Chemometric Studio.

## Overview

Chemometric Studio is licensed under Apache License 2.0. Bundled third-party components retain their respective original licenses. This directory organizes all required license notices and attributions.

**For the main project license, see:** `../LICENSE` (Apache License 2.0)

## License Files by Category

### Python Dependencies

Located in `Python/`:
- `THIRD-PARTY-NOTICES.md` — Comprehensive notice list for all pinned Python dependencies
- `sv_ttk-LICENSE.md` — MIT license for Sun Valley ttk theme (`sv-ttk`)
- `pyMCR-LICENSE.md` — NIST public-domain notice and disclaimer for pyMCR

### Font Assets

Located in `Fonts/Selawik/`:
- `OFL-1.1.txt` — SIL Open Font License (applies to Selawik font files)
- `NOTICE.txt` — Font distribution notice

Bundled fonts: `selawk.ttf`, `selawksb.ttf`

### Referenced Materials

Located in `References/`:
- `THIRD-PARTY-NOTICES.md` — Attribution list for referenced (non-bundled) open-source projects
- `MVC2_MVC3_NOTICE.md` — Attribution for mvc2/mvc3 MATLAB toolbox methodological references

## Distribution Requirements

When redistributing this application, include:
1. Root-level `LICENSE` file (Apache 2.0)
2. Root-level `EULA.md` (end-user terms)
3. This entire `Licenses/` folder

This ensures compliance with all bundled third-party licenses.
