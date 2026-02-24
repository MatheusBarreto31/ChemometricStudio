# Third-Party Licenses

This folder contains third-party licensing material that should ship with redistributions of Chemometric Studio.

## Why this exists

The project code can be licensed under GPL, while bundled third-party components keep their own licenses.
This folder centralizes those notices.

## Contents

- `Fonts/Selawik/NOTICE.txt`: Notice for Microsoft Selawik font distribution under SIL OFL 1.1.
- `Fonts/Selawik/OFL-1.1.txt`: Bundled SIL Open Font License text.
- `Python/THIRD-PARTY-NOTICES.md`: Third-party Python dependency notice list for pinned dependencies.
- `Python/generate_notices.py`: Helper script to regenerate/update dependency notices from installed metadata.

## Release checklist

1. Ensure any bundled font has its own license notice included here.
2. Run `python Licenses/Python/generate_notices.py` in the release environment.
3. Review `Licenses/Python/THIRD-PARTY-NOTICES.md` for missing/unknown entries.
4. Include this entire `Licenses` folder in source and binary distributions.

## Important note

Bundled splash/about font assets currently come from:
- `Fonts/Selawik/selawk.ttf`
- `Fonts/Selawik/selawksb.ttf`

The corresponding license text provided with the font is:
- `Licenses/Fonts/Selawik/OFL-1.1.txt`

For GPL-friendly distribution, keep that OFL license text and the Selawik notice in this folder.
