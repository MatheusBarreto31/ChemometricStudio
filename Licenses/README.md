# Third-Party Licenses

This folder contains third-party licensing material that should ship with redistributions of Chemometric Studio.

## Why this exists

The project code is licensed under Apache License 2.0, while bundled third-party components keep their own licenses.
This folder centralizes those notices.

## Contents

- `../LICENSE`: Apache License 2.0 text for the Chemometric Studio project code.
- `../EULA.md`: End-user license terms for distributed application use.
- `Fonts/Selawik/NOTICE.txt`: Notice for Microsoft Selawik font distribution under SIL OFL 1.1.
- `Fonts/Selawik/OFL-1.1.txt`: Bundled SIL Open Font License text.
- `Python/THIRD-PARTY-NOTICES.md`: Third-party Python dependency notice list for pinned dependencies.
- `Python/sv_ttk-LICENSE.md`: MIT license text and attribution for the bundled Sun Valley ttk theme (`sv-ttk`).
- `References/THIRD-PARTY-NOTICES.md`: Third-party notice list for referenced open-source materials not bundled as direct dependencies.
- `Python/pyMCR-LICENSE.md`: Required NIST public-domain notice/disclaimer text for pyMCR redistribution.
- `References/MVC2_MVC3_NOTICE.md`: Attribution and license text for mvc2/mvc3 MATLAB toolbox materials used as canonical methodological references.

## Important note

Bundled splash/about font assets currently come from:
- `Fonts/Selawik/selawk.ttf`
- `Fonts/Selawik/selawksb.ttf`

The corresponding license text provided with the font is:
- `Licenses/Fonts/Selawik/OFL-1.1.txt`

When redistributing this application, include this `Licenses` folder together with `LICENSE`, `NOTICE`, and `EULA.md`.
