# Python Third-Party Notices

This project depends on the following pinned Python packages (from `requirements.txt`).

> These license labels are the typical upstream licenses. The release process should verify exact metadata in the target environment using `Licenses/Python/generate_notices.py`.

| Package | Version | Typical License |
|---|---:|---|
| numpy | 2.3.5 | BSD-3-Clause |
| scipy | 1.15.0 | BSD-3-Clause |
| scikit-learn | 1.5.2 | BSD-3-Clause |
| matplotlib | 3.10.8 | Matplotlib License (PSF-style/BSD-compatible) |
| tensorly | 0.9.0 | BSD-3-Clause |
| pylatex | 1.4.2 | MIT |
| pandas | 2.3.3 | BSD-3-Clause |

## Verification command

Run this in the project root inside the release environment:

`python Licenses/Python/generate_notices.py`

This updates `Licenses/Python/THIRD-PARTY-NOTICES.md` with installed metadata (`License` and license classifiers).
