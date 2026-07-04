# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules
from pathlib import Path
import sys

block_cipher = None

tensorly_imports = collect_submodules('tensorly')
pymcr_imports = collect_submodules('pymcr')
sklearn_imports = collect_submodules('sklearn')
scipy_imports = collect_submodules('scipy')
numpy_imports = collect_submodules('numpy')
pandas_imports = collect_submodules('pandas')
matplotlib_imports = collect_submodules('matplotlib')
PIL_imports = collect_submodules('PIL')
pylatex_imports = collect_submodules('pylatex')
ddsimca_imports = collect_submodules('ddsimca')
prcv_imports = collect_submodules('prcv')
sv_ttk_datas, sv_ttk_binaries, sv_ttk_imports = collect_all('sv_ttk')


def include_if_exists(path, target='.'):
    p = Path(path)
    return [(str(p), target)] if p.exists() else []


def build_icon_path():
    if sys.platform == 'win32':
        p = Path('Graphics/Icon.ico')
        return str(p) if p.exists() else None
    if sys.platform == 'darwin':
        p = Path('Graphics/Icon.icns')
        return str(p) if p.exists() else None
    p = Path('Graphics/Icon.png')
    return str(p) if p.exists() else None


def build_user_icon_path():
    if sys.platform == 'win32':
        p = Path('Graphics/icon-user.ico')
        return str(p) if p.exists() else None
    if sys.platform == 'darwin':
        p = Path('Graphics/icon-user.icns')
        return str(p) if p.exists() else None
    p = Path('Graphics/icon-user.png')
    return str(p) if p.exists() else None


app_icon = build_icon_path()
user_app_icon = build_user_icon_path()

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=sv_ttk_binaries,
    datas=(
        include_if_exists('about_us.json')
        + include_if_exists('acknowledgements.json')
        + include_if_exists('function_specs.json')
        # model.json is not bundled; it is generated at runtime in a user-writable location
        + include_if_exists('pyproject.toml')
        + include_if_exists('EULA.md')
        + include_if_exists('LICENSE')
        + include_if_exists('NOTICE')
        + include_if_exists('Fonts', 'Fonts')
        + include_if_exists('Graphics', 'Graphics')
        + include_if_exists('gui_configs', 'gui_configs')
        + include_if_exists('languages', 'languages')
        + include_if_exists('Settings', 'Settings')
        + include_if_exists('Manual', 'Manual')
        + include_if_exists('Licenses', 'Licenses')
        + include_if_exists('chemometrics', 'chemometrics')
        + include_if_exists('themes', 'themes')
        + include_if_exists('Add-ons', 'Add-ons')
        + sv_ttk_datas
    ),
    hiddenimports=[
        'PIL._tkinter_finder',
        *tensorly_imports,
        *pymcr_imports,
        *sklearn_imports,
        *scipy_imports,
        *numpy_imports,
        *pandas_imports,
        *matplotlib_imports,
        *PIL_imports,
        *pylatex_imports,
        *ddsimca_imports,
        *prcv_imports,
        *sv_ttk_imports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a_user = Analysis(
    ['user.py'],
    pathex=[],
    binaries=sv_ttk_binaries,
    datas=(
        include_if_exists('about_us.json')
        + include_if_exists('acknowledgements.json')
        + include_if_exists('function_specs.json')
        # model.json is not bundled; it is generated at runtime in a user-writable location
        + include_if_exists('pyproject.toml')
        + include_if_exists('EULA.md')
        + include_if_exists('LICENSE')
        + include_if_exists('NOTICE')
        + include_if_exists('Fonts', 'Fonts')
        + include_if_exists('Graphics', 'Graphics')
        + include_if_exists('gui_configs', 'gui_configs')
        + include_if_exists('languages', 'languages')
        + include_if_exists('Settings', 'Settings')
        + include_if_exists('Manual', 'Manual')
        + include_if_exists('Licenses', 'Licenses')
        + include_if_exists('chemometrics', 'chemometrics')
        + include_if_exists('themes', 'themes')
        + include_if_exists('Add-ons', 'Add-ons')
        + sv_ttk_datas
    ),
    hiddenimports=[
        'PIL._tkinter_finder',
        *tensorly_imports,
        *pymcr_imports,
        *sklearn_imports,
        *scipy_imports,
        *numpy_imports,
        *pandas_imports,
        *matplotlib_imports,
        *PIL_imports,
        *pylatex_imports,
        *ddsimca_imports,
        *prcv_imports,
        *sv_ttk_imports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
pyz_user = PYZ(a_user.pure, a_user.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ChemometricStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=app_icon,
    contents_directory='.',
)

user_exe = EXE(
    pyz_user,
    a_user.scripts,
    [],
    exclude_binaries=True,
    name='ChemometricStudioUser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=user_app_icon,
    contents_directory='.',
)

coll = COLLECT(
    exe,
    user_exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ChemometricStudio',
)
