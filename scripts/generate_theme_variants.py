"""
Generate Sun Valley colour-variant themes for Chemometric Studio.

For each variant defined in VARIANTS:
  - Recolours the base sv_ttk spritesheet (light or dark) via HSV hue-shift.
  - Writes the recoloured PNG to  themes/sun-valley-variants/<id>/spritesheet.png
  - Writes a self-contained TCL theme file to themes/sun-valley-variants/<id>/theme.tcl

Run from the project root:
    python scripts/generate_theme_variants.py
"""

import importlib
import re
import sys
import warnings
from pathlib import Path

import numpy as np
np.seterr(divide="ignore", invalid="ignore")
from PIL import Image

# ---------------------------------------------------------------------------
# Locate the sv_ttk package
# ---------------------------------------------------------------------------
try:
    import sv_ttk
    SV_TTK_DIR = Path(sv_ttk.__file__).parent
except ImportError:
    print("sv_ttk not found – make sure the virtual-environment is active.")
    sys.exit(1)

THEME_DIR = SV_TTK_DIR / "theme"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "themes" / "sun-valley-variants"


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
#   base          : "light" or "dark"  – which sv_ttk spritesheet to recolour
#   src_hue_center: hue (degrees) of the accent pixels in the *source* sheet
#   src_hue_half  : half-width of the hue band to recolour (degrees)
#   tgt_hue       : desired hue for the accent pixels (degrees)
#   sat_factor    : scale applied to saturation of accent pixels
#   val_factor    : scale applied to value of accent pixels
#   colors        : replacement values for the TCL `colors` array
#   treeview_sel  : Treeview selected-row colours (bg, fg)
#   sash_color    : Panedwindow sash colour
# ---------------------------------------------------------------------------
VARIANTS = {
    "cobalt": {
        "label": "Cobalt",
        "base": "dark",
        "tcl_theme_name": "sv-cobalt",
        "src_hue_center": 200,
        "src_hue_half": 30,
        "src_sat_min": 0.30,
        "tgt_hue": 188,
        "sat_factor": 1.30,
        "val_factor": 0.85,
        "colors": {
            "-fg":    "#dcdfe4",
            "-bg":    "#1c1e26",
            "-disfg": "#5a5f72",
            "-selfg": "#ffffff",
            "-selbg": "#00b4cc",
            "-accent": "#00b4cc",
        },
        "treeview_bg":  "#1c1e26",
        "treeview_sel": ("#1c3c42", "#dcdfe4"),
        "sash_color": "#3a3f52",
    },
    "light-blue": {
        "label": "Light Blue",
        "base": "light",
        "tcl_theme_name": "sv-light-blue",
        "src_hue_center": 210,
        "src_hue_half": 25,
        "src_sat_min": 0.35,
        "tgt_hue": 213,
        "sat_factor": 0.95,
        "val_factor": 1.05,
        "colors": {
            "-fg":    "#10263d",
            "-bg":    "#d8e7f5",
            "-disfg": "#7a9ab8",
            "-selfg": "#ffffff",
            "-selbg": "#007aff",
            "-accent": "#1f5f90",
        },
        "treeview_bg":  "#edf4fb",
        "treeview_sel": ("#b7d0e4", "#10263d"),
        "sash_color": "#8ab4d8",
    },
    "grey": {
        "label": "Grey",
        "base": "light",
        "tcl_theme_name": "sv-grey",
        "src_hue_center": 210,
        "src_hue_half": 25,
        "src_sat_min": 0.35,
        "tgt_hue": 210,
        "sat_factor": 0.56,
        "val_factor": 1.00,
        "colors": {
            "-fg":    "#000000",
            "-bg":    "#f0f0f0",
            "-disfg": "#7a7a7a",
            "-selfg": "#ffffff",
            "-selbg": "#0078d7",
            "-accent": "#5a95c9",
        },
        "treeview_bg":  "#ffffff",
        "treeview_sel": ("#0078d7", "#ffffff"),
        "sash_color": "#c0c0c0",
    },
}


# ---------------------------------------------------------------------------
# Spritesheet recolouring
# ---------------------------------------------------------------------------

def _rgb_to_hsv_vec(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    """Vectorised RGB→HSV (inputs 0-1)."""
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    span = maxc - minc
    s = np.where(maxc != 0, span / maxc, 0.0)

    rc = np.where(span != 0, (maxc - r) / span, 0.0)
    gc = np.where(span != 0, (maxc - g) / span, 0.0)
    bc = np.where(span != 0, (maxc - b) / span, 0.0)

    h = np.where(r == maxc, bc - gc,
         np.where(g == maxc, 2.0 + rc - bc,
                              4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = np.where(span == 0, 0.0, h)
    return h, s, v


def _hsv_to_rgb_vec(h: np.ndarray, s: np.ndarray, v: np.ndarray):
    """Vectorised HSV→RGB (outputs 0-1)."""
    i = (h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i6 = i % 6
    r = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [v, q, p, p, t, v])
    g = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [t, v, v, q, p, p])
    b = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4, i6 == 5], [p, p, t, v, v, q])
    return r, g, b


def recolor_spritesheet(
    src_path: Path,
    dst_path: Path,
    src_hue_center_deg: float,
    src_hue_half_deg: float,
    src_sat_min: float,
    tgt_hue_deg: float,
    sat_factor: float,
    val_factor: float,
) -> None:
    img = Image.open(src_path).convert("RGBA")
    arr = np.array(img, dtype=np.float32)

    r = arr[..., 0] / 255.0
    g = arr[..., 1] / 255.0
    b = arr[..., 2] / 255.0
    a = arr[..., 3]

    h, s, v = _rgb_to_hsv_vec(r, g, b)

    src_lo = ((src_hue_center_deg - src_hue_half_deg) % 360) / 360.0
    src_hi = ((src_hue_center_deg + src_hue_half_deg) % 360) / 360.0
    tgt_h = tgt_hue_deg / 360.0
    src_c = src_hue_center_deg / 360.0

    # Mask: visible pixels whose hue falls in the accent band AND are saturated
    if src_lo < src_hi:
        in_band = (h >= src_lo) & (h <= src_hi)
    else:  # wraps 360°→0°
        in_band = (h >= src_lo) | (h <= src_hi)
    mask = in_band & (s >= src_sat_min) & (a > 10)

    # Shift hue while preserving the *relative* position within the band
    delta = h - src_c
    new_h = np.where(mask, (tgt_h + delta) % 1.0, h)
    new_s = np.where(mask, np.clip(s * sat_factor, 0.0, 1.0), s)
    new_v = np.where(mask, np.clip(v * val_factor, 0.0, 1.0), v)

    nr, ng, nb = _hsv_to_rgb_vec(new_h, new_s, new_v)

    out = np.stack(
        [
            np.clip(nr * 255, 0, 255).astype(np.uint8),
            np.clip(ng * 255, 0, 255).astype(np.uint8),
            np.clip(nb * 255, 0, 255).astype(np.uint8),
            a.astype(np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(out, "RGBA").save(dst_path, "PNG")
    px_changed = int(mask.sum())
    print(f"  recoloured {px_changed} accent pixels → {dst_path.name}")


# ---------------------------------------------------------------------------
# TCL theme file generation
# ---------------------------------------------------------------------------

# Load the original TCL source texts once
def _read_tcl(name: str) -> str:
    return (THEME_DIR / name).read_text(encoding="utf-8")


_LIGHT_TCL = _read_tcl("light.tcl")
_DARK_TCL = _read_tcl("dark.tcl")
_SPRITES_LIGHT_TCL = _read_tcl("sprites_light.tcl")


def _build_colors_block(colors: dict) -> str:
    lines = ["  array set colors {"]
    for k, v in colors.items():
        lines.append(f'    {k}      "{v}"')
    lines.append("  }")
    return "\n".join(lines)


def _patch_tcl(
    base_tcl: str,
    base_name: str,       # "sv_light" or "sv_dark"
    new_ns: str,          # e.g. "sv_arduino"
    theme_create: str,    # e.g. "sv-arduino"
    spritesheet_var: str, # unique Tk image name for the temp sheet
    colors: dict,
    treeview_bg: str,
    treeview_sel: tuple,
    sash_color: str,
    sprites_tcl_abs_path: str,
) -> str:
    tcl = base_tcl

    # 1. Replace the `source` directive so it points to the installed sprites file
    tcl = re.sub(
        r"source \[file join \[file dirname \[info script\]\] sprites_\w+\.tcl\]",
        f'source "{sprites_tcl_abs_path}"',
        tcl,
    )

    # 2. Replace namespace
    tcl = tcl.replace(f"ttk::theme::{base_name}", f"ttk::theme::{new_ns}")

    # 3. Replace package provide name
    tcl = re.sub(
        rf"package provide ttk::theme::{re.escape(new_ns)} \S+",
        f"package provide ttk::theme::{new_ns} 1.0",
        tcl,
    )

    # 4. Replace theme create name ("sun-valley-light" / "sun-valley-dark")
    old_create = "sun-valley-light" if base_name == "sv_light" else "sun-valley-dark"
    tcl = tcl.replace(
        f"ttk::style theme create {old_create}",
        f"ttk::style theme create {theme_create}",
    )

    # 5. Replace the temporary spritesheet image name so multiple variants
    #    don't collide on the global Tk image namespace.
    tcl = tcl.replace(
        "image create photo spritesheet ",
        f"image create photo {spritesheet_var} ",
    )
    # Also fix the copy reference and cleanup
    tcl = tcl.replace(
        f"$I($name) copy spritesheet",
        f"$I($name) copy {spritesheet_var}",
    )

    # 6. Replace colours array
    old_colors_re = re.compile(
        r"array set colors \{[^}]+\}", re.DOTALL
    )
    new_colors = _build_colors_block(colors)
    tcl = old_colors_re.sub(new_colors, tcl, count=1)

    # 6b. Explicitly set the top-level palette so Tk widgets and container
    # backgrounds don't fall back to the stock gray theme.
    tcl = tcl.replace(
        """    # Button
""",
        """    ttk::style configure . -background $colors(-bg) -foreground $colors(-fg)
    ttk::style map . -background [list disabled $colors(-bg) active $colors(-bg)] -foreground [list disabled $colors(-disfg) active $colors(-fg)]
    ttk::style configure TFrame -background $colors(-bg)
    ttk::style configure TLabel -background $colors(-bg) -foreground $colors(-fg)
    ttk::style configure TLabelframe -background $colors(-bg) -foreground $colors(-fg)
    ttk::style configure TNotebook -background $colors(-bg)

    # Button
""",
    )

    # 7. Replace treeview selected colours
    sel_bg, sel_fg = treeview_sel
    tcl = re.sub(
        r'ttk::style map Treeview -background \{selected "[^"]*"\} -foreground \{selected "[^"]*"\}',
        f'ttk::style map Treeview -background {{selected "{sel_bg}"}} -foreground {{selected "{sel_fg}"}}',
        tcl,
    )
    # dark variant uses a slightly different pattern:
    tcl = re.sub(
        r'ttk::style map Treeview -background \{selected "[^"]*"\} -foreground "selected [^"]*"',
        f'ttk::style map Treeview -background {{selected "{sel_bg}"}} -foreground {{selected "{sel_fg}"}}',
        tcl,
    )
    # treeview background
    tcl = re.sub(
        r'ttk::style configure Treeview\s*\\\s*\n(\s*-background) "[^"]*"',
        lambda m: m.group(0).replace(
            m.group(0).split('"')[1], treeview_bg
        ),
        tcl,
    )

    # 8. Sash colour
    tcl = re.sub(
        r'-lightcolor "[^"]*"(\s*\\\s*\n\s*-darkcolor) "[^"]*"(\s*\\\s*\n\s*-bordercolor) "[^"]*"',
        f'-lightcolor "{sash_color}"\\1 "{sash_color}"\\2 "{sash_color}"',
        tcl,
    )

    # 9. After loading images, delete the temporary spritesheet to free memory
    insert_after = f"$I($name) copy {spritesheet_var}"
    delete_line = f"\n    image delete {spritesheet_var}"
    # Find the closing brace of the foreach block and insert after it
    idx = tcl.rfind(insert_after)
    if idx != -1:
        # Find next closing brace after the copy line
        brace_idx = tcl.find("\n  }", idx)
        if brace_idx != -1:
            tcl = tcl[:brace_idx] + delete_line + tcl[brace_idx:]

    return tcl


def generate_variant(variant_id: str, cfg: dict) -> None:
    print(f"\n── {cfg['label']} ({variant_id}) ──")
    out = OUT_DIR / variant_id
    out.mkdir(parents=True, exist_ok=True)

    base = cfg["base"]
    src_sheet = THEME_DIR / f"spritesheet_{base}.png"
    dst_sheet = out / "spritesheet.png"

    # --- 1. Generate recoloured spritesheet ---
    recolor_spritesheet(
        src_path=src_sheet,
        dst_path=dst_sheet,
        src_hue_center_deg=cfg["src_hue_center"],
        src_hue_half_deg=cfg["src_hue_half"],
        src_sat_min=cfg["src_sat_min"],
        tgt_hue_deg=cfg["tgt_hue"],
        sat_factor=cfg["sat_factor"],
        val_factor=cfg["val_factor"],
    )

    # --- 2. Generate TCL theme file ---
    base_tcl = _LIGHT_TCL if base == "light" else _DARK_TCL
    base_ns = "sv_light" if base == "light" else "sv_dark"
    sprites_tcl_path = str(THEME_DIR / f"sprites_{base}.tcl").replace("\\", "/")

    tcl_content = _patch_tcl(
        base_tcl=base_tcl,
        base_name=base_ns,
        new_ns=cfg["tcl_theme_name"].replace("-", "_"),
        theme_create=cfg["tcl_theme_name"],
        spritesheet_var=f"sv_sheet_{variant_id.replace('-', '_')}",
        colors=cfg["colors"],
        treeview_bg=cfg["treeview_bg"],
        treeview_sel=cfg["treeview_sel"],
        sash_color=cfg["sash_color"],
        sprites_tcl_abs_path=sprites_tcl_path,
    )

    # Fix the spritesheet path in load_images to use the variant's PNG
    sheet_abs = str(dst_sheet).replace("\\", "/")
    tcl_content = re.sub(
        r"load_images \[file join \[file dirname \[info script\]\] spritesheet_(?:light|dark)\.png\]",
        f'load_images "{sheet_abs}"',
        tcl_content,
    )

    tcl_path = out / "theme.tcl"
    tcl_path.write_text(tcl_content, encoding="utf-8")
    print(f"  TCL theme written  → {tcl_path.name}  (theme: {cfg['tcl_theme_name']})")


def main() -> None:
    print(f"sv_ttk source : {THEME_DIR}")
    print(f"Output        : {OUT_DIR}")
    for vid, cfg in VARIANTS.items():
        generate_variant(vid, cfg)
    print("\nDone. Run the application to see the new themes in Settings → Theme.")


if __name__ == "__main__":
    main()
