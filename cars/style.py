import matplotlib as mpl
from matplotlib import font_manager as fm
import glob
for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
    fm.fontManager.addfont(f)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#a8a7a2"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"
mpl.rcParams.update({
    "font.family": "TeX Gyre Termes", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": "#8a8984",
    "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "mathtext.fontset": "stix",
})
