"""Shared UI color and font constants for tkinter GUIs.

Used by annotate.py and compare_val.py to keep the visual theme consistent.

Usage:
    from src.theme import ACCENT, BG_COLOR, SIDEBAR_BG, FONT_LABEL
"""

# ── Colors ───────────────────────────────────────────────────────────
BOX_COLOR       = "#00AA00"     # Default box outline (green)
BOX_COLOR_HOVER = "#FF0000"     # Hovered box (red)
ACTIVE_BOX_COLOR = "#CCCC00"    # Box being drawn (yellow)
BG_COLOR        = "#FFFFFF"     # Main background
SIDEBAR_BG      = "#F0F0F0"     # Sidebar panel
ACCENT          = "#0000FF"     # Headers, primary buttons
TEXT_COLOR       = "#000000"     # Default text
PROGRESS_DONE   = "#00AA00"     # Progress bar filled
PROGRESS_TODO   = "#CCCCCC"     # Progress bar empty
HUMAN_COLOR     = "#00AA00"     # Human label overlay (green)
MODEL_COLOR     = "#FF0000"     # Model prediction overlay (red)

# Annotation Box Colors (Hex and RGB)
OVERLAP_80_HEX  = "#FF0032"
OVERLAP_80_RGB  = (255, 0, 50)
OVERLAP_50_HEX  = "#FF7800"
OVERLAP_50_RGB  = (255, 120, 0)
OVERLAP_0_HEX   = "#FFE100"
OVERLAP_0_RGB   = (255, 225, 0)
AUTO_BOX_HEX    = "#8B5CF6"
AUTO_BOX_RGB    = (139, 92, 246)
BOX_RGB         = (0, 255, 136)

# ── Fonts ────────────────────────────────────────────────────────────
FONT_LABEL  = ("Segoe UI", 10, "bold")
FONT_INPUT  = ("Consolas", 10)
FONT_TITLE  = ("Segoe UI", 14, "bold")
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)
