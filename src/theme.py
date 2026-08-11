"""Shared UI color and font constants for tkinter GUIs.

Used by annotate.py and compare_val.py to keep the visual theme consistent.

Usage:
    from src.theme import ACCENT, BG_COLOR, SIDEBAR_BG, FONT_LABEL
"""

# ── Colors ───────────────────────────────────────────────────────────
BOX_COLOR       = "#10B981"     # Emerald Green
BOX_COLOR_HOVER = "#EF4444"     # Red
ACTIVE_BOX_COLOR = "#F59E0B"    # Amber
BG_COLOR        = "#F8FAFC"     # Dashboard Light Background
SIDEBAR_BG      = "#FFFFFF"     # Surface Card Background
ACCENT          = "#3B82F6"     # Primary Blue
TEXT_COLOR       = "#1E293B"     # Dark Slate Header
TEXT_MUTED      = "#64748B"     # Muted text
PROGRESS_DONE   = "#10B981"     
PROGRESS_TODO   = "#E2E8F0"     
HUMAN_COLOR     = "#10B981"     
MODEL_COLOR     = "#EF4444"     

# ── Fonts ────────────────────────────────────────────────────────────
FONT_FAMILY     = "Roboto"
FONT_MAIN       = (FONT_FAMILY, 10)
FONT_HEADER     = (FONT_FAMILY, 14, "bold")
FONT_LABEL      = (FONT_FAMILY, 11, "bold")

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
