## Documentation Protocol
- **Script Documentation**: Whenever you create or significantly modify an entry-point script in the `scripts/` directory, you MUST proactively update `README.md` to document its usage, and `README-Dev.md` to document any relevant technical breakthroughs, architecture changes, or developer context.
## GUI Design Language (Modern Dashboard Style)
When creating or modifying Tkinter/GUI elements, you MUST strictly adhere to this design language:
- **Core Aesthetic:** Clean, enterprise light-mode dashboard style.
- **Backgrounds:** Use #F8FAFC (soft grayish blue) for application backgrounds.
- **Surfaces/Cards:** Use #FFFFFF for primary surfaces (cards, panels).
- **Typography:** Exclusively use Roboto, Open Sans, or Calibri.
  - Headers: Dark slate #1E293B, bold.
  - Body Text: Muted gray-blue #64748B or #475569.
- **Accents & State Colors:**
  - Primary Action/Brand: Blue (#3B82F6) or Indigo (#6366F1)
  - Success/Active: Emerald Green (#10B981)
  - Warning/Pending: Amber/Yellow (#F59E0B)
  - Danger/Error/Rejected: Red (#EF4444)
- **Icons/Emojis:** Strictly DO NOT use emojis or special unicode icons (e.g., rocket, brain, gears, etc.) in any UI labels, buttons, or menus. Rely on clean typography and layout for hierarchy.
- **Layout:** Use ample padding/margins (flat design). Avoid harsh 3D borders (d=0, elief="flat").
