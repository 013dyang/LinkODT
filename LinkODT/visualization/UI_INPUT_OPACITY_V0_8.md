# Visualization v0.8 — numeric inputs and opacity

- All QSpinBox/QDoubleSpinBox controls use direct text entry only; up/down arrow buttons are removed.
- Opacity range is 0.00–1.00.
- Opacity is applied immediately to the selected actor.
- Translucent actors are explicitly routed through VTK's translucent rendering pass, including point clouds and intensity-mapped data.
