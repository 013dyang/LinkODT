# Intensity visibility update — v0.9

For `XYZ + Intensity` point clouds only:

- the lowest intensity no longer maps near black;
- the default minimum brightness is 35% of the selected base color;
- a gentle gamma lift (`t^0.72`) improves visibility of low-to-mid intensity shell points;
- the highest intensity still reaches the full selected base color;
- changing Object Color changes the hue while preserving the same brightness mapping;
- RGB instance colors, ordinary point clouds, and OBJ meshes are unchanged.
