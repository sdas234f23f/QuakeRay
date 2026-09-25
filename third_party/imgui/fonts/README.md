# GUI font

`Roboto-Regular.ttf` is the GUI font of the qr light editor. It is loaded at
startup from `<exe dir>/fonts/Roboto-Regular.ttf` (copied there by the build)
and falls back to the built-in ImGui font when it is missing.

Source: [google/fonts, `ofl/roboto/Roboto[wdth,wght].ttf`](https://github.com/google/fonts/tree/main/ofl/roboto).
It is a variable font; stb_truetype (embedded in Dear ImGui) renders its
default instance, which is Regular (weight 400, normal width).

Licensed under the SIL Open Font License 1.1, see `LICENSE-Roboto.txt`.
