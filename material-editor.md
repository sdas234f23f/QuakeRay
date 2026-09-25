# Material editor

`qr_light_editor_start` turns the view over to a free camera; the player stands where he stood.
Aim with the crosshair — a face under it is picked out by an outline — and fire to select it and
open the material panel on the right edge of the screen. The panel edits the `materials.yaml`
parameters of the picked texture, every animation frame of it at once (medkits, blinking buttons),
and the change is on screen the same frame: the material is re-synthesized and the traced world —
which bakes a material's texture indices when it is uploaded — is asked to re-upload itself, lights
included.

* `WASD` + mouse: fly; `Shift`: faster; jump / movedown: up / down; `~`: console; `Esc`: exit.
* `Apply` — writes all materials back to the `materials/*.yaml` files they were loaded from
  (newly created materials go to `materials/materials.yaml`).
* `Cancel` — reverts the live materials to the values the files hold.
* `Exit` (or `qr_light_editor_stop`) — closes the editor, restores the player's view, discards
  whatever was not applied.
* The panel owns the mouse while open: sliders with editable values, checkboxes, hex colour
  fields with an inline colour editor, texture path fields with a file dialog, and a scrollbar.
* Written files win only over the loose files on disk: a `materials.yaml` packed into a mounted
  `.pkz` is read in preference to the written one, so keep the runtime materials loose while
  editing.
