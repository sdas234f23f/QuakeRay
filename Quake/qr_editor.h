// qr_editor.h -- qr light editor: realtime material editor for the vkpt renderer.
//
// The editor is started and stopped from the console with qr_light_editor_start /
// qr_light_editor_stop. While it runs, the view is driven by a free camera (the
// player stands still); aiming at a face and pressing fire opens the material
// panel on the right side of the screen, where the materials.yaml parameters of
// every animation frame of the picked texture can be edited live and saved back
// to materials.yaml (Apply), reverted (Cancel), or the editor closed (Exit).

#ifndef QR_EDITOR_H
#define QR_EDITOR_H

#include "quakedef.h"
#include "glquake.h"

// Console commands.
void QR_Editor_Init (void);

// Editor mode state.
qboolean QR_Editor_Active (void);   // the editor owns the view (flying or panel)
qboolean QR_Editor_PanelOpen (void); // the material panel is on screen
qboolean QR_Editor_Flying (void);   // active, no panel: free camera + crosshair aim

// Picks the face under the crosshair and opens the material panel (fire button).
void QR_Editor_Pick (void);

// Per-frame hooks.
void QR_Editor_UpdateView (void);                  // camera move + r_refdef override (V_CalcRefdef)
void QR_Editor_DrawSelection (cb_context_t *cbx);  // face outlines (R_DrawViewModelTask)
void QR_Editor_DrawPanel (cb_context_t *cbx);      // panel UI (SCR_DrawGUI)

// Input hooks (keys.c / in_sdl.c).
qboolean QR_Editor_KeyEvent (int key, qboolean down); // true = the key was consumed
qboolean QR_Editor_CharEvent (int key);               // true = the char was consumed
qboolean QR_Editor_TextEntryActive (void);            // panel wants SDL text input

#endif /* QR_EDITOR_H */
