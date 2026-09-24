// qr_gui.h -- Dear ImGui bridge for the qr light editor.
//
// The host stays C: this header exposes ImGui as a small function set and the
// panel is built from qr_editor.c. The C++ side (qr_gui.cpp) owns the ImGui
// context, the SDL2 input backend, the style and fonts, and the render
// backend: ImGui draw lists are uploaded through rgUploadRasterizedGeometry
// (SWAPCHAIN render type) with per-draw scissor, so no ImGui Vulkan pipelines
// or descriptor pools exist next to the renderer.

#ifndef QR_GUI_H
#define QR_GUI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Creates the ImGui context, the SDL2 input backend and the font atlas.
// rg_instance is the RgInstance the draw lists are uploaded to; font_path may
// be NULL, the default ImGui font is used then.
void QR_GUI_Init (void *sdl_window, void *rg_instance, const char *font_path);
int  QR_GUI_Ready (void);
void QR_GUI_Shutdown (void);

// One ImGui frame. SCR_UpdateScreen can run more than once per host frame, so
// the frame id guards against a second NewFrame; returns 0 when the id was
// already started. The viewport is the drawable rect in GL convention
// (x, y, width, height) plus the height of the whole drawable (for the Vulkan
// Y flip of the viewport).
int  QR_GUI_BeginFrame (unsigned int frame_id, float dt, int x, int y, int width, int height, int drawable_height);

// Builds the draw data and uploads it to the rasterizer. Must be called in the
// same frame as QR_GUI_BeginFrame.
void QR_GUI_EndFrame (void);

// Feeds one SDL_Event (as const void *) into ImGui, returns 1 if the GUI wants
// the event and the engine must not handle it.
int  QR_GUI_ProcessEvent (const void *sdl_event);

// Returns 1 while the GUI owns the mouse (the panel is open).
int  QR_GUI_WantsMouse (void);
// Returns 1 while a GUI text field has keyboard focus.
int  QR_GUI_WantsKeyboard (void);
// Tells ImGui to draw the mouse cursor itself.
void QR_GUI_SetMouseCursor (int enable);

// ----- the panel -----

void QR_GUI_BeginPanel (const char *id, int x, int y, int width, int height);
void QR_GUI_EndPanel (void);
void QR_GUI_BeginScroll (void);
void QR_GUI_EndScroll (void);

void QR_GUI_Label (const char *text);
void QR_GUI_LabelDim (const char *text);
void QR_GUI_Separator (void);
void QR_GUI_Spacing (void);
void QR_GUI_SameLine (void);
void QR_GUI_Tooltip (const char *text);

int  QR_GUI_Button (const char *label);
int  QR_GUI_Checkbox (const char *label, int *value);
int  QR_GUI_SliderFloat (const char *label, float *value, float min, float max);
int  QR_GUI_SliderInt (const char *label, int *value, int min, int max);
// items are count NUL-terminated strings.
int  QR_GUI_Combo (const char *label, int *value, const char *const *items, int count);
int  QR_GUI_InputText (const char *label, char *buf, size_t capacity);
// A path field with a "..." button: returns 1 when the text changed and 2 when
// the browse button was pressed (both can be set: 3).
int  QR_GUI_TexturePath (const char *label, char *buf, size_t capacity);
// An enabled checkbox and a color editor with a hex field. Returns 1 if either
// changed.
int  QR_GUI_ColorHex (const char *label, float rgb[3], int *enabled);
// Returns nonzero while the section is open.
int  QR_GUI_Section (const char *label, int default_open);

// ID scope for the widgets of one material (animation frames share the same
// parameter names, so their widgets would collide without it).
void QR_GUI_PushID (const char *id);
void QR_GUI_PopID (void);

// 1 while any ImGui item is being dragged or edited.
int  QR_GUI_AnyItemActive (void);

// A short message shown in the corner of the editor interface (Apply/Cancel
// confirmations, errors). Fades out on its own.
void QR_GUI_Notify (const char *text);

// ----- the flying-mode overlay -----

// A small modern crosshair in the centre of the display.
void QR_GUI_DrawCrosshair (void);
// Hint lines in the bottom-left corner, drawn with a soft shadow.
void QR_GUI_DrawHint (const char *const *lines, int count);

#ifdef __cplusplus
}
#endif

#endif /* QR_GUI_H */
