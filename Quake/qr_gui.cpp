// qr_gui.cpp -- Dear ImGui bridge for the qr light editor (see qr_gui.h).
//
// The render backend lives here: ImGui draw lists are converted to RgVertex
// arrays and uploaded through rgUploadRasterizedGeometry with the SWAPCHAIN
// render type and a per-draw scissor rect, exactly like the engine's own 2D
// draws. No ImGui Vulkan backend, pipelines or descriptor pools are involved.

#include "qr_gui.h"

#include <imgui.h>
#include <imgui_impl_sdl2.h>

#include <SDL.h>

#include <vkpt/vkpt.h>

#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cfloat>

namespace
{

RgInstance   g_instance      = 0;
RgMaterial   g_font_material = RG_NO_MATERIAL;
bool         g_ready         = false;
bool         g_frame_open    = false;
unsigned int g_last_frame_id = 0xFFFFFFFFu;

int          g_fb_x = 0, g_fb_y = 0, g_fb_w = 0, g_fb_h = 0, g_drawable_h = 0;

char   g_notify[256] = "";
double g_notify_time = -1000.0;

std::vector<RgVertex> g_verts;
std::vector<uint32_t> g_indices;

// The label column of the panel: every widget is drawn next to its key name.
constexpr float kLabelWidth      = 158.0f;
// The reset button every parameter row carries at its right edge, and the
// browse button of a texture path.
constexpr float kResetButtonSize = 24.0f;
constexpr float kBrowseButtonW   = 26.0f;
constexpr float kPi              = 3.14159265358979323846f;

float ClampF (float v, float mn, float mx)
{
	return v < mn ? mn : (v > mx ? mx : v);
}

// A tooltip for the item just drawn (SetItemTooltip applies the panel's hover
// delay, see ApplyStyle).
void ItemTooltip (const char *text)
{
	if (text && *text)
		ImGui::SetItemTooltip ("%s", text);
}

// The width a row's main widget may take: the reset button and the spacing
// before it are reserved at the right edge of the panel.
float RowWidth (float extra)
{
	const ImGuiStyle &style = ImGui::GetStyle ();
	const float       width = ImGui::GetContentRegionAvail ().x - kResetButtonSize - style.ItemSpacing.x - extra;

	return width > 24.0f ? width : 24.0f;
}

void LabelColumn (const char *label, const char *tooltip)
{
	ImGui::AlignTextToFramePadding ();
	ImGui::TextUnformatted (label);
	ItemTooltip (tooltip);
	ImGui::SameLine (kLabelWidth);
}

void WidgetId (char *out, size_t outsize, const char *label)
{
	snprintf (out, outsize, "##%s", label);
}

void ApplyStyle (void)
{
	ImGui::StyleColorsDark ();

	ImGuiStyle &s = ImGui::GetStyle ();

	s.WindowRounding    = 0.0f;
	s.ChildRounding     = 6.0f;
	s.FrameRounding     = 5.0f;
	s.GrabRounding      = 5.0f;
	s.PopupRounding     = 6.0f;
	s.ScrollbarRounding = 6.0f;
	s.TabRounding       = 5.0f;
	s.WindowBorderSize  = 1.0f;
	s.FramePadding      = ImVec2 (9.0f, 5.0f);
	s.ItemSpacing       = ImVec2 (9.0f, 7.0f);
	s.ItemInnerSpacing  = ImVec2 (7.0f, 5.0f);
	s.WindowPadding     = ImVec2 (14.0f, 12.0f);
	s.ScrollbarSize     = 14.0f;
	s.GrabMinSize       = 10.0f;

	// A parameter's hint appears after a second of hovering, like every other
	// tooltip of the panel (the disabled ones included: a locked roughness still
	// says why it is locked).
	s.HoverDelayNormal          = 1.0f;
	s.HoverFlagsForTooltipMouse = ImGuiHoveredFlags_Stationary | ImGuiHoveredFlags_DelayNormal |
	                              ImGuiHoveredFlags_AllowWhenDisabled;

	const ImVec4 accent = ImVec4 (0.26f, 0.59f, 0.98f, 1.00f);

	s.Colors[ImGuiCol_WindowBg]             = ImVec4 (0.075f, 0.082f, 0.100f, 0.97f);
	s.Colors[ImGuiCol_ChildBg]              = ImVec4 (0.055f, 0.060f, 0.075f, 0.85f);
	s.Colors[ImGuiCol_PopupBg]              = ImVec4 (0.090f, 0.095f, 0.115f, 0.98f);
	s.Colors[ImGuiCol_Border]               = ImVec4 (0.240f, 0.260f, 0.310f, 1.00f);
	s.Colors[ImGuiCol_FrameBg]              = ImVec4 (0.160f, 0.170f, 0.210f, 1.00f);
	s.Colors[ImGuiCol_FrameBgHovered]       = ImVec4 (0.220f, 0.240f, 0.290f, 1.00f);
	s.Colors[ImGuiCol_FrameBgActive]        = ImVec4 (0.260f, 0.280f, 0.340f, 1.00f);
	s.Colors[ImGuiCol_Button]               = ImVec4 (0.190f, 0.210f, 0.260f, 1.00f);
	s.Colors[ImGuiCol_ButtonHovered]        = ImVec4 (0.270f, 0.300f, 0.370f, 1.00f);
	s.Colors[ImGuiCol_ButtonActive]         = accent;
	s.Colors[ImGuiCol_Header]               = ImVec4 (0.200f, 0.240f, 0.310f, 1.00f);
	s.Colors[ImGuiCol_HeaderHovered]        = ImVec4 (0.260f, 0.310f, 0.400f, 1.00f);
	s.Colors[ImGuiCol_HeaderActive]         = ImVec4 (0.300f, 0.360f, 0.460f, 1.00f);
	s.Colors[ImGuiCol_SliderGrab]           = accent;
	s.Colors[ImGuiCol_SliderGrabActive]     = ImVec4 (0.420f, 0.700f, 1.000f, 1.00f);
	s.Colors[ImGuiCol_CheckMark]            = accent;
	s.Colors[ImGuiCol_Separator]            = ImVec4 (0.240f, 0.260f, 0.310f, 1.00f);
	s.Colors[ImGuiCol_Text]                 = ImVec4 (0.900f, 0.920f, 0.950f, 1.00f);
	s.Colors[ImGuiCol_TextDisabled]         = ImVec4 (0.500f, 0.520f, 0.570f, 1.00f);
	s.Colors[ImGuiCol_ScrollbarBg]          = ImVec4 (0.060f, 0.065f, 0.080f, 1.00f);
	s.Colors[ImGuiCol_ScrollbarGrab]        = ImVec4 (0.300f, 0.330f, 0.390f, 1.00f);
	s.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4 (0.380f, 0.420f, 0.500f, 1.00f);
	s.Colors[ImGuiCol_ScrollbarGrabActive]  = accent;
	s.Colors[ImGuiCol_TitleBg]              = ImVec4 (0.100f, 0.110f, 0.140f, 1.00f);
}

void UploadDrawData (void)
{
	ImDrawData *dd = ImGui::GetDrawData ();

	if (!dd || dd->CmdListsCount == 0 || dd->DisplaySize.x <= 0.0f || dd->DisplaySize.y <= 0.0f)
		return;

	const float W = dd->DisplaySize.x;
	const float H = dd->DisplaySize.y;

	// The same ortho as the engine's 2D canvas: (0,0) top-left, Y down.
	float m[16] = {};
	m[0]  = 2.0f / W;
	m[5]  = 2.0f / H;
	m[10] = -1.0f;
	m[12] = -1.0f;
	m[13] = -1.0f;
	m[15] = 1.0f;

	// Vulkan viewport keeps the Y flip; the scissor is top-left in both.
	RgViewport vp = {};
	vp.x        = (float)g_fb_x;
	vp.y        = (float)(g_drawable_h - (g_fb_y + g_fb_h));
	vp.width    = (float)g_fb_w;
	vp.height   = (float)g_fb_h;
	vp.minDepth = 0.0f;
	vp.maxDepth = 1.0f;

	const int area_x0 = g_fb_x;
	const int area_y0 = g_drawable_h - (g_fb_y + g_fb_h);
	const int area_x1 = area_x0 + g_fb_w;
	const int area_y1 = area_y0 + g_fb_h;

	const RgTransform identity = [] {
		RgTransform t = {};
		t.matrix[0][0] = t.matrix[1][1] = t.matrix[2][2] = 1.0f;
		return t;
	} ();

	for (int n = 0; n < dd->CmdListsCount; n++)
	{
		const ImDrawList *dl = dd->CmdLists[n];

		g_verts.resize ((size_t)dl->VtxBuffer.Size);
		for (int i = 0; i < dl->VtxBuffer.Size; i++)
		{
			const ImDrawVert &v = dl->VtxBuffer[i];
			RgVertex         &rv = g_verts[(size_t)i];

			rv = RgVertex ();
			rv.position[0] = v.pos.x;
			rv.position[1] = v.pos.y;
			rv.position[2] = 0.0f;
			rv.normal[2]   = 1.0f;
			rv.texCoord[0] = v.uv.x;
			rv.texCoord[1] = v.uv.y;
			rv.packedColor = v.col;
		}

		for (const ImDrawCmd &cmd : dl->CmdBuffer)
		{
			if (cmd.UserCallback != nullptr || cmd.ElemCount == 0)
				continue;

			// the clip rect in display coordinates -> a scissor in the drawable
			float cx0 = ClampF ((float)cmd.ClipRect.x, 0.0f, W);
			float cy0 = ClampF ((float)cmd.ClipRect.y, 0.0f, H);
			float cx1 = ClampF ((float)cmd.ClipRect.z, 0.0f, W);
			float cy1 = ClampF ((float)cmd.ClipRect.w, 0.0f, H);

			int x0 = area_x0 + (int)floorf (cx0);
			int y0 = area_y0 + (int)floorf (cy0);
			int x1 = area_x0 + (int)ceilf (cx1);
			int y1 = area_y0 + (int)ceilf (cy1);

			if (x0 < area_x0) x0 = area_x0;
			if (y0 < area_y0) y0 = area_y0;
			if (x1 > area_x1) x1 = area_x1;
			if (y1 > area_y1) y1 = area_y1;
			if (x1 <= x0 || y1 <= y0)
				continue;

			// rebase the indices to the command's own vertex span
			g_indices.resize (cmd.ElemCount);
			unsigned maxv = 0;
			const ImDrawIdx *src = dl->IdxBuffer.Data + cmd.IdxOffset;
			for (unsigned i = 0; i < cmd.ElemCount; i++)
			{
				unsigned idx = (unsigned)src[i] - (unsigned)cmd.VtxOffset;
				g_indices[i] = idx;
				if (idx > maxv)
					maxv = idx;
			}

			RgRasterizedGeometryUploadInfo info = {};
			info.renderType = RG_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN;
			info.vertexCount = maxv + 1;
			info.pVertices = g_verts.data () + cmd.VtxOffset;
			info.indexCount = cmd.ElemCount;
			info.pIndices = g_indices.data ();
			info.transform = identity;
			info.color.data[0] = info.color.data[1] = info.color.data[2] = info.color.data[3] = 1.0f;
			info.material = (RgMaterial)(uintptr_t)cmd.GetTexID ();
			info.pipelineState = RG_RASTERIZED_GEOMETRY_STATE_BLEND_ENABLE;
			info.blendFuncSrc = RG_BLEND_FACTOR_SRC_ALPHA;
			info.blendFuncDst = RG_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			info.scissor = { x0, y0, (uint32_t)(x1 - x0), (uint32_t)(y1 - y0) };

			rgUploadRasterizedGeometry (g_instance, &info, m, &vp);
		}
	}
}

void DrawNotification (void)
{
	const double age = ImGui::GetTime () - g_notify_time;

	if (g_notify[0] == '\0' || age > 6.0)
		return;

	float alpha = 1.0f;
	if (age > 5.0)
		alpha = (float)(6.0 - age); // fade out over the last second

	ImDrawList  *dl = ImGui::GetForegroundDrawList ();
	const ImVec2 pos (14.0f, 14.0f);

	dl->AddText (ImVec2 (pos.x + 1.0f, pos.y + 1.0f), IM_COL32 (0, 0, 0, (int)(170.0f * alpha)), g_notify);
	dl->AddText (pos, IM_COL32 (255, 226, 138, (int)(235.0f * alpha)), g_notify);
}

} // namespace

void QR_GUI_Init (void *sdl_window, void *rg_instance, const char *font_path)
{
	if (g_ready || sdl_window == NULL || rg_instance == NULL)
		return;

	IMGUI_CHECKVERSION ();
	ImGui::CreateContext ();

	ImGuiIO &io = ImGui::GetIO ();
	io.IniFilename = nullptr;
	io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

	ApplyStyle ();

	if (!ImGui_ImplSDL2_InitForOther ((SDL_Window *)sdl_window))
	{
		fprintf (stderr, "qr gui: SDL2 backend init failed\n");
		return;
	}

	ImFont *font = nullptr;
	if (font_path && font_path[0])
		font = io.Fonts->AddFontFromFileTTF (font_path, 17.0f);
	if (!font)
	{
		ImFontConfig cfg;
		cfg.SizePixels = 17.0f;
		font = io.Fonts->AddFontDefault (&cfg);
		if (font_path && font_path[0])
			fprintf (stderr, "qr gui: cannot load '%s', using the default font\n", font_path);
	}

	g_instance = (RgInstance)rg_instance;

	// The legacy atlas path: one texture, one material, no ImTextureData flow.
	unsigned char *pixels = nullptr;
	int            w = 0, h = 0;
	io.Fonts->GetTexDataAsRGBA32 (&pixels, &w, &h, nullptr);

	if (pixels && w > 0 && h > 0)
	{
		RgMaterialCreateInfo info = {};
		info.flags = 0;
		info.size = { (uint32_t)w, (uint32_t)h };
		info.textures.pDataAlbedoAlpha = pixels;
		info.pRelativePath = nullptr;
		info.filter = RG_SAMPLER_FILTER_LINEAR;
		info.addressModeU = RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		info.addressModeV = RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

		RgResult r = rgCreateMaterial (g_instance, &info, &g_font_material);
		if (r != RG_SUCCESS)
		{
			fprintf (stderr, "qr gui: font atlas material creation failed (%d)\n", (int)r);
			g_font_material = RG_NO_MATERIAL;
		}
		io.Fonts->SetTexID ((ImTextureID)(uintptr_t)g_font_material);
	}

	g_ready = true;
}

int QR_GUI_Ready (void)
{
	return g_ready ? 1 : 0;
}

void QR_GUI_Shutdown (void)
{
	if (!g_ready)
		return;

	ImGui_ImplSDL2_Shutdown ();
	ImGui::DestroyContext ();
	g_ready = false;
}

int QR_GUI_BeginFrame (unsigned int frame_id, float dt, int x, int y, int width, int height, int drawable_height)
{
	if (!g_ready || g_frame_open || frame_id == g_last_frame_id)
		return 0;

	g_last_frame_id = frame_id;
	g_fb_x = x;
	g_fb_y = y;
	g_fb_w = width;
	g_fb_h = height;
	g_drawable_h = drawable_height;

	ImGui_ImplSDL2_NewFrame ();

	ImGuiIO &io = ImGui::GetIO ();
	io.DisplaySize = ImVec2 ((float)width, (float)height);
	io.DisplayFramebufferScale = ImVec2 (1.0f, 1.0f);
	io.DeltaTime = dt > 0.0f ? dt : (1.0f / 60.0f);

	ImGui::NewFrame ();
	g_frame_open = true;
	return 1;
}

void QR_GUI_EndFrame (void)
{
	if (!g_frame_open)
		return;

	DrawNotification ();

	ImGui::Render ();
	g_frame_open = false;

	UploadDrawData ();
}

int QR_GUI_ProcessEvent (const void *sdl_event)
{
	if (!g_ready || sdl_event == NULL)
		return 0;

	const SDL_Event *e = (const SDL_Event *)sdl_event;

	ImGui_ImplSDL2_ProcessEvent (e);

	switch (e->type)
	{
	case SDL_MOUSEMOTION:
	case SDL_MOUSEBUTTONDOWN:
	case SDL_MOUSEBUTTONUP:
	case SDL_MOUSEWHEEL:
	case SDL_TEXTINPUT:
	case SDL_TEXTEDITING:
	case SDL_KEYDOWN:
	case SDL_KEYUP:
		return 1;
	default:
		return 0;
	}
}

int QR_GUI_WantsMouse (void)
{
	if (!g_ready)
		return 0;
	return ImGui::GetIO ().WantCaptureMouse ? 1 : 0;
}

int QR_GUI_WantsKeyboard (void)
{
	if (!g_ready)
		return 0;
	return ImGui::GetIO ().WantTextInput ? 1 : 0;
}

void QR_GUI_SetMouseCursor (int enable)
{
	if (!g_ready)
		return;
	ImGui::GetIO ().MouseDrawCursor = enable != 0;
}

void QR_GUI_BeginPanel (const char *id, int x, int y, int width, int height)
{
	ImGui::SetNextWindowPos (ImVec2 ((float)x, (float)y));
	ImGui::SetNextWindowSize (ImVec2 ((float)width, (float)height));

	ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
	                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
	                         ImGuiWindowFlags_NoSavedSettings;

	ImGui::Begin (id, nullptr, flags);
}

void QR_GUI_EndPanel (void)
{
	ImGui::End ();
}

void QR_GUI_BeginScroll (void)
{
	ImGui::BeginChild ("##qr_scroll", ImVec2 (0.0f, 0.0f), ImGuiChildFlags_None,
	                   ImGuiWindowFlags_NoSavedSettings);
}

void QR_GUI_EndScroll (void)
{
	ImGui::EndChild ();
}

void QR_GUI_Label (const char *text)
{
	ImGui::TextUnformatted (text);
}

void QR_GUI_LabelDim (const char *text)
{
	ImGui::TextDisabled ("%s", text);
}

void QR_GUI_Separator (void)
{
	ImGui::Separator ();
}

void QR_GUI_Spacing (void)
{
	ImGui::Spacing ();
}

void QR_GUI_SameLine (void)
{
	ImGui::SameLine ();
}

void QR_GUI_Tooltip (const char *text)
{
	ImGui::SetItemTooltip ("%s", text);
}

int QR_GUI_Button (const char *label)
{
	return ImGui::Button (label) ? 1 : 0;
}

int QR_GUI_Checkbox (const char *label, int *value, const char *tooltip)
{
	bool v = *value != 0;
	bool changed;

	changed = ImGui::Checkbox (label, &v);
	ItemTooltip (tooltip);
	if (changed)
		*value = v ? 1 : 0;
	return changed ? 1 : 0;
}

int QR_GUI_SliderFloat (const char *label, float *value, float min, float max, const char *tooltip)
{
	char id[192];
	WidgetId (id, sizeof (id), label);

	LabelColumn (label, tooltip);
	ImGui::SetNextItemWidth (RowWidth (0.0f));
	bool changed = ImGui::SliderFloat (id, value, min, max, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	if (tooltip && *tooltip)
		ImGui::SetItemTooltip ("%s\nCtrl+click to type a value", tooltip);
	else
		ImGui::SetItemTooltip ("Ctrl+click to type a value");
	return changed ? 1 : 0;
}

int QR_GUI_SliderInt (const char *label, int *value, int min, int max, const char *tooltip)
{
	char id[192];
	WidgetId (id, sizeof (id), label);

	LabelColumn (label, tooltip);
	ImGui::SetNextItemWidth (RowWidth (0.0f));
	bool changed = ImGui::SliderInt (id, value, min, max, "%d", ImGuiSliderFlags_AlwaysClamp);
	if (tooltip && *tooltip)
		ImGui::SetItemTooltip ("%s\nCtrl+click to type a value", tooltip);
	else
		ImGui::SetItemTooltip ("Ctrl+click to type a value");
	return changed ? 1 : 0;
}

int QR_GUI_Combo (const char *label, int *value, const char *const *items, int count, const char *tooltip)
{
	char id[192];
	WidgetId (id, sizeof (id), label);

	LabelColumn (label, tooltip);
	ImGui::SetNextItemWidth (RowWidth (0.0f));
	int changed = ImGui::Combo (id, value, items, count) ? 1 : 0;
	ItemTooltip (tooltip);
	return changed;
}

int QR_GUI_InputText (const char *label, char *buf, size_t capacity, const char *tooltip)
{
	char id[192];
	WidgetId (id, sizeof (id), label);

	LabelColumn (label, tooltip);
	ImGui::SetNextItemWidth (RowWidth (0.0f));
	int changed = ImGui::InputText (id, buf, capacity) ? 1 : 0;
	ItemTooltip (tooltip);
	return changed;
}

int QR_GUI_TexturePath (const char *label, char *buf, size_t capacity, const char *tooltip)
{
	char id[192];
	WidgetId (id, sizeof (id), label);

	int result = 0;

	LabelColumn (label, tooltip);
	ImGui::SetNextItemWidth (RowWidth (kBrowseButtonW + ImGui::GetStyle ().ItemSpacing.x));
	// An unauthored path is shown as NONE; the buffer stays empty, and typing
	// NONE by hand commits as "no texture" as well.
	if (ImGui::InputTextWithHint (id, "NONE", buf, capacity))
		result |= 1;
	ItemTooltip (tooltip);

	ImGui::SameLine ();
	ImGui::PushID (label);
	if (ImGui::Button ("...", ImVec2 (kBrowseButtonW, 0.0f)))
		result |= 2;
	ImGui::SetItemTooltip ("Browse for a file");
	ImGui::PopID ();

	return result;
}

// A square button with a circular arrow, right-aligned in its row: a parameter
// reset the widget before it did not have. Greyed out while the parameter still
// holds its original value.
int QR_GUI_ResetButton (const char *label, int enabled)
{
	// an ID of its own: the row widget already owns "##<label>", and two items
	// under one ID hand the click to the first of them
	char id[192];
	snprintf (id, sizeof (id), "##reset_%s", label);

	// right-aligned, but never over the widget that was just drawn
	const ImVec2 prev_max = ImGui::GetItemRectMax ();
	const float  after    = (prev_max.x - ImGui::GetWindowPos ().x) + ImGui::GetStyle ().ItemSpacing.x;
	const float  right    = ImGui::GetWindowContentRegionMax ().x - kResetButtonSize;
	const float  x        = right > after ? right : after;

	ImGui::SameLine (x);
	// a square, centred on the row whose frame is a little taller
	ImGui::SetCursorPosY (ImGui::GetCursorPosY () +
	                      (ImGui::GetFrameHeight () - kResetButtonSize) * 0.5f);

	if (!enabled)
		ImGui::BeginDisabled ();

	ImGui::PushStyleColor (ImGuiCol_Button, ImVec4 (0.16f, 0.17f, 0.21f, 0.85f));
	const bool pressed = ImGui::Button (id, ImVec2 (kResetButtonSize, kResetButtonSize));
	ImGui::PopStyleColor ();

	// The circular arrow is drawn, not a glyph: the panel font has no U+21BA.
	const ImVec2 mn  = ImGui::GetItemRectMin ();
	const ImVec2 mx  = ImGui::GetItemRectMax ();
	const ImU32  col = ImGui::GetColorU32 (ImGuiCol_Text, enabled ? 1.0f : 0.45f);
	ImDrawList  *dl  = ImGui::GetWindowDrawList ();

	const ImVec2 c ((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f);
	const float  r  = kResetButtonSize * 0.27f;
	const float  a0 = kPi * 0.35f;
	const float  a1 = kPi * 1.85f;

	dl->PathArcTo (c, r, a0, a1, 24);
	dl->PathStroke (col, 0, 1.5f);

	const ImVec2 p (c.x + r * cosf (a1), c.y + r * sinf (a1));
	const ImVec2 dir (cosf (a1), sinf (a1));
	const ImVec2 tang (-dir.y, dir.x);
	const float  head = 3.4f;

	dl->AddTriangleFilled (ImVec2 (p.x + tang.x * head * 1.4f, p.y + tang.y * head * 1.4f),
	                       ImVec2 (p.x + dir.x * head, p.y + dir.y * head),
	                       ImVec2 (p.x - dir.x * head, p.y - dir.y * head), col);

	ImGui::SetItemTooltip ("Reset to the original value");

	if (!enabled)
		ImGui::EndDisabled ();

	return pressed ? 1 : 0;
}

static int g_disabled_depth = 0;

void QR_GUI_PushDisabled (int disabled)
{
	if (!disabled)
		return;

	ImGui::BeginDisabled ();
	++g_disabled_depth;
}

void QR_GUI_PopDisabled (void)
{
	if (g_disabled_depth <= 0)
		return;

	ImGui::EndDisabled ();
	--g_disabled_depth;
}

int QR_GUI_ColorHex (const char *label, float rgb[3], int *enabled, const char *tooltip)
{
	int result = 0;

	LabelColumn (label, tooltip);

	ImGui::PushID (label);

	bool en = *enabled != 0;
	if (ImGui::Checkbox ("##enabled", &en))
	{
		*enabled = en ? 1 : 0;
		result = 1;
	}
	ImGui::SetItemTooltip ("Enable the colour");

	ImGui::SameLine ();

	// The item needs an explicit width: with the default one its swatch lands
	// past the right edge of the panel and can never be clicked, which is what
	// made the picker unreachable.
	ImGui::SetNextItemWidth (RowWidth (0.0f));
	if (ImGui::ColorEdit3 ("##color", rgb, ImGuiColorEditFlags_DisplayHex))
	{
		// editing the colour is what enables it; the checkbox above only shows
		// the state and can turn it off again
		*enabled = 1;
		result = 1;
	}
	if (tooltip && *tooltip)
		ImGui::SetItemTooltip ("%s\nClick the swatch to pick, type the hex value", tooltip);
	else
		ImGui::SetItemTooltip ("Click the swatch to pick, type the hex value");

	ImGui::PopID ();

	return result;
}

int QR_GUI_Section (const char *label, int default_open)
{
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;
	if (default_open)
		flags |= ImGuiTreeNodeFlags_DefaultOpen;

	return ImGui::CollapsingHeader (label, flags) ? 1 : 0;
}

void QR_GUI_PushID (const char *id)
{
	ImGui::PushID (id);
}

void QR_GUI_PopID (void)
{
	ImGui::PopID ();
}

int QR_GUI_AnyItemActive (void)
{
	return ImGui::IsAnyItemActive () ? 1 : 0;
}

int QR_GUI_ImagePick (const char *id, int64_t texture, int tex_w, int tex_h, float *out_u, float *out_v)
{
	ImGui::PushID (id);

	float w = 260.0f;
	float h = 260.0f;

	if (tex_w > 0 && tex_h > 0)
	{
		h = 260.0f * (float)tex_h / (float)tex_w;
		if (h > 260.0f)
		{
			h = 260.0f;
			w = 260.0f * (float)tex_w / (float)tex_h;
		}
	}

	const ImVec2 pos = ImGui::GetCursorScreenPos ();
	const ImVec2 size (w, h);

	ImGui::Image ((ImTextureID)(uint64_t)texture, size, ImVec2 (0.0f, 0.0f), ImVec2 (1.0f, 1.0f));

	const bool hovered = ImGui::IsItemHovered ();
	const ImVec2 mouse = ImGui::GetIO ().MousePos;
	int          result = 0;

	if (hovered)
	{
		float u = (mouse.x - pos.x) / (w > 1.0f ? w : 1.0f);
		float v = (mouse.y - pos.y) / (h > 1.0f ? h : 1.0f);

		u = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
		v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
		if (out_u)
			*out_u = u;
		if (out_v)
			*out_v = v;

		if (ImGui::IsMouseDown (ImGuiMouseButton_Left))
			result = 1;

		ImGui::SetMouseCursor (ImGuiMouseCursor_Hand);
	}

	ImDrawList *dl = ImGui::GetWindowDrawList ();

	dl->AddRect (pos, ImVec2 (pos.x + w, pos.y + h), IM_COL32 (150, 160, 185, 255));

	if (hovered)
	{
		const ImU32 col = IM_COL32 (255, 230, 150, 230);

		dl->AddLine (ImVec2 (mouse.x - 8.0f, mouse.y), ImVec2 (mouse.x + 8.0f, mouse.y), col, 1.5f);
		dl->AddLine (ImVec2 (mouse.x, mouse.y - 8.0f), ImVec2 (mouse.x, mouse.y + 8.0f), col, 1.5f);
		dl->AddCircle (mouse, 4.0f, col, 16, 1.5f);
	}

	ImGui::PopID ();
	return result;
}

int QR_GUI_Dialog (const char *title, const char *text, const char *yes, const char *no)
{
	const ImVec2 center (ImGui::GetIO ().DisplaySize.x * 0.5f, ImGui::GetIO ().DisplaySize.y * 0.5f);
	int          result = 0;

	ImGui::SetNextWindowPos (center, ImGuiCond_Always, ImVec2 (0.5f, 0.5f));
	ImGui::SetNextWindowSize (ImVec2 (440.0f, 0.0f), ImGuiCond_Always);

	const ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
	                               ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
	                               ImGuiWindowFlags_AlwaysAutoResize;

	if (ImGui::Begin (title, nullptr, flags))
	{
		ImGui::TextWrapped ("%s", text);
		ImGui::Spacing ();
		if (ImGui::Button (yes, ImVec2 (150.0f, 0.0f)))
			result = 1;
		ImGui::SameLine ();
		if (ImGui::Button (no, ImVec2 (150.0f, 0.0f)))
			result = 2;
	}
	ImGui::End ();

	return result;
}

void QR_GUI_Notify (const char *text)
{
	snprintf (g_notify, sizeof (g_notify), "%s", text ? text : "");
	g_notify_time = ImGui::GetTime ();
}

void QR_GUI_DrawCrosshair (void)
{
	ImDrawList  *dl = ImGui::GetForegroundDrawList ();
	const ImVec2 c (ImGui::GetIO ().DisplaySize.x * 0.5f, ImGui::GetIO ().DisplaySize.y * 0.5f);

	const float  gap = 4.0f;
	const float  len = 7.0f;
	const float  th  = 1.6f;
	const ImU32  shadow = IM_COL32 (0, 0, 0, 130);
	const ImU32  col = IM_COL32 (255, 255, 255, 225);

	const ImVec2 a0 (c.x - gap - len, c.y), a1 (c.x - gap, c.y);
	const ImVec2 b0 (c.x + gap, c.y), b1 (c.x + gap + len, c.y);
	const ImVec2 c0 (c.x, c.y - gap - len), c1 (c.x, c.y - gap);
	const ImVec2 d0 (c.x, c.y + gap), d1 (c.x, c.y + gap + len);

	dl->AddLine (a0, a1, shadow, th + 2.0f);
	dl->AddLine (b0, b1, shadow, th + 2.0f);
	dl->AddLine (c0, c1, shadow, th + 2.0f);
	dl->AddLine (d0, d1, shadow, th + 2.0f);

	dl->AddLine (a0, a1, col, th);
	dl->AddLine (b0, b1, col, th);
	dl->AddLine (c0, c1, col, th);
	dl->AddLine (d0, d1, col, th);
}

void QR_GUI_DrawHint (const char *const *lines, int count)
{
	ImDrawList *dl = ImGui::GetForegroundDrawList ();

	const float pad = 14.0f;
	const float line_height = ImGui::GetTextLineHeight () + 3.0f;
	ImVec2      pos (pad, ImGui::GetIO ().DisplaySize.y - pad - line_height * (float)count);

	for (int i = 0; i < count; i++)
	{
		const ImU32 col = (i == 0) ? IM_COL32 (140, 205, 255, 235) : IM_COL32 (228, 234, 244, 220);

		dl->AddText (ImVec2 (pos.x + 1.0f, pos.y + 1.0f), IM_COL32 (0, 0, 0, 160), lines[i]);
		dl->AddText (pos, col, lines[i]);

		pos.y += line_height;
	}
}
