#include "RhiRtComposePass.h"

#include "RhiFrameContext.h"
#include "RhiPipeline.h"
#include "RhiResources.h"
#include "RhiTextureSource.h"

#include "../Framebuffers.h"
#include "../Generated/ShaderCommonC.h"
#include "../Tonemapping.h"
#include "../Utils.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <string>
#include <tuple>

using namespace vkpt;

namespace
{

// The thirteen engine blobs, by the file names the shader build writes into the folder the host
// passes in (ShaderManager.cpp:41-49) - the same `Cm*` names the legacy Q2Denoiser, Tonemapping and
// ImageComposition pipelines load, and the blobs the reconnaissance measured. The seven chain
// blobs are the ASVGF chain of A4.5: `CmQ2GradientReproject` (the pre-direct step),
// `CmQ2GradientImg` / `CmQ2GradientAtrous` / `CmQ2Temporal` / `CmQ2AtrousLF` / `CmQ2Atrous` (the
// post-indirect chain, Q2Denoiser.cpp:410-597), `CmQ2Adapter` (the ReSTIR -> ASVGF channel adapter,
// both branches) and `CmQ2TAAU` (the upscaler, :625-657). The adapter and the interleave are the
// `flt_enable = 0` half of the denoiser, the histogram and the average are the engine's exposure
// chain (Tonemapping.cpp:171-208), the checkerboard is the resolve
// ImageComposition::ProcessCheckerboard records (ImageComposition.cpp:163-202), and the final
// composition is ImageComposition::ApplyTonemapping (ImageComposition.cpp:113-161).
const char *const GRADIENT_REPROJECT_SHADER_FILE_NAME = "CmQ2GradientReproject.comp.spv";
const char *const ADAPTER_SHADER_FILE_NAME = "CmQ2Adapter.comp.spv";
const char *const GRADIENT_IMG_SHADER_FILE_NAME = "CmQ2GradientImg.comp.spv";
const char *const GRADIENT_ATROUS_SHADER_FILE_NAME = "CmQ2GradientAtrous.comp.spv";
const char *const TEMPORAL_SHADER_FILE_NAME = "CmQ2Temporal.comp.spv";
const char *const ATROUS_LF_SHADER_FILE_NAME = "CmQ2AtrousLF.comp.spv";
const char *const ATROUS_SHADER_FILE_NAME = "CmQ2Atrous.comp.spv";
const char *const INTERLEAVE_SHADER_FILE_NAME = "CmQ2Interleave.comp.spv";
const char *const HISTOGRAM_SHADER_FILE_NAME = "CmLuminanceHistogram.comp.spv";
const char *const AVERAGE_SHADER_FILE_NAME = "CmLuminanceAvg.comp.spv";
const char *const CHECKERBOARD_SHADER_FILE_NAME = "CmCheckerboard.comp.spv";
const char *const PREPARE_FINAL_SHADER_FILE_NAME = "CmPrepareFinal.comp.spv";
const char *const TAAU_SHADER_FILE_NAME = "CmQ2TAAU.comp.spv";

// The workgroup sizes the legacy dispatches use (all measured with `spirv-dis` over the blobs):
// every pass is `[numthreads(16, 16, 1)]` except `CmQ2Temporal`, which is `[numthreads(15, 15, 1)]`
// (Q2RTX GROUP_SIZE, Q2Denoiser.cpp:403-405), and `CmLuminanceAvg`, which is `[numthreads(128, 1,
// 1)]` and is dispatched once. The dispatch is the legacy `Utils::GetWorkGroupCount(size, group)`
// (`1 + ceil(size / group)`, Utils.cpp:319-328); Q2Denoiser.cpp:400-408, :472, :519, :593, :606-607,
// :652-653, ImageComposition.cpp:198-199, Tonemapping.cpp:176-179.
constexpr uint32_t COMPOSE_GROUP_SIZE = 16;
constexpr uint32_t TEMPORAL_GROUP_SIZE = 15;

// The gradient and LF-a-trous passes dispatch at 1/3 resolution: the legacy divides the render size
// by `COMPUTE_ASVGF_STRATA_SIZE` first and then asks for the full workgroup count
// (Q2Denoiser.cpp:344-345, :407-408, :473, :519).
constexpr uint32_t COMPOSE_STRATA_SIZE = 3;

// The 4-byte `IterationInfo_BT` push constant of `CmQ2GradientAtrous` and `CmQ2AtrousLF`, one
// `uint` at offset 0 (measured: both blobs declare the block as the pipeline's only push range and
// the legacy writes exactly that (Q2Denoiser.cpp:472, :518)).
constexpr uint32_t COMPOSE_ITERATION_PUSH_SIZE = 4;

// The iteration counts: 7 for the gradient filter (LF in all, HF/SPEC in the first 3, normalized in
// the last one, Q2Denoiser.h:88-91) and the engine's 4 for both LF and HF/SPEC a-trous
// (`COMPUTE_SVGF_ATROUS_ITERATION_COUNT`, Generated/ShaderCommonC.h:148).
constexpr uint32_t GRADIENT_ATROUS_ITERATION_COUNT = 7;
static_assert(COMPUTE_SVGF_ATROUS_ITERATION_COUNT == 4,
              "the atrous tables below are written for the engine's four iterations");

// The sets of the thirteen passes were measured 2026-09-26 with `spirv-dis` over the shipped
// blobs; the shader sources are the same files. The engine's raw bindings are the identity for the
// storage images (`ShFramebuffers_Bindings[index] == index`), `124 + index` for the sampled views
// (`ShFramebuffers_Sampled_Bindings`) and `248 + index` for the samplers
// (`ShFramebuffers_Sampler_Bindings`), so the layout offsets below turn a raw binding into its
// NVRHI slot: raw = offset + slot (RhiPipeline.h). The numbers are still derived from the generated
// arrays at run time (GetComposeSlot), so a generator change cannot drift from them.
constexpr uint32_t FRAMEBUFFER_UAV_OFFSET = 0;
constexpr uint32_t FRAMEBUFFER_SRV_OFFSET = 124;
constexpr uint32_t FRAMEBUFFER_SAMPLER_OFFSET = 248;

// The size class of one union image: the engine's `GetFramebufSize` maps the generated flags onto
// the render size, the `(render + 1) / 2` half (FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_1_2: the
// god-rays image 63), the `(render + 1) / 3` third (FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_1_3:
// 101-104 and 111-116) and the upscaled size (FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_UPSCALED_SIZE:
// 29, 119, 120) (Framebuffers.cpp:628-692).
enum class ComposeImageSize : uint8_t
{
    Render,
    Half,
    Third,
    Upscaled,
};

struct ComposeImage
{
    FramebufferImageIndex image;
    ComposeImageSize size;
};

// The union of the engine images the thirteen passes touch, deduplicated: 80 images, each wrapped
// once per slot and shared by the twelve framebuffer sets, so NVRHI's own UAV -> SRV -> UAV
// transitions on one wrap order the pass-to-pass hand-offs (the class comment's list). The first
// 27 entries are the A4.4 compose union; the rest are the chain's, both members of every `_PREV`
// role pair included (1, 4, 6, 8, 20, 24, 78, 80, 82, 84, 86, 92, 94, 96, 98, 100, 116, 120, 122),
// resolved through `Framebuffers::GetImageHandles(image, frameIndex)` which applies the engine's
// per-slot role swap (Framebuffers.cpp:33-53). Three entries are the god-rays hand-off: 63 GOD_RAYS
// (the trace's half-res storage image and the filter's sampled input), 64 GOD_RAYS_FILTERED (the
// filter's render-sized storage image, sampled by the final composition at raw 188) and 89
// Q2_GOD_RAYS_THROUGHPUT_DIST (the trace's sampled input). Only 64 is bound by a set of this module;
// the three are part of the union so the per-list announcement below names their resting state and
// the restore lists return the ones this module moves to it. The god-rays module owns their
// contents and has to leave them in GENERAL (see the class comment).
constexpr uint32_t COMPOSE_IMAGE_COUNT = 80;
constexpr ComposeImage COMPOSE_IMAGES[COMPOSE_IMAGE_COUNT] =
{
    { FB_IMAGE_INDEX_ALBEDO,                ComposeImageSize::Render }, //   0  framebufAlbedo           (adapter/gradient-reproj/atrous/PF)
    { FB_IMAGE_INDEX_ALBEDO_PREV,           ComposeImageSize::Render }, //   1  framebufAlbedo_Prev      (gradient-reproj SRV)
    { FB_IMAGE_INDEX_IS_SKY,                ComposeImageSize::Render }, //   2  framebufIsSky            (adapter/atrous SRV)
    { FB_IMAGE_INDEX_NORMAL,                ComposeImageSize::Render }, //   3  framebufNormal           (reproj/atrous UAV, adapter/temporal/PF SRV)
    { FB_IMAGE_INDEX_NORMAL_PREV,           ComposeImageSize::Render }, //   4  framebufNormal_Prev      (reproj/temporal SRV)
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY,       ComposeImageSize::Render }, //   5  framebufNormalGeometry   (reproj/temporal/atrousLF/atrous SRV)
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY_PREV,  ComposeImageSize::Render }, //   6  framebufNormalGeometry_Prev
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,    ComposeImageSize::Render }, //   7  framebufMetallicRoughness (reproj UAV, adapter/temporal/atrous SRV)
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS_PREV, ComposeImageSize::Render }, // 8 framebufMetallicRoughness_Prev
    { FB_IMAGE_INDEX_DEPTH_WORLD,           ComposeImageSize::Render }, //   9  framebufDepthWorld       (final SRV)
    { FB_IMAGE_INDEX_DEPTH_GRAD,            ComposeImageSize::Render }, //  11  framebufDepthGrad        (temporal/atrousLF/atrous SRV)
    { FB_IMAGE_INDEX_MOTION,                ComposeImageSize::Render }, //  13  framebufMotion           (reproj/gradientImg/temporal/final SRV)
    { FB_IMAGE_INDEX_UNFILTERED_DIRECT,     ComposeImageSize::Render }, //  14  framebufUnfilteredDirect  (adapter/final SRV)
    { FB_IMAGE_INDEX_UNFILTERED_SPECULAR,   ComposeImageSize::Render }, //  15  framebufUnfilteredSpecular (adapter/final SRV)
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R, ComposeImageSize::Render }, // 16 framebufUnfilteredIndirectSH_R (adapter SRV; final UAV; written by the indirect pass)
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G, ComposeImageSize::Render }, // 17 framebufUnfilteredIndirectSH_G
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B, ComposeImageSize::Render }, // 18 framebufUnfilteredIndirectSH_B
    { FB_IMAGE_INDEX_SURFACE_POSITION,      ComposeImageSize::Render }, //  19  framebufSurfacePosition  (reproj UAV)
    { FB_IMAGE_INDEX_SURFACE_POSITION_PREV, ComposeImageSize::Render }, //  20  framebufSurfacePosition_Prev (reproj SRV)
    { FB_IMAGE_INDEX_VIEW_DIRECTION,        ComposeImageSize::Render }, //  23  framebufViewDirection    (reproj UAV)
    { FB_IMAGE_INDEX_VIEW_DIRECTION_PREV,   ComposeImageSize::Render }, //  24  framebufViewDirection_Prev (reproj SRV)
    { FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR,  ComposeImageSize::Render }, //  25  framebufPrimaryToReflRefr (final SRV)
    { FB_IMAGE_INDEX_THROUGHPUT,            ComposeImageSize::Render }, //  26  framebufThroughput       (reproj/adapter/temporal/atrous/interleave/checkerboard/final SRV)
    { FB_IMAGE_INDEX_PRE_FINAL,             ComposeImageSize::Render }, //  27  framebufPreFinal         (interleave UAV, histogram/checkerboard SRV)
    { FB_IMAGE_INDEX_FINAL,                 ComposeImageSize::Render }, //  28  framebufFinal            (checkerboard/final/TAAU UAV/SRV)
    { FB_IMAGE_INDEX_UPSCALED_PING,         ComposeImageSize::Upscaled }, // 29 framebufUpscaledPing    (TAAU UAV, present source)
    { FB_IMAGE_INDEX_MOTION_DLSS,           ComposeImageSize::Render }, //  31  framebufMotionDlss       (TAAU SRV; written by the primary)
    { FB_IMAGE_INDEX_ACID_FOG_R_T,          ComposeImageSize::Render }, //  59  framebufAcidFogRT        (checkerboard SRV)
    { FB_IMAGE_INDEX_ACID_FOG,              ComposeImageSize::Render }, //  60  framebufAcidFog          (checkerboard UAV, final SRV)
    { FB_IMAGE_INDEX_SCREEN_EMIS_R_T,       ComposeImageSize::Render }, //  61  framebufScreenEmisRT     (checkerboard SRV)
    { FB_IMAGE_INDEX_SCREEN_EMISSION,       ComposeImageSize::Render }, //  62  framebufScreenEmission   (checkerboard UAV, final SRV)
    { FB_IMAGE_INDEX_GOD_RAYS,              ComposeImageSize::Half },   //  63  framebufGodRays          (god-rays trace UAV, filter SRV)
    { FB_IMAGE_INDEX_GOD_RAYS_FILTERED,     ComposeImageSize::Render }, //  64  framebufGodRaysFiltered  (god-rays filter UAV, final SRV at raw 188)
    { FB_IMAGE_INDEX_BLOOM_INPUT,           ComposeImageSize::Render }, //  65  framebufBloomInput       (final UAV; bloom is off)
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,      ComposeImageSize::Render }, //  73  framebufQ2ColorLF_SH     (adapter UAV, gradientImg/temporal SRV)
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_C_O_C_G,  ComposeImageSize::Render }, //  75  framebufQ2ColorLF_COCG   (adapter UAV, temporal SRV)
    { FB_IMAGE_INDEX_Q2_COLOR_H_F,          ComposeImageSize::Render }, //  77  framebufQ2ColorHF        (adapter UAV, gradientImg/temporal SRV)
    { FB_IMAGE_INDEX_Q2_COLOR_H_F_PREV,     ComposeImageSize::Render }, //  78  framebufQ2ColorHF_Prev   (reproj SRV)
    { FB_IMAGE_INDEX_Q2_COLOR_SPEC,         ComposeImageSize::Render }, //  79  framebufQ2ColorSpec      (adapter UAV, gradientImg/temporal SRV)
    { FB_IMAGE_INDEX_Q2_COLOR_SPEC_PREV,    ComposeImageSize::Render }, //  80  framebufQ2ColorSpec_Prev (reproj SRV)
    { FB_IMAGE_INDEX_Q2_VIEW_DEPTH,         ComposeImageSize::Render }, //  81  framebufQ2ViewDepth      (reproj/temporal/atrousLF/atrous SRV)
    { FB_IMAGE_INDEX_Q2_VIEW_DEPTH_PREV,    ComposeImageSize::Render }, //  82  framebufQ2ViewDepth_Prev (reproj/temporal SRV)
    { FB_IMAGE_INDEX_Q2_BASE_COLOR,         ComposeImageSize::Render }, //  83  framebufQ2BaseColor      (reproj UAV)
    { FB_IMAGE_INDEX_Q2_BASE_COLOR_PREV,    ComposeImageSize::Render }, //  84  framebufQ2BaseColor_Prev (reproj SRV)
    { FB_IMAGE_INDEX_Q2_METALLIC,           ComposeImageSize::Render }, //  85  framebufQ2Metallic       (reproj UAV)
    { FB_IMAGE_INDEX_Q2_METALLIC_PREV,      ComposeImageSize::Render }, //  86  framebufQ2Metallic_Prev  (reproj SRV)
    { FB_IMAGE_INDEX_Q2_TRANSPARENT,        ComposeImageSize::Render }, //  88  framebufQ2Transparent    (adapter/atrous SRV)
    { FB_IMAGE_INDEX_Q2_GOD_RAYS_THROUGHPUT_DIST, ComposeImageSize::Render }, // 89 framebufQ2GodRaysThroughputDist (primary/Q2-reflrefr UAV, god-rays trace SRV)
    { FB_IMAGE_INDEX_Q2_FOG_ACCUM,          ComposeImageSize::Render }, //  90  framebufQ2FogAccum       (adapter/atrous SRV)
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_S_H, ComposeImageSize::Render }, //  91  framebufQ2HistColorLF_SH (temporal UAV, atrous SRV)
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_S_H_PREV, ComposeImageSize::Render }, // 92 framebufQ2HistColorLF_SH_Prev (gradientImg/temporal SRV)
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_C_O_C_G, ComposeImageSize::Render }, // 93 framebufQ2HistColorLF_COCG (temporal UAV, atrous SRV)
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_C_O_C_G_PREV, ComposeImageSize::Render }, // 94 framebufQ2HistColorLF_COCG_Prev (temporal SRV)
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_H_F,     ComposeImageSize::Render }, //  95  framebufQ2HistColorHF    (atrous UAV)
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_H_F_PREV, ComposeImageSize::Render }, //  96 framebufQ2HistColorHF_Prev (temporal SRV)
    { FB_IMAGE_INDEX_Q2_HIST_MOMENTS_H_F,   ComposeImageSize::Render }, //  97  framebufQ2HistMomentsHF  (temporal UAV, atrous SRV)
    { FB_IMAGE_INDEX_Q2_HIST_MOMENTS_H_F_PREV, ComposeImageSize::Render }, // 98 framebufQ2HistMomentsHF_Prev (temporal SRV)
    { FB_IMAGE_INDEX_Q2_FILTERED_SPEC,      ComposeImageSize::Render }, //  99  framebufQ2FilteredSpec   (temporal UAV)
    { FB_IMAGE_INDEX_Q2_FILTERED_SPEC_PREV, ComposeImageSize::Render }, // 100  framebufQ2FilteredSpec_Prev (temporal SRV)
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_S_H, ComposeImageSize::Third }, // 101  framebufQ2AtrousPingLF_SH (temporal/atrousLF UAV, atrous SRV)
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_L_F_S_H, ComposeImageSize::Third }, // 102  framebufQ2AtrousPongLF_SH (atrousLF UAV)
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_C_O_C_G, ComposeImageSize::Third }, // 103 framebufQ2AtrousPingLF_COCG
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_L_F_C_O_C_G, ComposeImageSize::Third }, // 104 framebufQ2AtrousPongLF_COCG
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_H_F,    ComposeImageSize::Render }, // 105  framebufQ2AtrousPingHF   (temporal/atrous UAV)
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_H_F,    ComposeImageSize::Render }, // 106  framebufQ2AtrousPongHF
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_SPEC,   ComposeImageSize::Render }, // 107  framebufQ2AtrousPingSpec
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_SPEC,   ComposeImageSize::Render }, // 108  framebufQ2AtrousPongSpec
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_MOMENTS, ComposeImageSize::Render }, // 109 framebufQ2AtrousPingMoments
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_MOMENTS, ComposeImageSize::Render }, // 110 framebufQ2AtrousPongMoments
    { FB_IMAGE_INDEX_Q2_GRAD_L_F_PING,      ComposeImageSize::Third }, // 111  framebufQ2GradLFPing      (gradientImg/gradientAtrous UAV)
    { FB_IMAGE_INDEX_Q2_GRAD_L_F_PONG,      ComposeImageSize::Third }, // 112  framebufQ2GradLFPong      (gradientAtrous UAV, temporal SRV)
    { FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PING, ComposeImageSize::Third }, // 113  framebufQ2GradHFSpecPing  (reproj/gradientImg/gradientAtrous UAV)
    { FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PONG, ComposeImageSize::Third }, // 114  framebufQ2GradHFSpecPong  (gradientAtrous UAV, temporal SRV)
    { FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS,      ComposeImageSize::Third }, // 115  framebufQ2GradSmplPos     (reproj UAV, gradientImg SRV, raygen read)
    { FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS_PREV, ComposeImageSize::Third }, // 116  framebufQ2GradSmplPos_Prev (reproj SRV)
    { FB_IMAGE_INDEX_Q2_COLOR,              ComposeImageSize::Render }, // 117  framebufQ2Color           (adapter/atrous UAV, interleave SRV)
    { FB_IMAGE_INDEX_Q2_TAA_HISTORY,        ComposeImageSize::Upscaled }, // 119 framebufQ2TaaHistory    (TAAU UAV)
    { FB_IMAGE_INDEX_Q2_TAA_HISTORY_PREV,   ComposeImageSize::Upscaled }, // 120 framebufQ2TaaHistory_Prev (TAAU SRV + sampler)
    { FB_IMAGE_INDEX_Q2_RNG_SEED,           ComposeImageSize::Render }, // 121  framebufQ2RngSeed        (reproj UAV, raygen read)
    { FB_IMAGE_INDEX_Q2_RNG_SEED_PREV,      ComposeImageSize::Render }, // 122  framebufQ2RngSeed_Prev   (reproj SRV)
};

static_assert(COMPOSE_IMAGE_COUNT == 80,
              "the per-slot image arrays of RhiRtComposePass.h are sized 80 (COMPOSE_IMAGES)");

// One item of a pass's set 0: the engine image, the binding kind and whether the item is a sampler
// (only the TAAU history carries one). 'isUAV' selects both the layout item and the set item type;
// 'image' also carries the raw binding, through the generated array of the matching kind.
struct ComposeBinding
{
    FramebufferImageIndex image;
    bool isUAV;
    bool isSampler;
};

// CmQ2GradientReproject: 10 storage images and 17 sampled images (measured: exactly these 27
// bindings, all in set 0). The order is the reconnaissance's table order - the UAVs first, then the
// SRVs. The reproject reads the `_PREV` roles and writes the current ones for the checkerboard half
// the primary did not trace.
constexpr uint32_t GRADIENT_REPROJECT_BINDING_COUNT = 27;
constexpr ComposeBinding GRADIENT_REPROJECT_BINDINGS[GRADIENT_REPROJECT_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_ALBEDO,               true, false }, //   0  framebufAlbedo
    { FB_IMAGE_INDEX_NORMAL,               true, false }, //   3  framebufNormal
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,   true, false }, //   7  framebufMetallicRoughness
    { FB_IMAGE_INDEX_SURFACE_POSITION,     true, false }, //  19  framebufSurfacePosition
    { FB_IMAGE_INDEX_VIEW_DIRECTION,       true, false }, //  23  framebufViewDirection
    { FB_IMAGE_INDEX_Q2_BASE_COLOR,        true, false }, //  83  framebufQ2BaseColor
    { FB_IMAGE_INDEX_Q2_METALLIC,          true, false }, //  85  framebufQ2Metallic
    { FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PING, true, false }, // 113  framebufQ2GradHFSpecPing
    { FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS,     true, false }, // 115  framebufQ2GradSmplPos
    { FB_IMAGE_INDEX_Q2_RNG_SEED,          true, false }, // 121  framebufQ2RngSeed
    { FB_IMAGE_INDEX_ALBEDO_PREV,          false, false }, // 125  framebufAlbedo_Prev_Sampled
    { FB_IMAGE_INDEX_NORMAL_PREV,          false, false }, // 128  framebufNormal_Prev_Sampled
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY,      false, false }, // 129  framebufNormalGeometry_Sampled
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY_PREV, false, false }, // 130  framebufNormalGeometry_Prev_Sampled
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS_PREV, false, false }, // 132 framebufMetallicRoughness_Prev_Sampled
    { FB_IMAGE_INDEX_MOTION,               false, false }, // 137  framebufMotion_Sampled
    { FB_IMAGE_INDEX_SURFACE_POSITION_PREV, false, false }, // 144 framebufSurfacePosition_Prev_Sampled
    { FB_IMAGE_INDEX_VIEW_DIRECTION_PREV,  false, false }, // 148  framebufViewDirection_Prev_Sampled
    { FB_IMAGE_INDEX_THROUGHPUT,           false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_H_F_PREV,    false, false }, // 202  framebufQ2ColorHF_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_SPEC_PREV,   false, false }, // 204  framebufQ2ColorSpec_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_VIEW_DEPTH,        false, false }, // 205  framebufQ2ViewDepth_Sampled
    { FB_IMAGE_INDEX_Q2_VIEW_DEPTH_PREV,   false, false }, // 206  framebufQ2ViewDepth_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_BASE_COLOR_PREV,   false, false }, // 208  framebufQ2BaseColor_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_METALLIC_PREV,     false, false }, // 210  framebufQ2Metallic_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS_PREV, false, false }, // 240 framebufQ2GradSmplPos_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_RNG_SEED_PREV,     false, false }, // 246  framebufQ2RngSeed_Prev_Sampled
};

// CmQ2Adapter: 5 storage images and 12 sampled images (measured: exactly these 17 bindings, all in
// set 0; the module's own uniform layout carries set 1). The order is the reconnaissance's table
// order - the UAVs first, then the SRVs. 16-18 are sampled like the rest; the A4.3 indirect pass
// writes them earlier on the same command list.
constexpr uint32_t ADAPTER_BINDING_COUNT = 17;
constexpr ComposeBinding ADAPTER_BINDINGS[ADAPTER_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,     true, false }, //  73  framebufQ2ColorLF_SH
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_C_O_C_G, true, false }, //  75  framebufQ2ColorLF_COCG
    { FB_IMAGE_INDEX_Q2_COLOR_H_F,         true, false }, //  77  framebufQ2ColorHF
    { FB_IMAGE_INDEX_Q2_COLOR_SPEC,        true, false }, //  79  framebufQ2ColorSpec
    { FB_IMAGE_INDEX_Q2_COLOR,             true, false }, // 117  framebufQ2Color
    { FB_IMAGE_INDEX_ALBEDO,               false, false }, // 124  framebufAlbedo_Sampled
    { FB_IMAGE_INDEX_IS_SKY,               false, false }, // 126  framebufIsSky_Sampled
    { FB_IMAGE_INDEX_NORMAL,               false, false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,   false, false }, // 131  framebufMetallicRoughness_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_DIRECT,    false, false }, // 138  framebufUnfilteredDirect_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_SPECULAR,  false, false }, // 139  framebufUnfilteredSpecular_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R, false, false }, // 140  framebufUnfilteredIndirectSH_R_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G, false, false }, // 141  framebufUnfilteredIndirectSH_G_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B, false, false }, // 142  framebufUnfilteredIndirectSH_B_Sampled
    { FB_IMAGE_INDEX_THROUGHPUT,           false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_TRANSPARENT,       false, false }, // 212  framebufQ2Transparent_Sampled
    { FB_IMAGE_INDEX_Q2_FOG_ACCUM,         false, false }, // 214  framebufQ2FogAccum_Sampled
};

// CmQ2GradientImg: 2 storage images and 6 sampled images. The A4.5 pair fix reads
// `framebufQ2GradHFSpecPing` through its storage image, so its sampled view (raw 237) is
// dead-stripped and the measured item count is 8 instead of the pre-fix 9.
constexpr uint32_t GRADIENT_IMG_BINDING_COUNT = 8;
constexpr ComposeBinding GRADIENT_IMG_BINDINGS[GRADIENT_IMG_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_GRAD_L_F_PING,      true, false }, // 111  framebufQ2GradLFPing
    { FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PING, true, false }, // 113  framebufQ2GradHFSpecPing
    { FB_IMAGE_INDEX_MOTION,                false, false }, // 137  framebufMotion_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,      false, false }, // 197  framebufQ2ColorLF_SH_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_H_F,          false, false }, // 201  framebufQ2ColorHF_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_SPEC,         false, false }, // 203  framebufQ2ColorSpec_Sampled
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_S_H_PREV, false, false }, // 216 framebufQ2HistColorLF_SH_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS,      false, false }, // 239  framebufQ2GradSmplPos_Sampled
};

// CmQ2GradientAtrous: 4 storage images and no sampled image; the A4.5 pair fix reads all four
// through their storage images, so the measured item count is 4 instead of the pre-fix 8. The
// iteration travels by the 4-byte push constant.
constexpr uint32_t GRADIENT_ATROUS_BINDING_COUNT = 4;
constexpr ComposeBinding GRADIENT_ATROUS_BINDINGS[GRADIENT_ATROUS_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_GRAD_L_F_PING,      true, false }, // 111  framebufQ2GradLFPing
    { FB_IMAGE_INDEX_Q2_GRAD_L_F_PONG,      true, false }, // 112  framebufQ2GradLFPong
    { FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PING, true, false }, // 113  framebufQ2GradHFSpecPing
    { FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PONG, true, false }, // 114  framebufQ2GradHFSpecPong
};

// CmQ2Temporal: 9 storage images and 21 sampled images; `[numthreads(15, 15, 1)]`. It reads the
// `_PREV` histories and writes the current ones plus the a-trous inputs.
constexpr uint32_t TEMPORAL_BINDING_COUNT = 30;
constexpr ComposeBinding TEMPORAL_BINDINGS[TEMPORAL_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_S_H,      true, false }, //  91  framebufQ2HistColorLF_SH
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_C_O_C_G,  true, false }, //  93  framebufQ2HistColorLF_COCG
    { FB_IMAGE_INDEX_Q2_HIST_MOMENTS_H_F,        true, false }, //  97  framebufQ2HistMomentsHF
    { FB_IMAGE_INDEX_Q2_FILTERED_SPEC,           true, false }, //  99  framebufQ2FilteredSpec
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_S_H,     true, false }, // 101  framebufQ2AtrousPingLF_SH
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_C_O_C_G, true, false }, // 103  framebufQ2AtrousPingLF_COCG
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_H_F,         true, false }, // 105  framebufQ2AtrousPingHF
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_SPEC,        true, false }, // 107  framebufQ2AtrousPingSpec
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_MOMENTS,     true, false }, // 109  framebufQ2AtrousPingMoments
    { FB_IMAGE_INDEX_NORMAL,                     false, false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_NORMAL_PREV,                false, false }, // 128  framebufNormal_Prev_Sampled
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY,            false, false }, // 129  framebufNormalGeometry_Sampled
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY_PREV,       false, false }, // 130  framebufNormalGeometry_Prev_Sampled
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,         false, false }, // 131  framebufMetallicRoughness_Sampled
    { FB_IMAGE_INDEX_DEPTH_GRAD,                 false, false }, // 135  framebufDepthGrad_Sampled
    { FB_IMAGE_INDEX_MOTION,                     false, false }, // 137  framebufMotion_Sampled
    { FB_IMAGE_INDEX_THROUGHPUT,                 false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,           false, false }, // 197  framebufQ2ColorLF_SH_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_L_F_C_O_C_G,       false, false }, // 199  framebufQ2ColorLF_COCG_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_H_F,               false, false }, // 201  framebufQ2ColorHF_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR_SPEC,              false, false }, // 203  framebufQ2ColorSpec_Sampled
    { FB_IMAGE_INDEX_Q2_VIEW_DEPTH,              false, false }, // 205  framebufQ2ViewDepth_Sampled
    { FB_IMAGE_INDEX_Q2_VIEW_DEPTH_PREV,         false, false }, // 206  framebufQ2ViewDepth_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_S_H_PREV, false, false }, // 216  framebufQ2HistColorLF_SH_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_C_O_C_G_PREV, false, false }, // 218 framebufQ2HistColorLF_COCG_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_H_F_PREV,     false, false }, // 220  framebufQ2HistColorHF_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_HIST_MOMENTS_H_F_PREV,   false, false }, // 222  framebufQ2HistMomentsHF_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_FILTERED_SPEC_PREV,      false, false }, // 224  framebufQ2FilteredSpec_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_GRAD_L_F_PONG,           false, false }, // 236  framebufQ2GradLFPong_Sampled
    { FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PONG,      false, false }, // 238  framebufQ2GradHFSpecPong_Sampled
};

// CmQ2AtrousLF: 4 storage images and 3 sampled images; the A4.5 pair fix reads the ping/pong LF
// pair through its storage images, so the measured item count is 7 instead of the pre-fix 11. The
// same 4-byte push constant carries the iteration, dispatched at 1/3 resolution.
constexpr uint32_t ATROUS_LF_BINDING_COUNT = 7;
constexpr ComposeBinding ATROUS_LF_BINDINGS[ATROUS_LF_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_S_H,     true, false }, // 101  framebufQ2AtrousPingLF_SH
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_L_F_S_H,     true, false }, // 102  framebufQ2AtrousPongLF_SH
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_C_O_C_G, true, false }, // 103  framebufQ2AtrousPingLF_COCG
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_L_F_C_O_C_G, true, false }, // 104  framebufQ2AtrousPongLF_COCG
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY,            false, false }, // 129  framebufNormalGeometry_Sampled
    { FB_IMAGE_INDEX_DEPTH_GRAD,                 false, false }, // 135  framebufDepthGrad_Sampled
    { FB_IMAGE_INDEX_Q2_VIEW_DEPTH,              false, false }, // 205  framebufQ2ViewDepth_Sampled
};

// CmQ2Atrous: 8 storage images and 15 sampled images; the A4.5 pair fix reads the HF/SPEC/moments
// fields through their storage images, so the measured item count is 23 instead of the pre-fix 30.
// The iteration is the shader's specialization constant, so the module dispatches four
// specialized pipelines.
constexpr uint32_t ATROUS_BINDING_COUNT = 23;
constexpr ComposeBinding ATROUS_BINDINGS[ATROUS_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_H_F,          true, false }, //  95  framebufQ2HistColorHF
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_H_F,         true, false }, // 105  framebufQ2AtrousPingHF
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_H_F,         true, false }, // 106  framebufQ2AtrousPongHF
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_SPEC,        true, false }, // 107  framebufQ2AtrousPingSpec
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_SPEC,        true, false }, // 108  framebufQ2AtrousPongSpec
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_MOMENTS,     true, false }, // 109  framebufQ2AtrousPingMoments
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_MOMENTS,     true, false }, // 110  framebufQ2AtrousPongMoments
    { FB_IMAGE_INDEX_Q2_COLOR,                   true, false }, // 117  framebufQ2Color (iteration 3 composite)
    { FB_IMAGE_INDEX_ALBEDO,                     false, false }, // 124  framebufAlbedo_Sampled
    { FB_IMAGE_INDEX_IS_SKY,                     false, false }, // 126  framebufIsSky_Sampled
    { FB_IMAGE_INDEX_NORMAL,                     false, false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_NORMAL_GEOMETRY,            false, false }, // 129  framebufNormalGeometry_Sampled
    { FB_IMAGE_INDEX_METALLIC_ROUGHNESS,         false, false }, // 131  framebufMetallicRoughness_Sampled
    { FB_IMAGE_INDEX_DEPTH_GRAD,                 false, false }, // 135  framebufDepthGrad_Sampled
    { FB_IMAGE_INDEX_THROUGHPUT,                 false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_VIEW_DEPTH,              false, false }, // 205  framebufQ2ViewDepth_Sampled
    { FB_IMAGE_INDEX_Q2_TRANSPARENT,             false, false }, // 212  framebufQ2Transparent_Sampled
    { FB_IMAGE_INDEX_Q2_FOG_ACCUM,               false, false }, // 214  framebufQ2FogAccum_Sampled
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_S_H,      false, false }, // 215  framebufQ2HistColorLF_SH_Sampled
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_C_O_C_G,  false, false }, // 217  framebufQ2HistColorLF_COCG_Sampled
    { FB_IMAGE_INDEX_Q2_HIST_MOMENTS_H_F,        false, false }, // 221  framebufQ2HistMomentsHF_Sampled
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_S_H,     false, false }, // 225  framebufQ2AtrousPingLF_SH_Sampled
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_C_O_C_G, false, false }, // 227  framebufQ2AtrousPingLF_COCG_Sampled
};

// CmQ2Interleave: 1 storage image and 2 sampled images.
constexpr uint32_t INTERLEAVE_BINDING_COUNT = 3;
constexpr ComposeBinding INTERLEAVE_BINDINGS[INTERLEAVE_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_PRE_FINAL, true, false }, //  27  framebufPreFinal
    { FB_IMAGE_INDEX_THROUGHPUT, false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_Q2_COLOR,  false, false }, // 241  framebufQ2Color_Sampled
};

// CmLuminanceHistogram: 1 sampled image and no storage image - the exposure histogram only reads
// the interleave's PRE_FINAL (the 128-bin `groupshared` histogram is private to the workgroup).
constexpr uint32_t HISTOGRAM_BINDING_COUNT = 1;
constexpr ComposeBinding HISTOGRAM_BINDINGS[HISTOGRAM_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_PRE_FINAL, false, false }, // 151  framebufPreFinal_Sampled
};

// CmCheckerboard: 3 storage images and 4 sampled images.
constexpr uint32_t CHECKERBOARD_BINDING_COUNT = 7;
constexpr ComposeBinding CHECKERBOARD_BINDINGS[CHECKERBOARD_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_FINAL,            true, false }, //  28  framebufFinal
    { FB_IMAGE_INDEX_ACID_FOG,         true, false }, //  60  framebufAcidFog
    { FB_IMAGE_INDEX_SCREEN_EMISSION,  true, false }, //  62  framebufScreenEmission
    { FB_IMAGE_INDEX_THROUGHPUT,       false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_PRE_FINAL,        false, false }, // 151  framebufPreFinal_Sampled
    { FB_IMAGE_INDEX_ACID_FOG_R_T,     false, false }, // 183  framebufAcidFogRT_Sampled
    { FB_IMAGE_INDEX_SCREEN_EMIS_R_T,  false, false }, // 185  framebufScreenEmisRT_Sampled
};

// CmPrepareFinal: 5 storage images and 11 sampled images (measured: 17 set-0 items in the shipped
// blob, of which the FINAL sampled view at raw 152 is the pair the A4.4 S1 shader fix removes; see
// the class comment). Unlike the other four tables the UAVs come last: if a future edit ever binds
// one image both ways in this set, the last requirement applied would be the UAV's and the image
// would end the chain in GENERAL, which the engine's convention and the next frame's writes need;
// today no image is bound both ways here. The god-rays entry is the real union image 64, written by
// the god-rays filter before this module records (raw 188, `framebufGodRaysFiltered_Sampled`).
constexpr uint32_t PREPARE_FINAL_BINDING_COUNT = 16;
constexpr ComposeBinding PREPARE_FINAL_BINDINGS[PREPARE_FINAL_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_ALBEDO,               false, false }, // 124  framebufAlbedo_Sampled
    { FB_IMAGE_INDEX_NORMAL,               false, false }, // 127  framebufNormal_Sampled
    { FB_IMAGE_INDEX_DEPTH_WORLD,          false, false }, // 133  framebufDepthWorld_Sampled
    { FB_IMAGE_INDEX_MOTION,               false, false }, // 137  framebufMotion_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_DIRECT,    false, false }, // 138  framebufUnfilteredDirect_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_SPECULAR,  false, false }, // 139  framebufUnfilteredSpecular_Sampled
    { FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR, false, false }, // 149  framebufPrimaryToReflRefr_Sampled
    { FB_IMAGE_INDEX_THROUGHPUT,           false, false }, // 150  framebufThroughput_Sampled
    { FB_IMAGE_INDEX_ACID_FOG,             false, false }, // 184  framebufAcidFog_Sampled
    { FB_IMAGE_INDEX_SCREEN_EMISSION,      false, false }, // 186  framebufScreenEmission_Sampled
    { FB_IMAGE_INDEX_GOD_RAYS_FILTERED,    false, false }, // 188  framebufGodRaysFiltered_Sampled
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_R, true, false }, //  16  framebufUnfilteredIndirectSH_R
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_G, true, false }, //  17  framebufUnfilteredIndirectSH_G
    { FB_IMAGE_INDEX_UNFILTERED_INDIRECT_S_H_B, true, false }, //  18  framebufUnfilteredIndirectSH_B
    { FB_IMAGE_INDEX_FINAL,                true, false }, //  28  framebufFinal
    { FB_IMAGE_INDEX_BLOOM_INPUT,          true, false }, //  65  framebufBloomInput
};

// CmQ2TAAU: 2 storage images, 3 sampled images and the game's sampler for the history (image 120,
// raw 368). The dispatch runs over the upscaled size.
constexpr uint32_t TAAU_BINDING_COUNT = 6;
constexpr ComposeBinding TAAU_BINDINGS[TAAU_BINDING_COUNT] =
{
    { FB_IMAGE_INDEX_UPSCALED_PING,       true, false }, //  29  framebufUpscaledPing
    { FB_IMAGE_INDEX_Q2_TAA_HISTORY,      true, false }, // 119  framebufQ2TaaHistory
    { FB_IMAGE_INDEX_FINAL,               false, false }, // 152  framebufFinal_Sampled
    { FB_IMAGE_INDEX_MOTION_DLSS,         false, false }, // 155  framebufMotionDlss_Sampled
    { FB_IMAGE_INDEX_Q2_TAA_HISTORY_PREV, false, false }, // 244  framebufQ2TaaHistory_Prev_Sampled
    { FB_IMAGE_INDEX_Q2_TAA_HISTORY_PREV, false, true  }, // 368  framebufQ2TaaHistory_Prev_Sampler
};

// The engine raw binding of one entry: the storage-image array for the UAVs, the sampled array for
// the SRVs, the sampler array for the one sampler - the numbers the blobs' OpDecorate bindings
// carry.
uint32_t GetComposeRawBinding(const ComposeBinding &binding)
{
    if (binding.isSampler)
    {
        return ShFramebuffers_Sampler_Bindings[binding.image];
    }

    return binding.isUAV ? ShFramebuffers_Bindings[binding.image]
                         : ShFramebuffers_Sampled_Bindings[binding.image];
}

// The NVRHI slot of one entry under the module's set-0 offsets.
uint32_t GetComposeSlot(const ComposeBinding &binding)
{
    if (binding.isSampler)
    {
        return GetComposeRawBinding(binding) - FRAMEBUFFER_SAMPLER_OFFSET;
    }

    return binding.isUAV ? GetComposeRawBinding(binding) - FRAMEBUFFER_UAV_OFFSET
                         : GetComposeRawBinding(binding) - FRAMEBUFFER_SRV_OFFSET;
}

constexpr uint32_t COMPOSE_IMAGE_NONE = COMPOSE_IMAGE_COUNT;

// The union-table position of an engine image, or COMPOSE_IMAGE_NONE when the image is not part of
// the union (which the tables' static assertions rule out for every binding).
constexpr uint32_t FindComposeImage(FramebufferImageIndex image)
{
    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        if (COMPOSE_IMAGES[i].image == image)
        {
            return i;
        }
    }

    return COMPOSE_IMAGE_NONE;
}

// The extent the engine gives one union image at the passed resolution
// (`Framebuffers::GetFramebufSize`, Framebuffers.cpp:628-692); every wrap has to match it, because
// the shaders access the images at these sizes.
VkExtent2D GetComposeImageExtent(const ComposeImage &entry,
                                 uint32_t width, uint32_t height,
                                 uint32_t upscaledWidth, uint32_t upscaledHeight)
{
    switch (entry.size)
    {
        case ComposeImageSize::Half:
            return {std::max(1u, (width + 1) / 2), std::max(1u, (height + 1) / 2)};

        case ComposeImageSize::Third:
            return {std::max(1u, (width + 1) / COMPOSE_STRATA_SIZE),
                    std::max(1u, (height + 1) / COMPOSE_STRATA_SIZE)};

        case ComposeImageSize::Upscaled:
            return {upscaledWidth, upscaledHeight};

        case ComposeImageSize::Render:
        default:
            return {width, height};
    }
}

// The twelve pass tables have to agree with the union table item by item: every engine-bound entry
// has to resolve to an image the module wraps. The static assertion below runs at compile time, so
// a hand-edited table cannot produce a half-filled binding set at run time.
constexpr bool ComposeTablesAreConsistent()
{
    const ComposeBinding *const tables[] =
    {
        GRADIENT_REPROJECT_BINDINGS,
        ADAPTER_BINDINGS,
        GRADIENT_IMG_BINDINGS,
        GRADIENT_ATROUS_BINDINGS,
        TEMPORAL_BINDINGS,
        ATROUS_LF_BINDINGS,
        ATROUS_BINDINGS,
        INTERLEAVE_BINDINGS,
        HISTOGRAM_BINDINGS,
        CHECKERBOARD_BINDINGS,
        PREPARE_FINAL_BINDINGS,
        TAAU_BINDINGS,
    };
    const uint32_t counts[] =
    {
        GRADIENT_REPROJECT_BINDING_COUNT,
        ADAPTER_BINDING_COUNT,
        GRADIENT_IMG_BINDING_COUNT,
        GRADIENT_ATROUS_BINDING_COUNT,
        TEMPORAL_BINDING_COUNT,
        ATROUS_LF_BINDING_COUNT,
        ATROUS_BINDING_COUNT,
        INTERLEAVE_BINDING_COUNT,
        HISTOGRAM_BINDING_COUNT,
        CHECKERBOARD_BINDING_COUNT,
        PREPARE_FINAL_BINDING_COUNT,
        TAAU_BINDING_COUNT,
    };

    for (uint32_t table = 0; table < 12; table++)
    {
        for (uint32_t i = 0; i < counts[table]; i++)
        {
            if (FindComposeImage(tables[table][i].image) == COMPOSE_IMAGE_NONE)
            {
                return false;
            }
        }
    }

    return true;
}

constexpr bool AreComposeImagesDistinct()
{
    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        for (uint32_t j = i + 1; j < COMPOSE_IMAGE_COUNT; j++)
        {
            if (COMPOSE_IMAGES[i].image == COMPOSE_IMAGES[j].image)
            {
                return false;
            }
        }
    }

    return true;
}

// The images whose last use on the list was a sampled read, per path, plus the god-rays hand-off.
// The pass that reads an image last decides its state: NVRHI moves every sampled image of a bound
// set to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (vulkan-resource-bindings.cpp:398-435) and
// applies the requirements of every binding item, used by the shader or not
// (vulkan-state-tracking.cpp:29-63). The engine leaves every framebuffer image in
// VK_IMAGE_LAYOUT_GENERAL (= UnorderedAccess) and the next frame's passes write or sample through
// their own wraps, so each entry point has to move its own sampled reads back to UnorderedAccess
// (the class comment's state discipline). An image bound only as a storage image ends in GENERAL
// already and is not listed.
//
// The unfiltered path is A4.4's 18-image list plus the three god-rays images below. The filtered
// path adds the images the ASVGF chain reads last: the `_PREV` histories and the reproject's
// sampled inputs, the temporal and a-trous SRVs, and the adapter's four channel UAVs (73/75/77/79),
// which the temporal pass reads as sampled images after the adapter wrote them as storage images.
// The god-rays images are in both lists: 64 is sampled by the final composition in either path, so
// it is the image this module really moves to read-only and has to move back; 63/89 are bound by no
// compose set, and their requirement is a same-state UnorderedAccess barrier that names the
// god-rays module's hand-off (the class comment's contract), not a transition.
constexpr uint32_t COMPOSE_RESTORE_COUNT = 21;
constexpr FramebufferImageIndex COMPOSE_RESTORE_IMAGES[COMPOSE_RESTORE_COUNT] =
{
    FB_IMAGE_INDEX_ALBEDO,
    FB_IMAGE_INDEX_IS_SKY,
    FB_IMAGE_INDEX_NORMAL,
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS,
    FB_IMAGE_INDEX_DEPTH_WORLD,
    FB_IMAGE_INDEX_MOTION,
    FB_IMAGE_INDEX_UNFILTERED_DIRECT,
    FB_IMAGE_INDEX_UNFILTERED_SPECULAR,
    FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR,
    FB_IMAGE_INDEX_THROUGHPUT,
    FB_IMAGE_INDEX_PRE_FINAL,
    FB_IMAGE_INDEX_ACID_FOG_R_T,
    FB_IMAGE_INDEX_ACID_FOG,
    FB_IMAGE_INDEX_SCREEN_EMIS_R_T,
    FB_IMAGE_INDEX_SCREEN_EMISSION,
    FB_IMAGE_INDEX_GOD_RAYS,
    FB_IMAGE_INDEX_GOD_RAYS_FILTERED,
    FB_IMAGE_INDEX_Q2_TRANSPARENT,
    FB_IMAGE_INDEX_Q2_GOD_RAYS_THROUGHPUT_DIST,
    FB_IMAGE_INDEX_Q2_FOG_ACCUM,
    FB_IMAGE_INDEX_Q2_COLOR,
};

constexpr uint32_t CHAIN_RESTORE_COUNT = 54;
constexpr FramebufferImageIndex CHAIN_RESTORE_IMAGES[CHAIN_RESTORE_COUNT] =
{
    FB_IMAGE_INDEX_ALBEDO,
    FB_IMAGE_INDEX_ALBEDO_PREV,
    FB_IMAGE_INDEX_IS_SKY,
    FB_IMAGE_INDEX_NORMAL,
    FB_IMAGE_INDEX_NORMAL_PREV,
    FB_IMAGE_INDEX_NORMAL_GEOMETRY,
    FB_IMAGE_INDEX_NORMAL_GEOMETRY_PREV,
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS,
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS_PREV,
    FB_IMAGE_INDEX_DEPTH_WORLD,
    FB_IMAGE_INDEX_DEPTH_GRAD,
    FB_IMAGE_INDEX_MOTION,
    FB_IMAGE_INDEX_UNFILTERED_DIRECT,
    FB_IMAGE_INDEX_UNFILTERED_SPECULAR,
    FB_IMAGE_INDEX_SURFACE_POSITION_PREV,
    FB_IMAGE_INDEX_VIEW_DIRECTION_PREV,
    FB_IMAGE_INDEX_PRIMARY_TO_REFL_REFR,
    FB_IMAGE_INDEX_THROUGHPUT,
    FB_IMAGE_INDEX_PRE_FINAL,
    FB_IMAGE_INDEX_ACID_FOG_R_T,
    FB_IMAGE_INDEX_ACID_FOG,
    FB_IMAGE_INDEX_SCREEN_EMIS_R_T,
    FB_IMAGE_INDEX_SCREEN_EMISSION,
    FB_IMAGE_INDEX_GOD_RAYS,
    FB_IMAGE_INDEX_GOD_RAYS_FILTERED,
    FB_IMAGE_INDEX_Q2_COLOR_L_F_S_H,
    FB_IMAGE_INDEX_Q2_COLOR_L_F_C_O_C_G,
    FB_IMAGE_INDEX_Q2_COLOR_H_F,
    FB_IMAGE_INDEX_Q2_COLOR_H_F_PREV,
    FB_IMAGE_INDEX_Q2_COLOR_SPEC,
    FB_IMAGE_INDEX_Q2_COLOR_SPEC_PREV,
    FB_IMAGE_INDEX_Q2_VIEW_DEPTH,
    FB_IMAGE_INDEX_Q2_VIEW_DEPTH_PREV,
    FB_IMAGE_INDEX_Q2_BASE_COLOR_PREV,
    FB_IMAGE_INDEX_Q2_METALLIC_PREV,
    FB_IMAGE_INDEX_Q2_TRANSPARENT,
    FB_IMAGE_INDEX_Q2_GOD_RAYS_THROUGHPUT_DIST,
    FB_IMAGE_INDEX_Q2_FOG_ACCUM,
    FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_S_H,
    FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_S_H_PREV,
    FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_C_O_C_G,
    FB_IMAGE_INDEX_Q2_HIST_COLOR_L_F_C_O_C_G_PREV,
    FB_IMAGE_INDEX_Q2_HIST_COLOR_H_F_PREV,
    FB_IMAGE_INDEX_Q2_HIST_MOMENTS_H_F,
    FB_IMAGE_INDEX_Q2_HIST_MOMENTS_H_F_PREV,
    FB_IMAGE_INDEX_Q2_FILTERED_SPEC_PREV,
    FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_S_H,
    FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_C_O_C_G,
    FB_IMAGE_INDEX_Q2_GRAD_L_F_PONG,
    FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PONG,
    FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS,
    FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS_PREV,
    FB_IMAGE_INDEX_Q2_COLOR,
    FB_IMAGE_INDEX_Q2_RNG_SEED_PREV,
};

// The reproject's sampled reads and the TAAU's: each call is self-contained, so it restores what it
// read even when a later call of the frame is skipped.
constexpr uint32_t GRADIENT_REPROJECT_RESTORE_COUNT = 17;
constexpr FramebufferImageIndex GRADIENT_REPROJECT_RESTORE_IMAGES[GRADIENT_REPROJECT_RESTORE_COUNT] =
{
    FB_IMAGE_INDEX_ALBEDO_PREV,
    FB_IMAGE_INDEX_NORMAL_PREV,
    FB_IMAGE_INDEX_NORMAL_GEOMETRY,
    FB_IMAGE_INDEX_NORMAL_GEOMETRY_PREV,
    FB_IMAGE_INDEX_METALLIC_ROUGHNESS_PREV,
    FB_IMAGE_INDEX_MOTION,
    FB_IMAGE_INDEX_SURFACE_POSITION_PREV,
    FB_IMAGE_INDEX_VIEW_DIRECTION_PREV,
    FB_IMAGE_INDEX_THROUGHPUT,
    FB_IMAGE_INDEX_Q2_COLOR_H_F_PREV,
    FB_IMAGE_INDEX_Q2_COLOR_SPEC_PREV,
    FB_IMAGE_INDEX_Q2_VIEW_DEPTH,
    FB_IMAGE_INDEX_Q2_VIEW_DEPTH_PREV,
    FB_IMAGE_INDEX_Q2_BASE_COLOR_PREV,
    FB_IMAGE_INDEX_Q2_METALLIC_PREV,
    FB_IMAGE_INDEX_Q2_GRAD_SMPL_POS_PREV,
    FB_IMAGE_INDEX_Q2_RNG_SEED_PREV,
};

constexpr uint32_t TAAU_RESTORE_COUNT = 3;
constexpr FramebufferImageIndex TAAU_RESTORE_IMAGES[TAAU_RESTORE_COUNT] =
{
    FB_IMAGE_INDEX_FINAL,
    FB_IMAGE_INDEX_MOTION_DLSS,
    FB_IMAGE_INDEX_Q2_TAA_HISTORY_PREV,
};

// The images each iteration of an iterative pass writes, in the order the barrier requirement is
// queued after the dispatch. The last iteration is not listed: the next pass's set is a different
// object, so NVRHI re-applies its requirements by itself. The parity is the shader's: the gradient
// atrous and the LF atrous write the opposite ping of the field they read (even iterations write
// Pong); the HF atrous writes HistColorHF + PongSpec + PongMoments first, PingHF/Spec/Moments
// second, PongHF/Spec/Moments third and Q2_COLOR last (CmQ2GradientAtrous.comp.hlsl:144-162,
// CmQ2AtrousLF.comp.hlsl:295-313, CmQ2Atrous.comp.hlsl:305-330).
constexpr uint32_t GRADIENT_ATROUS_WRITE_COUNT = 2;
constexpr FramebufferImageIndex GRADIENT_ATROUS_WRITES[2][GRADIENT_ATROUS_WRITE_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_GRAD_L_F_PONG, FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PONG },
    { FB_IMAGE_INDEX_Q2_GRAD_L_F_PING, FB_IMAGE_INDEX_Q2_GRAD_H_F_SPEC_PING },
};

constexpr uint32_t ATROUS_LF_WRITE_COUNT = 2;
constexpr FramebufferImageIndex ATROUS_LF_WRITES[2][ATROUS_LF_WRITE_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_L_F_S_H, FB_IMAGE_INDEX_Q2_ATROUS_PONG_L_F_C_O_C_G },
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_S_H, FB_IMAGE_INDEX_Q2_ATROUS_PING_L_F_C_O_C_G },
};

constexpr uint32_t ATROUS_WRITE_COUNT = 3;
constexpr FramebufferImageIndex ATROUS_WRITES[COMPUTE_SVGF_ATROUS_ITERATION_COUNT - 1][ATROUS_WRITE_COUNT] =
{
    { FB_IMAGE_INDEX_Q2_HIST_COLOR_H_F,   FB_IMAGE_INDEX_Q2_ATROUS_PONG_SPEC, FB_IMAGE_INDEX_Q2_ATROUS_PONG_MOMENTS },
    { FB_IMAGE_INDEX_Q2_ATROUS_PING_H_F,  FB_IMAGE_INDEX_Q2_ATROUS_PING_SPEC, FB_IMAGE_INDEX_Q2_ATROUS_PING_MOMENTS },
    { FB_IMAGE_INDEX_Q2_ATROUS_PONG_H_F,  FB_IMAGE_INDEX_Q2_ATROUS_PONG_SPEC, FB_IMAGE_INDEX_Q2_ATROUS_PONG_MOMENTS },
};

constexpr bool AreRestoreImagesWrapped()
{
    for (uint32_t i = 0; i < GRADIENT_REPROJECT_RESTORE_COUNT; i++)
    {
        if (FindComposeImage(GRADIENT_REPROJECT_RESTORE_IMAGES[i]) == COMPOSE_IMAGE_NONE)
        {
            return false;
        }
    }

    for (uint32_t i = 0; i < TAAU_RESTORE_COUNT; i++)
    {
        if (FindComposeImage(TAAU_RESTORE_IMAGES[i]) == COMPOSE_IMAGE_NONE)
        {
            return false;
        }
    }

    for (uint32_t i = 0; i < COMPOSE_RESTORE_COUNT; i++)
    {
        if (FindComposeImage(COMPOSE_RESTORE_IMAGES[i]) == COMPOSE_IMAGE_NONE)
        {
            return false;
        }
    }

    for (uint32_t i = 0; i < CHAIN_RESTORE_COUNT; i++)
    {
        if (FindComposeImage(CHAIN_RESTORE_IMAGES[i]) == COMPOSE_IMAGE_NONE)
        {
            return false;
        }
    }

    return true;
}

static_assert(ComposeTablesAreConsistent(),
              "every compose binding has to resolve to a union image");
static_assert(AreComposeImagesDistinct(), "the compose image union must not repeat an image");
static_assert(AreRestoreImagesWrapped(), "every restored image has to be in the compose image union");

// The union positions of the two images the present can sample (GetFinalTexture,
// GetUpscaledTexture).
constexpr uint32_t FINAL_IMAGE_SLOT = FindComposeImage(FB_IMAGE_INDEX_FINAL);
static_assert(FINAL_IMAGE_SLOT != COMPOSE_IMAGE_NONE, "FINAL has to be part of the compose image union");

constexpr uint32_t UPSCALED_IMAGE_SLOT = FindComposeImage(FB_IMAGE_INDEX_UPSCALED_PING);
static_assert(UPSCALED_IMAGE_SLOT != COMPOSE_IMAGE_NONE,
              "UPSCALED_PING has to be part of the compose image union");

// The exact set-0 layout of one pass: Compute visibility (the RT passes' AllRayTracing layouts have
// no Compute bit and the pinned backend would skip them for a compute pipeline,
// validation-device.cpp:1002-1003), one item per declared raw binding, the block's offsets.
nvrhi::BindingLayoutHandle CreateFramebufferLayout(nvrhi::IDevice *device,
                                                   const ComposeBinding *pBindings,
                                                   uint32_t count)
{
    nvrhi::BindingLayoutDesc desc;
    desc.visibility = nvrhi::ShaderType::Compute;
    desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                               .setShaderResourceOffset(FRAMEBUFFER_SRV_OFFSET)
                               .setUnorderedAccessViewOffset(FRAMEBUFFER_UAV_OFFSET)
                               .setSamplerOffset(FRAMEBUFFER_SAMPLER_OFFSET));

    for (uint32_t i = 0; i < count; i++)
    {
        if (pBindings[i].isSampler)
        {
            desc.addItem(nvrhi::BindingLayoutItem::Sampler(GetComposeSlot(pBindings[i])));
        }
        else
        {
            desc.addItem(pBindings[i].isUAV
                             ? nvrhi::BindingLayoutItem::Texture_UAV(GetComposeSlot(pBindings[i]))
                             : nvrhi::BindingLayoutItem::Texture_SRV(GetComposeSlot(pBindings[i])));
        }
    }

    return device->createBindingLayout(desc);
}

// One compute pipeline over the pass's layouts, in descriptor-set order: the list order is the
// descriptor-set number the backend binds positionally (vulkan-resource-bindings.cpp:940-1019), so
// a pass's framebuffer set is first, the shared uniform set second, and the tonemapping, empty and
// volumetric sets follow where the pass declares them. The two push-constant pipelines add the
// module's push-constant layout last; no set is bound for it, which the pinned backend allows
// because the layout declares no descriptors (bindBindingSets walks the state's sets only,
// vulkan-resource-bindings.cpp:942-1019).
nvrhi::ComputePipelineHandle CreateComposePipeline(nvrhi::IDevice *device,
                                                   nvrhi::IShader *pShader,
                                                   std::initializer_list<nvrhi::IBindingLayout *> layouts)
{
    if (device == nullptr || pShader == nullptr || layouts.size() == 0)
    {
        return nullptr;
    }

    nvrhi::ComputePipelineDesc desc;
    desc.setComputeShader(pShader);

    for (nvrhi::IBindingLayout *pLayout : layouts)
    {
        if (pLayout == nullptr)
        {
            return nullptr;
        }

        desc.addBindingLayout(pLayout);
    }

    return rhi::createComputePipeline(device, desc, "RhiRtComposePass");
}

// The set over one exact layout. Every engine-bound entry takes the slot's wrap of that image; the
// sampler entry takes the module's TAAU history sampler. Returns null when an entry cannot be
// filled, which the module's static assertions make unreachable for the shipped tables.
nvrhi::BindingSetHandle CreateFramebufferSet(nvrhi::IDevice *device,
                                             const ComposeBinding *pBindings,
                                             uint32_t count,
                                             const nvrhi::TextureHandle *pEngineTextures,
                                             nvrhi::ISampler *pSampler,
                                             nvrhi::IBindingLayout *pLayout)
{
    if (device == nullptr || pLayout == nullptr)
    {
        return nullptr;
    }

    nvrhi::BindingSetDesc setDesc;

    for (uint32_t i = 0; i < count; i++)
    {
        if (pBindings[i].isSampler)
        {
            if (pSampler == nullptr)
            {
                return nullptr;
            }

            setDesc.addItem(nvrhi::BindingSetItem::Sampler(GetComposeSlot(pBindings[i]), pSampler));
            continue;
        }

        const uint32_t imageSlot = FindComposeImage(pBindings[i].image);
        nvrhi::ITexture *texture =
            imageSlot != COMPOSE_IMAGE_NONE ? pEngineTextures[imageSlot].Get() : nullptr;

        if (texture == nullptr)
        {
            return nullptr;
        }

        setDesc.addItem(pBindings[i].isUAV
                            ? nvrhi::BindingSetItem::Texture_UAV(GetComposeSlot(pBindings[i]), texture)
                            : nvrhi::BindingSetItem::Texture_SRV(GetComposeSlot(pBindings[i]), texture));
    }

    return device->createBindingSet(setDesc, pLayout);
}

void LogMessage(const RhiRtComposePass::PrintFunction &print, const std::string &message)
{
    if (print != nullptr)
    {
        print(message.c_str());
    }
}

}

RhiRtComposePass::RhiRtComposePass() = default;

RhiRtComposePass::~RhiRtComposePass()
{
    if (device != nullptr)
    {
        // The wraps reference engine images and buffers and the stand-ins are NVRHI resources; the
        // host destroys the pass while it can still idle the device (VulkanDevice does that before
        // the skeleton as well), so nothing has to go through a retire queue here.
        device->waitForIdle();
    }

    for (Target &target : targets)
    {
        target.gradientReprojectSet = nullptr;
        target.adapterSet = nullptr;
        target.gradientImgSet = nullptr;
        target.gradientAtrousSet = nullptr;
        target.temporalSet = nullptr;
        target.atrousLfSet = nullptr;
        target.atrousSet = nullptr;
        target.interleaveSet = nullptr;
        target.histogramSet = nullptr;
        target.checkerboardSet = nullptr;
        target.prepareFinalSet = nullptr;
        target.taauSet = nullptr;

        for (nvrhi::TextureHandle &texture : target.engineTextures)
        {
            texture = nullptr;
        }

        target.uniformSet = nullptr;
        target.uniformBuffer = nullptr;
        std::memset(target.imageHandles, 0, sizeof(target.imageHandles));
        target.width = 0;
        target.height = 0;
        target.upscaledWidth = 0;
        target.upscaledHeight = 0;
    }

    // The tonemapping wraps and their sets reference an engine buffer that outlives the pass; the
    // handles themselves are the module's.
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        tonemappingUavSets[i] = nullptr;
        tonemappingSrvSets[i] = nullptr;
        tonemappingBuffers[i] = nullptr;
    }

    volumetricSet = nullptr;
    emptySet = nullptr;
    volumetricDummySampler = nullptr;
    volumetricDummyTexture = nullptr;
    taauHistorySampler = nullptr;

    for (nvrhi::ComputePipelineHandle &pipeline : atrousPipelines)
    {
        pipeline = nullptr;
    }

    taauPipeline = nullptr;
    prepareFinalPipeline = nullptr;
    checkerboardPipeline = nullptr;
    averagePipeline = nullptr;
    histogramPipeline = nullptr;
    interleavePipeline = nullptr;
    atrousLfPipeline = nullptr;
    temporalPipeline = nullptr;
    gradientAtrousPipeline = nullptr;
    gradientImgPipeline = nullptr;
    adapterPipeline = nullptr;
    gradientReprojectPipeline = nullptr;

    volumetricLayout = nullptr;
    emptyLayout = nullptr;
    tonemappingSrvLayout = nullptr;
    tonemappingUavLayout = nullptr;
    pushConstantLayout = nullptr;
    uniformLayout = nullptr;
    taauFramebufferLayout = nullptr;
    prepareFinalFramebufferLayout = nullptr;
    checkerboardFramebufferLayout = nullptr;
    histogramFramebufferLayout = nullptr;
    interleaveFramebufferLayout = nullptr;
    atrousFramebufferLayout = nullptr;
    atrousLfFramebufferLayout = nullptr;
    temporalFramebufferLayout = nullptr;
    gradientAtrousFramebufferLayout = nullptr;
    gradientImgFramebufferLayout = nullptr;
    adapterFramebufferLayout = nullptr;
    gradientReprojectFramebufferLayout = nullptr;

    for (nvrhi::ShaderHandle &shader : atrousIterationShaders)
    {
        shader = nullptr;
    }

    taauShader = nullptr;
    prepareFinalShader = nullptr;
    checkerboardShader = nullptr;
    averageShader = nullptr;
    histogramShader = nullptr;
    interleaveShader = nullptr;
    atrousShader = nullptr;
    atrousLfShader = nullptr;
    temporalShader = nullptr;
    gradientAtrousShader = nullptr;
    gradientImgShader = nullptr;
    adapterShader = nullptr;
    gradientReprojectShader = nullptr;
}

bool RhiRtComposePass::Create(nvrhi::IDevice *pDevice,
                              rhi::RhiFrameContext *pFrameContext,
                              const Tonemapping *pTonemapping,
                              const char *pShaderFolderPath,
                              PrintFunction pfnPrint)
{
    if (created)
    {
        return true;
    }

    device = pDevice;
    print = std::move(pfnPrint);
    shaderFolderPath = pShaderFolderPath != nullptr ? pShaderFolderPath : "";
    frameContext = pFrameContext;

    if (device == nullptr)
    {
        LogMessage(print, "Warning: RHI: the compose pass needs an RHI device");
        return false;
    }

    if (frameContext == nullptr || !frameContext->IsCreated())
    {
        LogMessage(print, "Warning: RHI: the compose pass needs the frame context of the RHI layer");
        return false;
    }

    // The engine tonemapping object is what the exposure chain writes and the final composition
    // reads; without it the module has no ShTonemapping buffer to bind and the chain would produce
    // the NaN the class comment warns about, so a missing object makes the pass unusable.
    if (pTonemapping == nullptr)
    {
        LogMessage(print, "Warning: RHI: the compose pass needs the engine tonemapping object");
        return false;
    }

    // The widest pipeline declares three binding layouts (the two push-constant iterations), so the
    // pinned NVRHI's binding-layout cap does not matter here and the module has no
    // `c_MaxBindingLayouts` guard: the RT passes' twelve-layout pipelines are the only users of
    // the raised cap (third_party/nvrhi-max-binding-layouts.patch).

    if (!LoadShader(GRADIENT_REPROJECT_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, gradientReprojectShader) ||
        !LoadShader(ADAPTER_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, adapterShader) ||
        !LoadShader(GRADIENT_IMG_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, gradientImgShader) ||
        !LoadShader(GRADIENT_ATROUS_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, gradientAtrousShader) ||
        !LoadShader(TEMPORAL_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, temporalShader) ||
        !LoadShader(ATROUS_LF_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, atrousLfShader) ||
        !LoadShader(ATROUS_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, atrousShader) ||
        !LoadShader(INTERLEAVE_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, interleaveShader) ||
        !LoadShader(HISTOGRAM_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, histogramShader) ||
        !LoadShader(AVERAGE_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, averageShader) ||
        !LoadShader(CHECKERBOARD_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, checkerboardShader) ||
        !LoadShader(PREPARE_FINAL_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, prepareFinalShader) ||
        !LoadShader(TAAU_SHADER_FILE_NAME, nvrhi::ShaderType::Compute, taauShader))
    {
        return false;
    }

    // The twelve exact set-0 layouts: 27, 17, 8, 4, 30, 7, 23, 3, 1, 7, 16 and 6 items, Compute
    // visibility, SRV offset 124, UAV offset 0 and sampler offset 248. A union layout or a shared
    // set would fail the exact-set rule (validation-device.cpp:1855-1871), and the primary pass's
    // AllRayTracing uniform layout would leave the compute stage undeclared, so each pass owns its
    // own layout handle here.
    gradientReprojectFramebufferLayout =
        CreateFramebufferLayout(device, GRADIENT_REPROJECT_BINDINGS, GRADIENT_REPROJECT_BINDING_COUNT);
    adapterFramebufferLayout =
        CreateFramebufferLayout(device, ADAPTER_BINDINGS, ADAPTER_BINDING_COUNT);
    gradientImgFramebufferLayout =
        CreateFramebufferLayout(device, GRADIENT_IMG_BINDINGS, GRADIENT_IMG_BINDING_COUNT);
    gradientAtrousFramebufferLayout =
        CreateFramebufferLayout(device, GRADIENT_ATROUS_BINDINGS, GRADIENT_ATROUS_BINDING_COUNT);
    temporalFramebufferLayout =
        CreateFramebufferLayout(device, TEMPORAL_BINDINGS, TEMPORAL_BINDING_COUNT);
    atrousLfFramebufferLayout =
        CreateFramebufferLayout(device, ATROUS_LF_BINDINGS, ATROUS_LF_BINDING_COUNT);
    atrousFramebufferLayout =
        CreateFramebufferLayout(device, ATROUS_BINDINGS, ATROUS_BINDING_COUNT);
    interleaveFramebufferLayout =
        CreateFramebufferLayout(device, INTERLEAVE_BINDINGS, INTERLEAVE_BINDING_COUNT);
    histogramFramebufferLayout =
        CreateFramebufferLayout(device, HISTOGRAM_BINDINGS, HISTOGRAM_BINDING_COUNT);
    checkerboardFramebufferLayout =
        CreateFramebufferLayout(device, CHECKERBOARD_BINDINGS, CHECKERBOARD_BINDING_COUNT);
    prepareFinalFramebufferLayout =
        CreateFramebufferLayout(device, PREPARE_FINAL_BINDINGS, PREPARE_FINAL_BINDING_COUNT);
    taauFramebufferLayout =
        CreateFramebufferLayout(device, TAAU_BINDINGS, TAAU_BINDING_COUNT);

    // Set 1 for all thirteen passes: the engine's global uniform at raw binding 0
    // (ShaderCommonHLSL.hlsli:104 spells it with BINDING_GLOBAL_UNIFORM and DESC_SET_GLOBAL_UNIFORM).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setConstantBufferOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(BINDING_GLOBAL_UNIFORM));

        uniformLayout = device->createBindingLayout(desc);
    }

    // The two push-constant pipelines' last layout, and its only job is the 4-byte iteration index:
    // the pinned backend skips the item when it builds the Vulkan descriptor set layout and takes
    // the pipeline's single VkPushConstantRange from it (vulkan-resource-bindings.cpp:90-94,
    // :1110-1140), so the layout creates an empty descriptor set layout and no set is bound for it.
    // RhiSkyPass uses the same shape for its raster push constants (RhiSkyPass.cpp:408-421, :1516).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.addItem(nvrhi::BindingLayoutItem::PushConstants(0, COMPOSE_ITERATION_PUSH_SIZE));

        pushConstantLayout = device->createBindingLayout(desc);
    }

    // Set 2 of the histogram and the average: the engine's tonemapping buffer as
    // `RWStructuredBuffer<ShTonemapping>` at raw binding 0 (measured: `OpDecorate %tonemapping
    // DescriptorSet 2 / Binding 0`, `type_RWStructuredBuffer_ShTonemapping`, no NonWritable).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setUnorderedAccessViewOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(BINDING_LUM_HISTOGRAM));

        tonemappingUavLayout = device->createBindingLayout(desc);
    }

    // Set 2 of the final composition: the same buffer as a read-only
    // `StructuredBuffer<ShTonemapping>` (measured: `OpDecorate %tonemapping DescriptorSet 2 /
    // Binding 0`, `type_StructuredBuffer_ShTonemapping`, NonWritable on the member).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets().setShaderResourceOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(BINDING_LUM_HISTOGRAM));

        tonemappingSrvLayout = device->createBindingLayout(desc);
    }

    // The holes: the average declares no set 0 at all and the final composition's set 3 is the
    // LPM block the shipped blob does not reference (measured: no `DescriptorSet 3` decoration).
    // A real, zero-item layout keeps the positions of the uniform and tonemapping sets, and the
    // empty set below fills each position because the automatic-barrier pass dereferences every
    // entry (vulkan-state-tracking.cpp:105). Visibility still has to be set: the validation device
    // rejects a layout with visibility = None (validation-device.cpp:1340-1344).
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;

        emptyLayout = device->createBindingLayout(desc);
    }

    // Set 4: the sampled volumetric volume at raw binding 1 and its sampler at raw binding 2
    // (measured: `OpDecorate %g_volumetric_Sampled DescriptorSet 4 / Binding 1`, `Texture3D<float4>`
    // rgba16f, and `%g_volumetric_Sampler` at raw 2). The sampler offset of 0 keeps the slot as the
    // raw binding. The read is dead at runtime (see the volumetric dummy below), so the module binds
    // a 1x1x1 dummy instead of the engine's 160x88x64 image.
    {
        nvrhi::BindingLayoutDesc desc;
        desc.visibility = nvrhi::ShaderType::Compute;
        desc.setBindingOffsets(nvrhi::VulkanBindingOffsets()
                                   .setShaderResourceOffset(0)
                                   .setSamplerOffset(0));
        desc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(BINDING_VOLUMETRIC_SAMPLED));
        desc.addItem(nvrhi::BindingLayoutItem::Sampler(BINDING_VOLUMETRIC_SAMPLER));

        volumetricLayout = device->createBindingLayout(desc);
    }

    if (gradientReprojectFramebufferLayout == nullptr || adapterFramebufferLayout == nullptr ||
        gradientImgFramebufferLayout == nullptr || gradientAtrousFramebufferLayout == nullptr ||
        temporalFramebufferLayout == nullptr || atrousLfFramebufferLayout == nullptr ||
        atrousFramebufferLayout == nullptr || interleaveFramebufferLayout == nullptr ||
        histogramFramebufferLayout == nullptr || checkerboardFramebufferLayout == nullptr ||
        prepareFinalFramebufferLayout == nullptr || taauFramebufferLayout == nullptr ||
        uniformLayout == nullptr || pushConstantLayout == nullptr ||
        tonemappingUavLayout == nullptr || tonemappingSrvLayout == nullptr ||
        emptyLayout == nullptr || volumetricLayout == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a compose pass binding layout");
        return false;
    }

    // The module's nearest-filter sampler for the TAAU history: the engine binds its own nearest
    // sampler for image 120, whose flags carry no BILINEAR_SAMPLER bit (Framebuffers.cpp:813-815),
    // and the shader's bicubic Catmull-Rom taps are at texel centers - a linear sampler would blend
    // across texels (CmQ2TAAU.comp.hlsl:106-124, :221). Min/mag nearest, mip linear and clamp on
    // all axes mirror the engine's nearestSampler (Framebuffers.cpp:198-217).
    {
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.setMinFilter(false);
        samplerDesc.setMagFilter(false);
        samplerDesc.setMipFilter(true);
        samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Clamp);

        taauHistorySampler = rhi::createSampler(device, samplerDesc, "RhiRtComposePass TAAU history sampler");

        if (taauHistorySampler == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the compose pass TAAU history sampler");
            return false;
        }
    }

    // The HF atrous is the only pass whose iteration is a specialization constant (`SpecId 0`,
    // measured). Four derived shaders over the one base module, one per iteration (the legacy
    // rewrites the same VkSpecializationInfo per pipeline, Q2Denoiser.cpp:214-239); the derived
    // handles keep a reference to the base, which owns the module (vulkan-shader.cpp:63-79), so the
    // module keeps all five alive.
    for (uint32_t i = 0; i < COMPUTE_SVGF_ATROUS_ITERATION_COUNT; i++)
    {
        const nvrhi::ShaderSpecialization specialization =
            nvrhi::ShaderSpecialization::UInt32(0, i);

        atrousIterationShaders[i] =
            device->createShaderSpecialization(atrousShader.Get(), &specialization, 1);

        if (atrousIterationShaders[i] == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to specialize the compose pass HF atrous shader");
            return false;
        }
    }

    gradientReprojectPipeline = CreateComposePipeline(device, gradientReprojectShader,
                                                      { gradientReprojectFramebufferLayout, uniformLayout });
    adapterPipeline = CreateComposePipeline(device, adapterShader,
                                            { adapterFramebufferLayout, uniformLayout });
    gradientImgPipeline = CreateComposePipeline(device, gradientImgShader,
                                                { gradientImgFramebufferLayout, uniformLayout });
    gradientAtrousPipeline = CreateComposePipeline(device, gradientAtrousShader,
                                                   { gradientAtrousFramebufferLayout, uniformLayout,
                                                     pushConstantLayout });
    temporalPipeline = CreateComposePipeline(device, temporalShader,
                                             { temporalFramebufferLayout, uniformLayout });
    atrousLfPipeline = CreateComposePipeline(device, atrousLfShader,
                                             { atrousLfFramebufferLayout, uniformLayout,
                                               pushConstantLayout });

    for (uint32_t i = 0; i < COMPUTE_SVGF_ATROUS_ITERATION_COUNT; i++)
    {
        atrousPipelines[i] = CreateComposePipeline(device, atrousIterationShaders[i],
                                                   { atrousFramebufferLayout, uniformLayout });
    }

    interleavePipeline = CreateComposePipeline(device, interleaveShader,
                                               { interleaveFramebufferLayout, uniformLayout });
    histogramPipeline = CreateComposePipeline(device, histogramShader,
                                              { histogramFramebufferLayout, uniformLayout,
                                                tonemappingUavLayout });
    averagePipeline = CreateComposePipeline(device, averageShader,
                                            { emptyLayout, uniformLayout, tonemappingUavLayout });
    checkerboardPipeline = CreateComposePipeline(device, checkerboardShader,
                                                 { checkerboardFramebufferLayout, uniformLayout });
    prepareFinalPipeline = CreateComposePipeline(device, prepareFinalShader,
                                                 { prepareFinalFramebufferLayout, uniformLayout,
                                                   tonemappingSrvLayout, emptyLayout,
                                                   volumetricLayout });
    taauPipeline = CreateComposePipeline(device, taauShader,
                                         { taauFramebufferLayout, uniformLayout });

    if (gradientReprojectPipeline == nullptr || adapterPipeline == nullptr ||
        gradientImgPipeline == nullptr || gradientAtrousPipeline == nullptr ||
        temporalPipeline == nullptr || atrousLfPipeline == nullptr ||
        interleavePipeline == nullptr || histogramPipeline == nullptr ||
        averagePipeline == nullptr || checkerboardPipeline == nullptr ||
        prepareFinalPipeline == nullptr || taauPipeline == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create a compose pass compute pipeline");
        return false;
    }

    for (uint32_t i = 0; i < COMPUTE_SVGF_ATROUS_ITERATION_COUNT; i++)
    {
        if (atrousPipelines[i] == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create a compose pass HF atrous pipeline");
            return false;
        }
    }

    // The engine's per-slot `ShTonemapping` buffers as module-owned native wraps, and the two sets
    // over each. What the wrap has to declare, measured in the pinned NVRHI:
    //  - `structStride != 0` for either structured kind (validation-device.cpp:1693-1699), which the
    //    backend also asserts when the set is created (vulkan-resource-bindings.cpp:535-536);
    //  - `canHaveUAVs` for the StructuredBuffer_UAV of the exposure pair
    //    (validation-device.cpp:1709-1713);
    //  - `initialState = UnorderedAccess` with `keepInitialState = true`: the exposure pair writes
    //    the buffer on every list and the final composition reads it later in the same list, so the
    //    slot starts in the state the first UAV binding requires. A buffer has no image layout, so
    //    the choice only decides the direction of the memory-dependency barriers; the raster mode's
    //    separate SRV wrap (NvrhiFrameSkeleton::PrepareWorld) is never alive at the same time.
    // The engine buffer is created once in the Tonemapping constructor and `OnShaderReload` only
    // rebuilds its pipelines, so the wraps and the sets never have to follow a handle change.
    {
        const uint32_t elementSize = pTonemapping->GetElementSize();

        if (elementSize == 0)
        {
            LogMessage(print, "Warning: RHI: the engine tonemapping buffer has no element size");
            return false;
        }

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            const uint64_t bufferHandle =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pTonemapping->GetBuffer(i)));

            if (bufferHandle == 0)
            {
                LogMessage(print, "Warning: RHI: the engine tonemapping buffer is null");
                return false;
            }

            nvrhi::BufferDesc desc;
            desc.byteSize = elementSize;
            desc.structStride = elementSize;
            desc.canHaveUAVs = true;
            desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            desc.keepInitialState = true;
            desc.debugName = "RhiRtComposePass tonemapping wrap " + std::to_string(i);

            tonemappingBuffers[i] = device->createHandleForNativeBuffer(
                nvrhi::ObjectTypes::VK_Buffer, nvrhi::Object(bufferHandle), desc);

            if (tonemappingBuffers[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap the engine tonemapping buffer");
                return false;
            }

            nvrhi::BindingSetDesc uavSetDesc;
            uavSetDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(
                BINDING_LUM_HISTOGRAM, tonemappingBuffers[i]));
            tonemappingUavSets[i] = device->createBindingSet(uavSetDesc, tonemappingUavLayout);

            nvrhi::BindingSetDesc srvSetDesc;
            srvSetDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(
                BINDING_LUM_HISTOGRAM, tonemappingBuffers[i]));
            tonemappingSrvSets[i] = device->createBindingSet(srvSetDesc, tonemappingSrvLayout);

            if (tonemappingUavSets[i] == nullptr || tonemappingSrvSets[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to create a compose pass tonemapping binding set");
                return false;
            }
        }
    }

    // The one real empty set the average's position 0 and the final composition's position 3 bind.
    // One set bound at two positions is legal Vulkan; the layouts are the same object.
    emptySet = device->createBindingSet(nvrhi::BindingSetDesc(), emptyLayout);
    if (emptySet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the compose pass empty binding set");
        return false;
    }

    // The 1x1x1 volumetric dummy of set 4. `applyVolumetrics` returns before touching the volume
    // when `coreQ2RTX != 0` (CmPrepareFinal.comp.hlsl:269-280) and FillUniform forces that value
    // (VulkanDevice.cpp:330), so a one-texel RGBA16F volume is behaviourally exact; a real wrap of
    // the engine's 160x88x64 GENERAL-layout image would need a Volumetric accessor and a state
    // announcement for zero pixels. The dummy is cleared to zero once, on the first list that binds
    // it (Render): isUAV is what `clearTextureFloat`'s validation requires and the state below is
    // the compute-visibility SRV resting state, which keeps the cleared contents across lists.
    {
        nvrhi::TextureDesc desc;
        desc.dimension = nvrhi::TextureDimension::Texture3D;
        desc.width = 1;
        desc.height = 1;
        desc.depth = 1;
        desc.mipLevels = 1;
        desc.arraySize = 1;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.isShaderResource = true;
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::NonPixelShaderResource;
        desc.keepInitialState = true;
        desc.debugName = "RhiRtComposePass volumetric dummy (1x1x1)";

        volumetricDummyTexture = rhi::createTexture(device, desc, desc.debugName);
        volumetricDummySampler =
            rhi::createEngineTextureSampler(device, "RhiRtComposePass volumetric dummy sampler");

        if (volumetricDummyTexture == nullptr || volumetricDummySampler == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the compose pass volumetric dummy");
            return false;
        }

        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(BINDING_VOLUMETRIC_SAMPLED,
                                                           volumetricDummyTexture));
        setDesc.addItem(nvrhi::BindingSetItem::Sampler(BINDING_VOLUMETRIC_SAMPLER,
                                                       volumetricDummySampler));

        volumetricSet = device->createBindingSet(setDesc, volumetricLayout);
        if (volumetricSet == nullptr)
        {
            LogMessage(print, "Warning: RHI: failed to create the compose pass volumetric binding set");
            return false;
        }
    }

    created = true;
    return true;
}

RhiRtComposePass::Target *RhiRtComposePass::PrepareFrame(nvrhi::ICommandList *pCommandList,
                                                         uint32_t frameIndex,
                                                         const Framebuffers *pFramebuffers,
                                                         uint32_t width,
                                                         uint32_t height,
                                                         uint32_t upscaledWidth,
                                                         uint32_t upscaledHeight,
                                                         nvrhi::IBuffer *pUniformBuffer)
{
    if (!created || pCommandList == nullptr || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return nullptr;
    }

    if (width == 0 || height == 0 || upscaledWidth == 0 || upscaledHeight == 0)
    {
        return nullptr;
    }

    if (pFramebuffers == nullptr)
    {
        if (!warnedMissingFramebuffers)
        {
            warnedMissingFramebuffers = true;
            LogMessage(print, "Warning: RHI: the compose pass got no engine framebuffers, the compose is skipped");
        }
        return nullptr;
    }

    if (pUniformBuffer == nullptr)
    {
        if (!warnedMissingUniform)
        {
            warnedMissingUniform = true;
            LogMessage(print, "Warning: RHI: the compose pass got no global uniform, the compose is skipped");
        }
        return nullptr;
    }

    Target &target = targets[frameIndex];

    // The 80 images of this slot. `GetImageHandles` resolves the engine's ping-pong and history-role
    // swap (Framebuffers.cpp:33-53), so the handle of a `_Prev`-paired variable is the slot's
    // current image, which is the one the engine's own per-slot descriptor set binds to the
    // variable's fixed raw binding - the shader never sees the swap. The 4-tuple overload supplies
    // each image's extent, which every image here has to answer with the extent its size class
    // predicts (the generated flags and `Framebuffers::GetFramebufSize`); a different extent is a
    // wrongly sized wrap and the frame is skipped instead of sampling outside the image.
    const ResolutionState resolutionState = { width, height, upscaledWidth, upscaledHeight };

    uint64_t imageHandles[COMPOSE_IMAGE_COUNT] = {};
    uint64_t imageViews[COMPOSE_IMAGE_COUNT] = {};
    VkFormat imageFormats[COMPOSE_IMAGE_COUNT] = {};
    VkExtent2D imageExtents[COMPOSE_IMAGE_COUNT] = {};

    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        const auto [image, view, format, extent] =
            pFramebuffers->GetImageHandles(COMPOSE_IMAGES[i].image, frameIndex, resolutionState);

        imageHandles[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(image));
        imageViews[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(view));
        imageFormats[i] = format;
        imageExtents[i] = extent;

        if (imageHandles[i] == 0)
        {
            if (!warnedMissingFramebuffers)
            {
                warnedMissingFramebuffers = true;
                LogMessage(print, "Warning: RHI: the compose pass got no framebuffer image, the compose is skipped");
            }
            return nullptr;
        }

        const VkExtent2D expectedExtent =
            GetComposeImageExtent(COMPOSE_IMAGES[i], width, height, upscaledWidth, upscaledHeight);

        if (imageExtents[i].width != expectedExtent.width || imageExtents[i].height != expectedExtent.height)
        {
            if (!warnedUnexpectedSize)
            {
                warnedUnexpectedSize = true;
                LogMessage(print, std::string("Warning: RHI: the compose pass got \"") +
                                      ShFramebuffers_DebugNames[COMPOSE_IMAGES[i].image] +
                                      "\", which is not sized as the compose table expects; the compose is skipped");
            }
            return nullptr;
        }
    }

    bool framebuffersChanged = target.width != width || target.height != height ||
                               target.upscaledWidth != upscaledWidth ||
                               target.upscaledHeight != upscaledHeight;

    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        framebuffersChanged = framebuffersChanged || target.imageHandles[i] != imageHandles[i];
    }

    if (framebuffersChanged)
    {
        // The engine re-created its framebuffers (a resize or an image-format change) or the
        // resolution changed: the wraps and the sets over them are retired and rebuilt. Retiring
        // first is safe even if the rebuild below fails - the next frame retries, and the compose is
        // skipped until it succeeds.
        ReleaseFramebufferTarget(target);

        for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
        {
            const std::string debugName = std::string("RhiRtComposePass ") +
                                          ShFramebuffers_DebugNames[COMPOSE_IMAGES[i].image] +
                                          " frame " + std::to_string(frameIndex);

            target.engineTextures[i] = rhi::wrapEngineStorageImage(
                device, imageHandles[i], imageViews[i], imageFormats[i],
                imageExtents[i].width, imageExtents[i].height, debugName);

            if (target.engineTextures[i] == nullptr)
            {
                LogMessage(print, "Warning: RHI: failed to wrap a compose pass framebuffer image");
                ReleaseFramebufferTarget(target);
                return nullptr;
            }
        }

        std::memcpy(target.imageHandles, imageHandles, sizeof(target.imageHandles));
        target.width = width;
        target.height = height;
        target.upscaledWidth = upscaledWidth;
        target.upscaledHeight = upscaledHeight;
    }

    if (!PrepareFramebufferSets(target) || !PrepareUniformSet(target, pUniformBuffer))
    {
        return nullptr;
    }

    return &target;
}

void RhiRtComposePass::AnnounceFrameImages(nvrhi::ICommandList *pCommandList, const Target &target) const
{
    // The image state contract, spelled out on the class: the engine leaves every framebuffer image
    // in VK_IMAGE_LAYOUT_GENERAL - NVRHI's UnorderedAccess - and this native wrap keeps no state
    // between command lists (RhiTextureSource.h), so every list announces that state for all 80
    // images before the first use. The primary, direct, indirect and god-rays passes wrote or read
    // some of them on this very list through their own wraps; the physical layout they left them in
    // is exactly this GENERAL state - the god-rays module restores its sampled reads like every
    // other pass does (the class comment's hand-off contract) - so the announcement is the truth
    // and the SRV/UAV bindings' automatic transitions start from it.
    for (uint32_t i = 0; i < COMPOSE_IMAGE_COUNT; i++)
    {
        pCommandList->beginTrackingTextureState(
            target.engineTextures[i], nvrhi::AllSubresources, nvrhi::ResourceStates::UnorderedAccess);
    }
}

void RhiRtComposePass::RequireImageUnorderedAccess(nvrhi::ICommandList *pCommandList,
                                                   const Target &target,
                                                   FramebufferImageIndex image) const
{
    const uint32_t imageSlot = FindComposeImage(image);

    if (imageSlot != COMPOSE_IMAGE_NONE)
    {
        pCommandList->setTextureState(target.engineTextures[imageSlot], nvrhi::AllSubresources,
                                      nvrhi::ResourceStates::UnorderedAccess);
    }
}

void RhiRtComposePass::Render(nvrhi::ICommandList *pCommandList,
                              uint32_t frameIndex,
                              const Framebuffers *pFramebuffers,
                              uint32_t width,
                              uint32_t height,
                              uint32_t upscaledWidth,
                              uint32_t upscaledHeight,
                              bool filterEnabled,
                              nvrhi::IBuffer *pUniformBuffer)
{
    Target *pTarget = PrepareFrame(pCommandList, frameIndex, pFramebuffers,
                                   width, height, upscaledWidth, upscaledHeight, pUniformBuffer);
    if (pTarget == nullptr)
    {
        return;
    }

    Target &target = *pTarget;

    AnnounceFrameImages(pCommandList, target);

    // The volumetric dummy, cleared to zero once per pass lifetime on the first list that binds it
    // (`coreQ2RTX = 1` makes the shader's read unreachable, but the descriptor has to be valid and
    // a defined texel costs one 1x1 clear). clearTextureFloat moves it to CopyDest and the explicit
    // state call puts it back to the resting state of the compute-visibility SRV binding.
    if (!volumetricDummyCleared)
    {
        pCommandList->clearTextureFloat(volumetricDummyTexture, nvrhi::AllSubresources,
                                        nvrhi::Color(0.f, 0.f, 0.f, 0.f));
        pCommandList->setTextureState(volumetricDummyTexture, nvrhi::AllSubresources,
                                      nvrhi::ResourceStates::NonPixelShaderResource);
        volumetricDummyCleared = true;
    }

    // The legacy dispatch arithmetic (Q2Denoiser.cpp:344-345, :400-408, :472, :519, :593,
    // :606-607, ImageComposition.cpp:198-199, Tonemapping.cpp:176-179). `Utils::GetWorkGroupCount`
    // is `1 + ceil(size / 16)`, one workgroup beyond the ceiling; the shaders' bounds check on
    // `globalUniform.renderWidth/renderHeight` discards those invocations. The gradient and
    // LF-a-trous dispatches run over the third of the render size the strata cover.
    const uint32_t groupsX = Utils::GetWorkGroupCount(width, COMPOSE_GROUP_SIZE);
    const uint32_t groupsY = Utils::GetWorkGroupCount(height, COMPOSE_GROUP_SIZE);
    const uint32_t temporalGroupsX = Utils::GetWorkGroupCount(width, TEMPORAL_GROUP_SIZE);
    const uint32_t temporalGroupsY = Utils::GetWorkGroupCount(height, TEMPORAL_GROUP_SIZE);
    const uint32_t gradientGroupsX = Utils::GetWorkGroupCount(width / COMPOSE_STRATA_SIZE, COMPOSE_GROUP_SIZE);
    const uint32_t gradientGroupsY = Utils::GetWorkGroupCount(height / COMPOSE_STRATA_SIZE, COMPOSE_GROUP_SIZE);

    // The adapter always runs; it dispatches by the switch itself (CmQ2Adapter.comp.hlsl:193-213),
    // so it either composites the raw ReSTIR signal into Q2_COLOR or anti-fireflies it into the
    // ASVGF channel images the chain below filters (Q2Denoiser.cpp:410-426).
    RecordDispatch(pCommandList, adapterPipeline, { target.adapterSet, target.uniformSet },
                   groupsX, groupsY, 1);

    if (filterEnabled)
    {
        // The ASVGF chain of the legacy `Denoise` (Q2Denoiser.cpp:438-597), between the adapter and
        // the interleave. Its first consumer of the temporal result is the LF a-trous, and the HF
        // a-trous composite writes Q2_COLOR, which the interleave then resolves.
        RecordDispatch(pCommandList, gradientImgPipeline,
                       { target.gradientImgSet, target.uniformSet },
                       gradientGroupsX, gradientGroupsY, 1);

        // Seven gradient a-trous iterations at 1/3 resolution, the iteration in the 4-byte push
        // constant. The writes of each dispatch are required in UnorderedAccess, which is the
        // same-state UAV barrier of the next iteration's read (see the class comment); the last
        // iteration's write is ordered by the temporal set's own transition.
        for (uint32_t i = 0; i < GRADIENT_ATROUS_ITERATION_COUNT; i++)
        {
            RecordDispatch(pCommandList, gradientAtrousPipeline,
                           { target.gradientAtrousSet, target.uniformSet },
                           gradientGroupsX, gradientGroupsY, 1, &i);

            if (i + 1 < GRADIENT_ATROUS_ITERATION_COUNT)
            {
                const FramebufferImageIndex *const writes = GRADIENT_ATROUS_WRITES[i % 2];

                for (uint32_t j = 0; j < GRADIENT_ATROUS_WRITE_COUNT; j++)
                {
                    RequireImageUnorderedAccess(pCommandList, target, writes[j]);
                }
            }
        }

        // The temporal accumulation over the frame's render size, `[numthreads(15, 15, 1)]`.
        RecordDispatch(pCommandList, temporalPipeline, { target.temporalSet, target.uniformSet },
                       temporalGroupsX, temporalGroupsY, 1);

        // Four LF a-trous iterations at 1/3 resolution (push-constant iteration).
        for (uint32_t i = 0; i < COMPUTE_SVGF_ATROUS_ITERATION_COUNT; i++)
        {
            RecordDispatch(pCommandList, atrousLfPipeline,
                           { target.atrousLfSet, target.uniformSet },
                           gradientGroupsX, gradientGroupsY, 1, &i);

            if (i + 1 < COMPUTE_SVGF_ATROUS_ITERATION_COUNT)
            {
                const FramebufferImageIndex *const writes = ATROUS_LF_WRITES[i % 2];

                for (uint32_t j = 0; j < ATROUS_LF_WRITE_COUNT; j++)
                {
                    RequireImageUnorderedAccess(pCommandList, target, writes[j]);
                }
            }
        }

        // Four HF/SPEC a-trous iterations over the render size, the iteration baked into each
        // pipeline's specialization; the fourth composites Q2_COLOR.
        for (uint32_t i = 0; i < COMPUTE_SVGF_ATROUS_ITERATION_COUNT; i++)
        {
            RecordDispatch(pCommandList, atrousPipelines[i],
                           { target.atrousSet, target.uniformSet }, groupsX, groupsY, 1);

            if (i < COMPUTE_SVGF_ATROUS_ITERATION_COUNT - 1)
            {
                const FramebufferImageIndex *const writes = ATROUS_WRITES[i];

                for (uint32_t j = 0; j < ATROUS_WRITE_COUNT; j++)
                {
                    RequireImageUnorderedAccess(pCommandList, target, writes[j]);
                }
            }
        }
    }

    // The interleave resolves the checkerboard-packed Q2_COLOR into PRE_FINAL, which the exposure
    // pair then histograms and averages into the slot's tonemapping buffer. In the unfiltered branch
    // it follows the adapter directly (the legacy `InterleaveCheckerboard` short-circuit,
    // Q2Denoiser.cpp:428-436); in the filtered one it follows the composite.
    RecordDispatch(pCommandList, interleavePipeline, { target.interleaveSet, target.uniformSet },
                   groupsX, groupsY, 1);

    // The exposure pair on the fresh PRE_FINAL and the slot's tonemapping buffer: the histogram
    // over the render resolution, then the average over exactly the 128 bins in one workgroup
    // (Tonemapping.cpp:171-208).
    RecordDispatch(pCommandList, histogramPipeline,
                   { target.histogramSet, target.uniformSet, tonemappingUavSets[frameIndex] },
                   groupsX, groupsY, 1);
    RecordDispatch(pCommandList, averagePipeline,
                   { emptySet, target.uniformSet, tonemappingUavSets[frameIndex] },
                   1, 1, 1);

    RecordDispatch(pCommandList, checkerboardPipeline, { target.checkerboardSet, target.uniformSet },
                   groupsX, groupsY, 1);

    // The final composition: set 3 is the dead LPM hole, set 4 the volumetric dummy, and the
    // tonemapping buffer is bound read-only, after the exposure pair wrote it in this list.
    RecordDispatch(pCommandList, prepareFinalPipeline,
                   { target.prepareFinalSet, target.uniformSet, tonemappingSrvSets[frameIndex],
                     emptySet, volumetricSet },
                   groupsX, groupsY, 1);

    // The images whose last use was a sampled read rest in the read-only layout now; move them back
    // to UnorderedAccess, the engine's GENERAL, so the next frame's passes start from the state
    // their own announcements claim and their UAV writes are legal. The unfiltered path restores
    // A4.4's list plus the god-rays images (which includes 27 and 117: without them the next
    // frame's interleave and adapter UAV writes would run against a read-only image); the filtered
    // path restores the chain's list as well. 64 is the god-rays image this module samples; 63/89
    // end in a same-state requirement that only names the god-rays module's hand-off (the restore
    // lists' comment).
    if (filterEnabled)
    {
        for (uint32_t i = 0; i < CHAIN_RESTORE_COUNT; i++)
        {
            RequireImageUnorderedAccess(pCommandList, target, CHAIN_RESTORE_IMAGES[i]);
        }
    }
    else
    {
        for (uint32_t i = 0; i < COMPOSE_RESTORE_COUNT; i++)
        {
            RequireImageUnorderedAccess(pCommandList, target, COMPOSE_RESTORE_IMAGES[i]);
        }
    }
}

void RhiRtComposePass::RenderGradientReproject(nvrhi::ICommandList *pCommandList,
                                               uint32_t frameIndex,
                                               const Framebuffers *pFramebuffers,
                                               uint32_t width,
                                               uint32_t height,
                                               uint32_t upscaledWidth,
                                               uint32_t upscaledHeight,
                                               nvrhi::IBuffer *pUniformBuffer)
{
    Target *pTarget = PrepareFrame(pCommandList, frameIndex, pFramebuffers,
                                   width, height, upscaledWidth, upscaledHeight, pUniformBuffer);
    if (pTarget == nullptr)
    {
        return;
    }

    Target &target = *pTarget;

    AnnounceFrameImages(pCommandList, target);

    // The legacy dispatch over the third of the render size (Q2Denoiser.cpp:344-345): the shader
    // bounds itself with `renderWidth / Q2_GRAD_DWN` (CmQ2GradientReproject.comp.hlsl:64-70).
    const uint32_t groupsX = Utils::GetWorkGroupCount(width / COMPOSE_STRATA_SIZE, COMPOSE_GROUP_SIZE);
    const uint32_t groupsY = Utils::GetWorkGroupCount(height / COMPOSE_STRATA_SIZE, COMPOSE_GROUP_SIZE);

    RecordDispatch(pCommandList, gradientReprojectPipeline,
                   { target.gradientReprojectSet, target.uniformSet }, groupsX, groupsY, 1);

    // The reproject reads 17 images as sampled views and writes 10 as storage images; the sampled
    // ones end in the read-only layout, so they go back to UnorderedAccess before the direct pass's
    // own announcements (which claim GENERAL) and the next frame's uses. The writes already rest in
    // GENERAL.
    for (uint32_t i = 0; i < GRADIENT_REPROJECT_RESTORE_COUNT; i++)
    {
        RequireImageUnorderedAccess(pCommandList, target, GRADIENT_REPROJECT_RESTORE_IMAGES[i]);
    }
}

void RhiRtComposePass::RenderTaaU(nvrhi::ICommandList *pCommandList,
                                  uint32_t frameIndex,
                                  const Framebuffers *pFramebuffers,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t upscaledWidth,
                                  uint32_t upscaledHeight,
                                  nvrhi::IBuffer *pUniformBuffer)
{
    Target *pTarget = PrepareFrame(pCommandList, frameIndex, pFramebuffers,
                                   width, height, upscaledWidth, upscaledHeight, pUniformBuffer);
    if (pTarget == nullptr)
    {
        return;
    }

    Target &target = *pTarget;

    AnnounceFrameImages(pCommandList, target);

    // The dispatch runs over the upscaled size; the shader maps the output pixel to the render
    // resolution with the ratio of the two uniform sizes (CmQ2TAAU.comp.hlsl:134-150,
    // Q2Denoiser.cpp:652-653).
    const uint32_t groupsX = Utils::GetWorkGroupCount(upscaledWidth, COMPOSE_GROUP_SIZE);
    const uint32_t groupsY = Utils::GetWorkGroupCount(upscaledHeight, COMPOSE_GROUP_SIZE);

    RecordDispatch(pCommandList, taauPipeline, { target.taauSet, target.uniformSet },
                   groupsX, groupsY, 1);

    // The TAAU reads FINAL, MOTION_DLSS and the previous history as sampled images; restore them to
    // UnorderedAccess so the next frame's checkerboard/primary writes and this frame's present find
    // the engine's GENERAL layout.
    for (uint32_t i = 0; i < TAAU_RESTORE_COUNT; i++)
    {
        RequireImageUnorderedAccess(pCommandList, target, TAAU_RESTORE_IMAGES[i]);
    }
}

nvrhi::ITexture *RhiRtComposePass::GetFinalTexture(uint32_t frameIndex) const
{
    if (!created || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return nullptr;
    }

    return targets[frameIndex].engineTextures[FINAL_IMAGE_SLOT].Get();
}

nvrhi::ITexture *RhiRtComposePass::GetUpscaledTexture(uint32_t frameIndex) const
{
    if (!created || frameIndex >= MAX_FRAMES_IN_FLIGHT)
    {
        return nullptr;
    }

    return targets[frameIndex].engineTextures[UPSCALED_IMAGE_SLOT].Get();
}

void RhiRtComposePass::ReleaseTargets()
{
    for (Target &target : targets)
    {
        ReleaseTarget(target);
    }
}

void RhiRtComposePass::ReleaseFramebufferSets(Target &target)
{
    // Anything a recorded list may still reference has to go through the frame context's retire
    // queue: the sets reference the wraps of engine images the GPU may still be reading. The queue
    // takes its reference now, so the handles below can be cleared immediately.
    if (frameContext != nullptr)
    {
        if (target.gradientReprojectSet != nullptr)
        {
            frameContext->Retire(target.gradientReprojectSet);
        }
        if (target.adapterSet != nullptr)
        {
            frameContext->Retire(target.adapterSet);
        }
        if (target.gradientImgSet != nullptr)
        {
            frameContext->Retire(target.gradientImgSet);
        }
        if (target.gradientAtrousSet != nullptr)
        {
            frameContext->Retire(target.gradientAtrousSet);
        }
        if (target.temporalSet != nullptr)
        {
            frameContext->Retire(target.temporalSet);
        }
        if (target.atrousLfSet != nullptr)
        {
            frameContext->Retire(target.atrousLfSet);
        }
        if (target.atrousSet != nullptr)
        {
            frameContext->Retire(target.atrousSet);
        }
        if (target.interleaveSet != nullptr)
        {
            frameContext->Retire(target.interleaveSet);
        }
        if (target.histogramSet != nullptr)
        {
            frameContext->Retire(target.histogramSet);
        }
        if (target.checkerboardSet != nullptr)
        {
            frameContext->Retire(target.checkerboardSet);
        }
        if (target.prepareFinalSet != nullptr)
        {
            frameContext->Retire(target.prepareFinalSet);
        }
        if (target.taauSet != nullptr)
        {
            frameContext->Retire(target.taauSet);
        }
    }

    target.gradientReprojectSet = nullptr;
    target.adapterSet = nullptr;
    target.gradientImgSet = nullptr;
    target.gradientAtrousSet = nullptr;
    target.temporalSet = nullptr;
    target.atrousLfSet = nullptr;
    target.atrousSet = nullptr;
    target.interleaveSet = nullptr;
    target.histogramSet = nullptr;
    target.checkerboardSet = nullptr;
    target.prepareFinalSet = nullptr;
    target.taauSet = nullptr;
}

void RhiRtComposePass::ReleaseFramebufferTarget(Target &target)
{
    ReleaseFramebufferSets(target);

    if (frameContext != nullptr)
    {
        for (nvrhi::TextureHandle &texture : target.engineTextures)
        {
            if (texture != nullptr)
            {
                frameContext->Retire(texture);
            }
        }
    }

    for (nvrhi::TextureHandle &texture : target.engineTextures)
    {
        texture = nullptr;
    }

    std::memset(target.imageHandles, 0, sizeof(target.imageHandles));
    target.width = 0;
    target.height = 0;
    target.upscaledWidth = 0;
    target.upscaledHeight = 0;
}

void RhiRtComposePass::ReleaseTarget(Target &target)
{
    ReleaseFramebufferTarget(target);

    if (frameContext != nullptr && target.uniformSet != nullptr)
    {
        frameContext->Retire(target.uniformSet);
    }

    target.uniformSet = nullptr;
    target.uniformBuffer = nullptr;
}

bool RhiRtComposePass::PrepareFramebufferSets(Target &target)
{
    nvrhi::ISampler *const taauSampler = taauHistorySampler.Get();

    if (target.gradientReprojectSet == nullptr)
    {
        target.gradientReprojectSet = CreateFramebufferSet(
            device, GRADIENT_REPROJECT_BINDINGS, GRADIENT_REPROJECT_BINDING_COUNT,
            target.engineTextures, nullptr, gradientReprojectFramebufferLayout);

        if (target.gradientReprojectSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass gradient reproject binding set");
            }
            return false;
        }
    }

    if (target.adapterSet == nullptr)
    {
        target.adapterSet = CreateFramebufferSet(device, ADAPTER_BINDINGS, ADAPTER_BINDING_COUNT,
                                                 target.engineTextures, nullptr,
                                                 adapterFramebufferLayout);

        if (target.adapterSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass adapter binding set");
            }
            return false;
        }
    }

    if (target.gradientImgSet == nullptr)
    {
        target.gradientImgSet = CreateFramebufferSet(device, GRADIENT_IMG_BINDINGS, GRADIENT_IMG_BINDING_COUNT,
                                                     target.engineTextures, nullptr,
                                                     gradientImgFramebufferLayout);

        if (target.gradientImgSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass gradient image binding set");
            }
            return false;
        }
    }

    if (target.gradientAtrousSet == nullptr)
    {
        target.gradientAtrousSet = CreateFramebufferSet(device, GRADIENT_ATROUS_BINDINGS, GRADIENT_ATROUS_BINDING_COUNT,
                                                        target.engineTextures, nullptr,
                                                        gradientAtrousFramebufferLayout);

        if (target.gradientAtrousSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass gradient atrous binding set");
            }
            return false;
        }
    }

    if (target.temporalSet == nullptr)
    {
        target.temporalSet = CreateFramebufferSet(device, TEMPORAL_BINDINGS, TEMPORAL_BINDING_COUNT,
                                                  target.engineTextures, nullptr,
                                                  temporalFramebufferLayout);

        if (target.temporalSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass temporal binding set");
            }
            return false;
        }
    }

    if (target.atrousLfSet == nullptr)
    {
        target.atrousLfSet = CreateFramebufferSet(device, ATROUS_LF_BINDINGS, ATROUS_LF_BINDING_COUNT,
                                                  target.engineTextures, nullptr,
                                                  atrousLfFramebufferLayout);

        if (target.atrousLfSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass LF atrous binding set");
            }
            return false;
        }
    }

    if (target.atrousSet == nullptr)
    {
        target.atrousSet = CreateFramebufferSet(device, ATROUS_BINDINGS, ATROUS_BINDING_COUNT,
                                                target.engineTextures, nullptr,
                                                atrousFramebufferLayout);

        if (target.atrousSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass HF atrous binding set");
            }
            return false;
        }
    }

    if (target.interleaveSet == nullptr)
    {
        target.interleaveSet = CreateFramebufferSet(device, INTERLEAVE_BINDINGS, INTERLEAVE_BINDING_COUNT,
                                                    target.engineTextures, nullptr,
                                                    interleaveFramebufferLayout);

        if (target.interleaveSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass interleave binding set");
            }
            return false;
        }
    }

    if (target.histogramSet == nullptr)
    {
        target.histogramSet = CreateFramebufferSet(device, HISTOGRAM_BINDINGS, HISTOGRAM_BINDING_COUNT,
                                                   target.engineTextures, nullptr,
                                                   histogramFramebufferLayout);

        if (target.histogramSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass histogram binding set");
            }
            return false;
        }
    }

    if (target.checkerboardSet == nullptr)
    {
        target.checkerboardSet = CreateFramebufferSet(device, CHECKERBOARD_BINDINGS, CHECKERBOARD_BINDING_COUNT,
                                                      target.engineTextures, nullptr,
                                                      checkerboardFramebufferLayout);

        if (target.checkerboardSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass checkerboard binding set");
            }
            return false;
        }
    }

    if (target.prepareFinalSet == nullptr)
    {
        target.prepareFinalSet = CreateFramebufferSet(device, PREPARE_FINAL_BINDINGS, PREPARE_FINAL_BINDING_COUNT,
                                                      target.engineTextures, nullptr,
                                                      prepareFinalFramebufferLayout);

        if (target.prepareFinalSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass prepare-final binding set");
            }
            return false;
        }
    }

    if (target.taauSet == nullptr)
    {
        target.taauSet = CreateFramebufferSet(device, TAAU_BINDINGS, TAAU_BINDING_COUNT,
                                              target.engineTextures, taauSampler,
                                              taauFramebufferLayout);

        if (target.taauSet == nullptr)
        {
            if (!warnedBadTable)
            {
                warnedBadTable = true;
                LogMessage(print, "Warning: RHI: failed to create the compose pass TAAU binding set");
            }
            return false;
        }
    }

    return true;
}

bool RhiRtComposePass::PrepareUniformSet(Target &target, nvrhi::IBuffer *pUniformBuffer)
{
    if (target.uniformSet != nullptr && target.uniformBuffer == pUniformBuffer)
    {
        return true;
    }

    // The shader declares set 1 as ConstantBuffer<ShGlobalUniform>; the validation device refuses
    // such a binding on a buffer whose desc lacks isConstantBuffer (validation-device.cpp:1717-1723)
    // and on a volatile one (:1725-1730, which would also become a dynamic-offset binding the
    // static layout item cannot take).
    if (!pUniformBuffer->getDesc().isConstantBuffer || pUniformBuffer->getDesc().isVolatile)
    {
        if (!warnedBadUniform)
        {
            warnedBadUniform = true;
            LogMessage(print, "Warning: RHI: the compose pass needs the global uniform as a static constant-buffer wrap");
        }
        return false;
    }

    if (target.uniformSet != nullptr)
    {
        frameContext->Retire(target.uniformSet);
    }
    target.uniformSet = nullptr;

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::ConstantBuffer(BINDING_GLOBAL_UNIFORM, pUniformBuffer));

    target.uniformSet = device->createBindingSet(setDesc, uniformLayout);

    if (target.uniformSet == nullptr)
    {
        LogMessage(print, "Warning: RHI: failed to create the compose pass uniform binding set");
        return false;
    }

    target.uniformBuffer = pUniformBuffer;
    return true;
}

void RhiRtComposePass::RecordDispatch(nvrhi::ICommandList *pCommandList,
                                      nvrhi::IComputePipeline *pPipeline,
                                      std::initializer_list<nvrhi::IBindingSet *> sets,
                                      uint32_t groupsX,
                                      uint32_t groupsY,
                                      uint32_t groupsZ,
                                      const uint32_t *pPushConstant)
{
    nvrhi::ComputeState state;
    state.setPipeline(pPipeline);

    for (nvrhi::IBindingSet *pSet : sets)
    {
        state.addBindingSet(pSet);
    }

    pCommandList->setComputeState(state);

    // The iteration index of the two push-constant filters. The write needs the pipeline's layout,
    // which `setComputeState` just installed; the state cached by the backend keeps it until the
    // next state set (vulkan-compute.cpp:142-145), so the dispatch below sees the same layout.
    if (pPushConstant != nullptr)
    {
        pCommandList->setPushConstants(pPushConstant, sizeof(uint32_t));
    }

    pCommandList->dispatch(groupsX, groupsY, groupsZ);
}

bool RhiRtComposePass::LoadShader(const char *pFileName, nvrhi::ShaderType type, nvrhi::ShaderHandle &result)
{
    const std::string path = shaderFolderPath + pFileName;

    // The helper stays silent about a missing or unreadable blob, so that this class keeps its own
    // warning and its 'created == false' path (RhiPipeline.h).
    result = rhi::loadShader(device, path, type, pFileName);
    if (result == nullptr)
    {
        LogMessage(print, "Warning: RHI: cannot load the compose pass shader \"" + path + "\"");
        return false;
    }

    return true;
}
