// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of asvgf.c from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
// which is distributed under the terms of the GNU General Public License
// version 2.  It has been adapted to the renderer interface of this project.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//

#pragma once

#include "ASManager.h"
#include "Framebuffers.h"
#include "GlobalUniform.h"
#include "IShaderDependency.h"
#include "ShaderManager.h"

namespace vkpt
{

// New Q2RTX-style denoiser: full ASVGF (gradient atrous, temporal, LF atrous at
// CmQ2Adapter.comp, followed by checkerboard interleave into PreFinal.
//
// The ASVGF uses the Q2RTX channel format (LF luma-SH YCoCg, HF/SPEC packed
// (DISPingGradient), which is computed here by running the legacy ASVGF
// gradient atrous pass.
class Q2Denoiser : public IShaderDependency
{
public:
    Q2Denoiser(
        VkDevice device,
        std::shared_ptr<Framebuffers> framebuffers,
        const std::shared_ptr<const ShaderManager> &shaderManager,
        const std::shared_ptr<const GlobalUniform> &uniform,
        const std::shared_ptr<const ASManager> &asManager);
    ~Q2Denoiser();

    Q2Denoiser(const Q2Denoiser &other) = delete;
    Q2Denoiser(Q2Denoiser &&other) noexcept = delete;
    Q2Denoiser &operator=(const Q2Denoiser &other) = delete;
    Q2Denoiser &operator=(Q2Denoiser &&other) noexcept = delete;

    void Denoise(
        VkCommandBuffer cmd, uint32_t frameIndex,
        const std::shared_ptr<const GlobalUniform> &uniform);
    void GradientReproject(
        VkCommandBuffer cmd, uint32_t frameIndex,
        const std::shared_ptr<const GlobalUniform> &uniform);
    // TAAU upscaler for the new core path. Takes the tonemapped Final at render
    // resolution and writes UpscaledPing at the upscaled resolution.
    void ApplyTAAU(
        VkCommandBuffer cmd, uint32_t frameIndex,
        const std::shared_ptr<const GlobalUniform> &uniform);

    // Q2RTX-style fog volumes: blends the final HDR image toward the fog color.
    void ApplyFog(
        VkCommandBuffer cmd, uint32_t frameIndex,
        const std::shared_ptr<const GlobalUniform> &uniform);

    void OnShaderReload(const ShaderManager *shaderManager) override;

private:
    // Resolves the checkerboard packed Q2Color image into PreFinal, which uses
    // the regular pixel layout. Shared by the denoised and the unfiltered path.
    void InterleaveCheckerboard(
        VkCommandBuffer cmd, uint32_t frameIndex,
        const std::shared_ptr<const GlobalUniform> &uniform);

    void CreatePipelineLayout(VkDescriptorSetLayout *pSetLayouts, uint32_t setLayoutCount);
    void CreatePipelines(const ShaderManager *shaderManager);
    void DestroyPipelines();

private:
    // Number of a-trous iterations of the Q2RTX-style gradient filter
    // (CmQ2GradientAtrous.comp). 7 like Q2RTX: LF in all, HF/SPEC in the
    // first 3, LF normalized in the last one.
    static constexpr uint32_t Q2_GRADIENT_ATROUS_ITERATION_COUNT = 7;

    VkDevice device;

    std::shared_ptr<Framebuffers> framebuffers;

    VkPipelineLayout pipelineLayout;

    // Q2RTX-style gradient pipeline (produces Q2GradLF / Q2GradHFSpec,
    // consumed by the temporal pass): reproject + gradient img + gradient atrous.
    VkPipeline gradientReproject;
    VkPipeline gradientImg;
    VkPipeline gradientAtrous[Q2_GRADIENT_ATROUS_ITERATION_COUNT];

    VkPipeline adapter;
    VkPipeline temporal;
    VkPipeline atrousLF[4];
    VkPipeline atrous[4];
    VkPipeline interleave;
    VkPipeline fog;
    VkPipeline taau;
};

}
