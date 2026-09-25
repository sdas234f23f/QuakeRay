// Copyright (c) 2022 Sultim Tsyrendashiev
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <filesystem>
#include <fstream>

namespace vkpt::LibraryConfig
{
    struct Config
    {
        bool vulkanValidation = false;
        bool developerMode = false;
        bool dlssValidation = false;
        bool fpsMonitor = false;
        // Draws the frame through the RHI layer instead of the renderer.
        // Only for the bring-up of the RHI frame skeleton.
        bool rhiFrameSkeleton = false;
        // Draws the frame through the RHI debug ray-tracing pass (A3) instead of the rasterized RHI
        // chain. Only for the bring-up of the acceleration structures and the first traced image.
        bool rhiDebugTrace = false;
        // Draws the frame through the real ray-tracing passes of the RHI path (A4) instead of the
        // rasterized chain. As of A4.1 this is the primary-visibility pass: the engine's primary
        // raygen fills the checkerboard G-buffer (ALBEDO included) and the present shows it without
        // lighting; A4.2 added the direct-lighting pass and the present's diagnostic compose. The
        // later A4 cuts add the denoiser and the full composition. Requires 'rhiframe'; when
        // 'rhitrace' is set as well, this mode wins.
        bool rhiRayTracing = false;
        // Draws the fully composed traced frame: the real adapter -> interleave -> exposure
        // histogram/average -> checkerboard -> prepare-final chain writes the display-referred
        // FINAL (the ASVGF denoiser is not ported yet) and the present shows it raw, instead of the
        // A4.2a diagnostic present of ALBEDO + the direct term. Requires 'rhirt'.
        bool rhiCompose = false;
    };

    namespace detail
    {
        inline void ProcessEntry(Config &dst, std::string_view entry)
        {
            if (entry == "vulkanvalidation")
            {
                dst.vulkanValidation = true;
            }
            else if (entry == "developer")
            {
                dst.developerMode = true;
            }
            else if (entry == "dlssvalidation")
            {
                dst.dlssValidation = true;
            }
            else if (entry == "fpsmonitor")
            {
                dst.fpsMonitor = true;
            }
            else if (entry == "rhiframe")
            {
                dst.rhiFrameSkeleton = true;
            }
            else if (entry == "rhitrace")
            {
                dst.rhiDebugTrace = true;
            }
            else if (entry == "rhirt")
            {
                dst.rhiRayTracing = true;
            }
            else if (entry == "rhicompose")
            {
                dst.rhiCompose = true;
            }
        }
    }

    inline Config Read(const char *pPath)
    {
        if (pPath == nullptr || pPath[0] == '\0')
        {
            pPath = "vkpt.txt";
        }

        auto path = std::filesystem::path(pPath);

        if (std::filesystem::exists(path))
        {
            std::ifstream file(path);

            if (file.is_open())
            {
                Config result = {};

                for (std::string line; std::getline(file, line); )
                {
                    std::ranges::transform(line, line.begin(), ::tolower);

                    detail::ProcessEntry(result, line);
                }

                return result;
            }
        }

        return {};
    }
}
