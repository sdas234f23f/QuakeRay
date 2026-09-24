// Copyright (c) 2020-2021 Sultim Tsyrendashiev
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

#include <list>
#include <string>

#include "Common.h"
#include "CommandBufferManager.h"
#include "Material.h"
#include "AutoBuffer.h"
#include "ImageLoader.h"
#include "ImageLoaderDev.h"
#include "IMaterialDependency.h"
#include "MemoryAllocator.h"
#include "SamplerManager.h"
#include "TextureDescriptors.h"
#include "TextureUploader.h"
#include "LibraryConfig.h"
#include "TextureObserver.h"

namespace vkpt
{

namespace rhi
{
class RhiTextureTable;
}

class TextureManager
{
public:
    explicit TextureManager(
        VkDevice device,
        std::shared_ptr<MemoryAllocator> memAllocator,
        std::shared_ptr<SamplerManager> samplerManager,
        const std::shared_ptr<CommandBufferManager> &cmdManager,
        std::shared_ptr<UserFileLoad> userFileLoad,
        const RgInstanceCreateInfo &info,
        const LibraryConfig::Config &config);
    ~TextureManager();

    TextureManager(const TextureManager &other) = delete;
    TextureManager(TextureManager &&other) noexcept = delete;
    TextureManager &operator=(const TextureManager &other) = delete;
    TextureManager &operator=(TextureManager &&other) noexcept = delete;

    void PrepareForFrame(uint32_t frameIndex);
    void SubmitDescriptors(uint32_t frameIndex,
                           const RgDrawFrameTexturesParams *pTexturesParams,
                           bool forceUpdateAllDescriptors = false); // true, if mip lod bias was changed, for example

    uint32_t CreateMaterial(VkCommandBuffer cmd, uint32_t frameIndex, const RgMaterialCreateInfo &createInfo);
    uint32_t CreateAnimatedMaterial(VkCommandBuffer cmd, uint32_t frameIndex, const RgAnimatedMaterialCreateInfo &createInfo);
    bool ChangeAnimatedMaterialFrame(uint32_t animMaterial, uint32_t materialFrame);
    bool UpdateMaterial(VkCommandBuffer cmd, uint32_t frameIndex, const RgMaterialUpdateInfo &updateInfo);
    void DestroyMaterial(uint32_t currentFrameIndex, uint32_t materialIndex);

    void CheckForHotReload(VkCommandBuffer cmd, uint32_t frameIndex);

    MaterialTextures GetMaterialTextures(uint32_t materialIndex) const;

    VkBuffer GetTalCdfBuffer() const;

    static constexpr uint32_t GetEmptyTextureIndex();
    uint32_t GetWaterNormalTextureIndex() const;

    // The RHI texture table (vkpt::rhi::RhiTextureTable) mirrors this manager's texture and sampler
    // slots. The host sets it after creating the table; while it is null every RHI call is skipped,
    // so the legacy behaviour is unchanged. The manager does not own the table.
    void SetRhiTextureTable(rhi::RhiTextureTable *pTable);

    VkDescriptorSet GetDescSet(uint32_t frameIndex) const;
    VkDescriptorSetLayout GetDescSetLayout() const;

    // Subscribe to material change event.
    // shared_ptr will be transformed to weak_ptr
    void Subscribe(std::shared_ptr<IMaterialDependency> subscriber);
    void Unsubscribe(const IMaterialDependency *subscriber);

private:
    void CreateEmptyTexture(VkCommandBuffer cmd, uint32_t frameIndex);
    void CreateWaterNormalTexture(VkCommandBuffer cmd, uint32_t frameIndex, const char *pFilePath);

    uint32_t PrepareTexture( VkCommandBuffer                                 cmd,
                             uint32_t                                        frameIndex,
                             const std::optional< ImageLoader::ResultInfo >& info,
                             SamplerManager::Handle                          samplerHandle,
                             bool                                            useMipmaps,
                             const char*                                     debugName,
                             bool                                            isUpdateable,
                             std::optional< RgTextureSwizzling >             swizzling );

    uint32_t InsertTexture(uint32_t frameIndex, VkImage image, VkImageView view,
                           SamplerManager::Handle samplerHandle, VkFormat format,
                           VkExtent2D baseSize, uint32_t mipLevels);
    void DestroyTexture(const Texture &texture);
    void AddToBeDestroyed(uint32_t frameIndex, const Texture &texture);

    // Descriptor writes are tracked per slot instead of rescanning every slot each frame
    void MarkDescDirty(uint32_t textureIndex);
    void MarkAllDescDirty();

    void RebuildTalCdf(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t textureIndex, const uint8_t *pData);

    uint32_t GenerateMaterialIndex(const MaterialTextures &materialTextures);
    uint32_t GenerateMaterialIndex(const std::vector<uint32_t> &materialIndices);

    uint32_t InsertMaterial(const MaterialTextures &materialTextures, bool isUpdateable);
    uint32_t InsertAnimatedMaterial(std::vector<uint32_t> &materialIndices);

    void DestroyMaterialTextures(uint32_t frameIndex, uint32_t materialIndex);
    void DestroyMaterialTextures(uint32_t frameIndex, const Material &material);

private:
    struct TalCdfSource
    {
        RgExtent2D  baseSize = {};
        VkFormat    format = VK_FORMAT_UNDEFINED;
        uint32_t    level0Size = 0;
    };

    VkDevice device;
    RgTextureSwizzling pbrSwizzling;

    std::shared_ptr<ImageLoader> imageLoader;

    std::shared_ptr<ImageLoaderDev> imageLoaderDev;
    std::shared_ptr<TextureObserver> observer;

    std::shared_ptr<SamplerManager> samplerMgr;
    std::shared_ptr<TextureDescriptors> textureDesc;
    std::shared_ptr<TextureUploader> textureUploader;

    std::shared_ptr<AutoBuffer> talCdfBuffer;
    std::vector<TalCdfSource> talCdfSources;

    std::vector<Texture> textures;
    // Optional mirror of 'textures' in the RHI bindless table (RHI/RhiTextureTable.h). Not owned;
    // the host sets it with SetRhiTextureTable. All RHI writes are guarded by it being non-null.
    rhi::RhiTextureTable *rhiTextureTable = nullptr;
    // Textures are not destroyed immediately, but when
    // they won't be in use
    std::vector<Texture> texturesToDestroy[MAX_FRAMES_IN_FLIGHT];

    // Texture indices whose descriptor is not written yet. Per descriptor set, because
    // each frame in flight has its own: a changed slot must reach all of them.
    std::vector<uint32_t> texturesToUpdateDesc[MAX_FRAMES_IN_FLIGHT];

    // Marks the slots that are already in the matching texturesToUpdateDesc list: a slot can be
    // marked dirty many times between two submits, but must be written only once per desc set.
    std::vector<uint8_t> texturesToUpdateDescMarked[MAX_FRAMES_IN_FLIGHT];

    rgl::unordered_map<uint32_t, AnimatedMaterial> animatedMaterials;
    rgl::unordered_map<uint32_t, Material> materials;

    uint32_t waterNormalTextureIndex;

    RgSamplerFilter currentDynamicSamplerFilter;

    std::string defaultTexturesPath;
    std::string postfixes[TEXTURES_PER_MATERIAL_COUNT];
    bool overridenIsSRGB[TEXTURES_PER_MATERIAL_COUNT];
    bool originalIsSRGB[TEXTURES_PER_MATERIAL_COUNT];

    bool forceNormalMapFilterLinear;

    std::list<std::weak_ptr<IMaterialDependency>> subscribers;
};

inline constexpr uint32_t TextureManager::GetEmptyTextureIndex()
{
    return EMPTY_TEXTURE_INDEX;
}

}