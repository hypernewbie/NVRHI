// Metal framebuffers retain their attachments.
#include <nvrhi/metal.h>
#include <cstdio>

int main()
{
    auto device = nvrhi::metal::createDevice({});
    if (!device) return 1;

    nvrhi::TextureDesc desc;
    desc.width = desc.height = 16;
    desc.isRenderTarget = true;
    desc.format = nvrhi::Format::RGBA8_UNORM;
    auto color = device->createTexture(desc);
    desc.format = nvrhi::Format::D32;
    auto depth = device->createTexture(desc);
    if (!color || !depth) return 1;

    const auto colorRefs = color->GetRefCount();
    const auto depthRefs = depth->GetRefCount();
    auto fb = device->createFramebuffer(nvrhi::FramebufferDesc().addColorAttachment(color).setDepthAttachment(depth));
    if (!fb || color->GetRefCount() <= colorRefs || depth->GetRefCount() <= depthRefs)
    {
        std::puts("Framebuffer does not retain its attachments");
        return 1;
    }
    fb = nullptr;
    return color->GetRefCount() == colorRefs && depth->GetRefCount() == depthRefs ? 0 : 1;
}
