/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include "metal-backend.h"

namespace nvrhi::metal
{
    MTLPixelFormat convertFormat(Format format)
    {
        switch (format)
        {
        case Format::R8_UINT:       return MTLPixelFormatR8Uint;
        case Format::R8_SINT:       return MTLPixelFormatR8Sint;
        case Format::R8_UNORM:      return MTLPixelFormatR8Unorm;
        case Format::R8_SNORM:      return MTLPixelFormatR8Snorm;
        case Format::RG8_UINT:      return MTLPixelFormatRG8Uint;
        case Format::RG8_SINT:      return MTLPixelFormatRG8Sint;
        case Format::RG8_UNORM:     return MTLPixelFormatRG8Unorm;
        case Format::RG8_SNORM:     return MTLPixelFormatRG8Snorm;
        case Format::R16_UINT:      return MTLPixelFormatR16Uint;
        case Format::R16_SINT:      return MTLPixelFormatR16Sint;
        case Format::R16_UNORM:     return MTLPixelFormatR16Unorm;
        case Format::R16_SNORM:     return MTLPixelFormatR16Snorm;
        case Format::R16_FLOAT:     return MTLPixelFormatR16Float;
        case Format::RG16_UINT:     return MTLPixelFormatRG16Uint;
        case Format::RG16_SINT:     return MTLPixelFormatRG16Sint;
        case Format::RG16_UNORM:    return MTLPixelFormatRG16Unorm;
        case Format::RG16_SNORM:    return MTLPixelFormatRG16Snorm;
        case Format::RG16_FLOAT:    return MTLPixelFormatRG16Float;
        case Format::R32_UINT:      return MTLPixelFormatR32Uint;
        case Format::R32_SINT:      return MTLPixelFormatR32Sint;
        case Format::R32_FLOAT:     return MTLPixelFormatR32Float;
        case Format::RG32_UINT:     return MTLPixelFormatRG32Uint;
        case Format::RG32_SINT:     return MTLPixelFormatRG32Sint;
        case Format::RG32_FLOAT:    return MTLPixelFormatRG32Float;
        case Format::RGBA8_UINT:    return MTLPixelFormatRGBA8Uint;
        case Format::RGBA8_SINT:    return MTLPixelFormatRGBA8Sint;
        case Format::RGBA8_UNORM:   return MTLPixelFormatRGBA8Unorm;
        case Format::RGBA8_SNORM:   return MTLPixelFormatRGBA8Snorm;
        case Format::BGRA8_UNORM:   return MTLPixelFormatBGRA8Unorm;
        case Format::BGRX8_UNORM:   return MTLPixelFormatBGRA8Unorm; // Metal has no X channel
        case Format::SRGBA8_UNORM:  return MTLPixelFormatRGBA8Unorm_sRGB;
        case Format::SBGRA8_UNORM:  return MTLPixelFormatBGRA8Unorm_sRGB;
        case Format::SBGRX8_UNORM:  return MTLPixelFormatBGRA8Unorm_sRGB; // Metal has no X channel
        case Format::R10G10B10A2_UNORM: return MTLPixelFormatRGB10A2Unorm;
        case Format::R11G11B10_FLOAT:   return MTLPixelFormatRG11B10Float;
        case Format::RGBA16_UINT:   return MTLPixelFormatRGBA16Uint;
        case Format::RGBA16_SINT:   return MTLPixelFormatRGBA16Sint;
        case Format::RGBA16_UNORM:  return MTLPixelFormatRGBA16Unorm;
        case Format::RGBA16_SNORM:  return MTLPixelFormatRGBA16Snorm;
        case Format::RGBA16_FLOAT:  return MTLPixelFormatRGBA16Float;
        case Format::RGBA32_UINT:   return MTLPixelFormatRGBA32Uint;
        case Format::RGBA32_SINT:   return MTLPixelFormatRGBA32Sint;
        case Format::RGBA32_FLOAT:  return MTLPixelFormatRGBA32Float;
        case Format::D16:           return MTLPixelFormatDepth16Unorm;
        case Format::D24S8:         return MTLPixelFormatDepth32Float_Stencil8;
        case Format::D32:           return MTLPixelFormatDepth32Float;
        case Format::D32S8:         return MTLPixelFormatDepth32Float_Stencil8;
        case Format::BC1_UNORM:     return MTLPixelFormatBC1_RGBA;
        case Format::BC1_UNORM_SRGB: return MTLPixelFormatBC1_RGBA_sRGB;
        case Format::BC2_UNORM:     return MTLPixelFormatBC2_RGBA;
        case Format::BC2_UNORM_SRGB: return MTLPixelFormatBC2_RGBA_sRGB;
        case Format::BC3_UNORM:     return MTLPixelFormatBC3_RGBA;
        case Format::BC3_UNORM_SRGB: return MTLPixelFormatBC3_RGBA_sRGB;
        case Format::BC4_UNORM:     return MTLPixelFormatBC4_RUnorm;
        case Format::BC4_SNORM:     return MTLPixelFormatBC4_RSnorm;
        case Format::BC5_UNORM:     return MTLPixelFormatBC5_RGUnorm;
        case Format::BC5_SNORM:     return MTLPixelFormatBC5_RGSnorm;
        case Format::BC6H_UFLOAT:   return MTLPixelFormatBC6H_RGBUfloat;
        case Format::BC6H_SFLOAT:   return MTLPixelFormatBC6H_RGBFloat;
        case Format::BC7_UNORM:     return MTLPixelFormatBC7_RGBAUnorm;
        case Format::BC7_UNORM_SRGB: return MTLPixelFormatBC7_RGBAUnorm_sRGB;
        default:                   return MTLPixelFormatInvalid;
        }
    }

    MTLSamplerMinMagFilter convertSamplerFilter(bool linear)
    {
        return linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    }

    MTLSamplerMipFilter convertSamplerMipFilter(bool linear)
    {
        return linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    }

    MTLSamplerAddressMode convertSamplerAddressMode(SamplerAddressMode mode)
    {
        switch (mode)
        {
        case SamplerAddressMode::Wrap: return MTLSamplerAddressModeRepeat;
        case SamplerAddressMode::Mirror: return MTLSamplerAddressModeMirrorRepeat;
        case SamplerAddressMode::MirrorOnce: return MTLSamplerAddressModeMirrorClampToEdge;
        case SamplerAddressMode::Clamp: return MTLSamplerAddressModeClampToEdge;
        case SamplerAddressMode::Border: return MTLSamplerAddressModeClampToBorderColor;
        default: return MTLSamplerAddressModeRepeat;
        }
    }

    MTLSamplerBorderColor convertSamplerBorderColor(const Color& color)
    {
        if (color.r == 0.f && color.g == 0.f && color.b == 0.f)
            return color.a == 0.f ? MTLSamplerBorderColorTransparentBlack : MTLSamplerBorderColorOpaqueBlack;

        if (color.r == 1.f && color.g == 1.f && color.b == 1.f)
            return MTLSamplerBorderColorOpaqueWhite;

        return MTLSamplerBorderColorTransparentBlack;
    }

    MTLCompareFunction convertCompareFunc(ComparisonFunc op)
    {
        switch (op)
        {
        case ComparisonFunc::Never: return MTLCompareFunctionNever;
        case ComparisonFunc::Less: return MTLCompareFunctionLess;
        case ComparisonFunc::Equal: return MTLCompareFunctionEqual;
        case ComparisonFunc::LessOrEqual: return MTLCompareFunctionLessEqual;
        case ComparisonFunc::Greater: return MTLCompareFunctionGreater;
        case ComparisonFunc::NotEqual: return MTLCompareFunctionNotEqual;
        case ComparisonFunc::GreaterOrEqual: return MTLCompareFunctionGreaterEqual;
        case ComparisonFunc::Always: return MTLCompareFunctionAlways;
        default: return MTLCompareFunctionAlways;
        }
    }

    // A comparison sampler compares the sampled value against the reference value.
    // Metal puts the reference value on the left of the comparison, Vulkan and D3D
    // put the sampled value there, so the operator is mirrored here.
    MTLCompareFunction convertSamplerCompareFunc(ComparisonFunc op)
    {
        switch (op)
        {
        case ComparisonFunc::Never: return MTLCompareFunctionNever;
        case ComparisonFunc::Less: return MTLCompareFunctionGreaterEqual;
        case ComparisonFunc::Equal: return MTLCompareFunctionEqual;
        case ComparisonFunc::LessOrEqual: return MTLCompareFunctionGreater;
        case ComparisonFunc::Greater: return MTLCompareFunctionLessEqual;
        case ComparisonFunc::NotEqual: return MTLCompareFunctionNotEqual;
        case ComparisonFunc::GreaterOrEqual: return MTLCompareFunctionLess;
        case ComparisonFunc::Always: return MTLCompareFunctionAlways;
        default: return MTLCompareFunctionAlways;
        }
    }

    MTLVertexFormat convertVertexFormat(Format format)
    {
        switch (format)
        {
        case Format::R8_UINT:      return MTLVertexFormatUInt;
        case Format::R8_SINT:      return MTLVertexFormatChar;
        case Format::R8_UNORM:     return MTLVertexFormatUCharNormalized;
        case Format::R8_SNORM:     return MTLVertexFormatCharNormalized;
        case Format::RG8_UINT:     return MTLVertexFormatUInt2;
        case Format::RG8_SINT:     return MTLVertexFormatChar2;
        case Format::RG8_UNORM:    return MTLVertexFormatUChar2Normalized;
        case Format::RG8_SNORM:    return MTLVertexFormatChar2Normalized;
        case Format::R16_UINT:     return MTLVertexFormatUShort;
        case Format::R16_SINT:     return MTLVertexFormatShort;
        case Format::R16_UNORM:    return MTLVertexFormatUShortNormalized;
        case Format::R16_SNORM:    return MTLVertexFormatShortNormalized;
        case Format::R16_FLOAT:    return MTLVertexFormatHalf;
        case Format::RG16_UINT:    return MTLVertexFormatUShort2;
        case Format::RG16_SINT:    return MTLVertexFormatShort2;
        case Format::RG16_UNORM:   return MTLVertexFormatUShort2Normalized;
        case Format::RG16_SNORM:   return MTLVertexFormatShort2Normalized;
        case Format::RG16_FLOAT:   return MTLVertexFormatHalf2;
        case Format::R32_UINT:     return MTLVertexFormatUInt;
        case Format::R32_SINT:     return MTLVertexFormatInt;
        case Format::R32_FLOAT:    return MTLVertexFormatFloat;
        case Format::RGB32_UINT:   return MTLVertexFormatUInt3;
        case Format::RGB32_SINT:   return MTLVertexFormatInt3;
        case Format::RGB32_FLOAT:  return MTLVertexFormatFloat3;
        case Format::RG32_UINT:    return MTLVertexFormatUInt2;
        case Format::RG32_SINT:    return MTLVertexFormatInt2;
        case Format::RG32_FLOAT:   return MTLVertexFormatFloat2;
        case Format::RGBA8_UINT:   return MTLVertexFormatUInt4;
        case Format::RGBA8_SINT:   return MTLVertexFormatChar4;
        case Format::RGBA8_UNORM:  return MTLVertexFormatUChar4Normalized;
        case Format::RGBA8_SNORM:  return MTLVertexFormatChar4Normalized;
        case Format::RGBA16_UINT:  return MTLVertexFormatUShort4;
        case Format::RGBA16_SINT:  return MTLVertexFormatShort4;
        case Format::RGBA16_UNORM: return MTLVertexFormatUShort4Normalized;
        case Format::RGBA16_SNORM: return MTLVertexFormatShort4Normalized;
        case Format::RGBA16_FLOAT: return MTLVertexFormatHalf4;
        case Format::RGBA32_UINT:  return MTLVertexFormatUInt4;
        case Format::RGBA32_SINT:  return MTLVertexFormatInt4;
        case Format::RGBA32_FLOAT: return MTLVertexFormatFloat4;
        default:                  return MTLVertexFormatInvalid;
        }
    }

    MTLPrimitiveType convertPrimitiveType(PrimitiveType type)
    {
        switch (type)
        {
        case PrimitiveType::TriangleList:  return MTLPrimitiveTypeTriangle;
        case PrimitiveType::TriangleStrip: return MTLPrimitiveTypeTriangleStrip;
        case PrimitiveType::LineList:      return MTLPrimitiveTypeLine;
        case PrimitiveType::LineStrip:     return MTLPrimitiveTypeLineStrip;
        case PrimitiveType::PointList:     return MTLPrimitiveTypePoint;
        default:                          return MTLPrimitiveTypeTriangle;
        }
    }

    MTLCullMode convertCullMode(RasterCullMode mode)
    {
        switch (mode)
        {
        case RasterCullMode::Back:  return MTLCullModeBack;
        case RasterCullMode::Front: return MTLCullModeFront;
        case RasterCullMode::None:  return MTLCullModeNone;
        default:                    return MTLCullModeBack;
        }
    }

    MTLBlendFactor convertBlendFactor(BlendFactor factor)
    {
        switch (factor)
        {
        case BlendFactor::Zero: return MTLBlendFactorZero;
        case BlendFactor::One: return MTLBlendFactorOne;
        case BlendFactor::SrcColor: return MTLBlendFactorSourceColor;
        case BlendFactor::InvSrcColor: return MTLBlendFactorOneMinusSourceColor;
        case BlendFactor::SrcAlpha: return MTLBlendFactorSourceAlpha;
        case BlendFactor::InvSrcAlpha: return MTLBlendFactorOneMinusSourceAlpha;
        case BlendFactor::DstColor: return MTLBlendFactorDestinationColor;
        case BlendFactor::InvDstColor: return MTLBlendFactorOneMinusDestinationColor;
        case BlendFactor::DstAlpha: return MTLBlendFactorDestinationAlpha;
        case BlendFactor::InvDstAlpha: return MTLBlendFactorOneMinusDestinationAlpha;
        case BlendFactor::SrcAlphaSaturate: return MTLBlendFactorSourceAlphaSaturated;
        case BlendFactor::ConstantColor: return MTLBlendFactorBlendColor;
        case BlendFactor::InvConstantColor: return MTLBlendFactorOneMinusBlendColor;
        case BlendFactor::Src1Color: return MTLBlendFactorSource1Color;
        case BlendFactor::InvSrc1Color: return MTLBlendFactorOneMinusSource1Color;
        case BlendFactor::Src1Alpha: return MTLBlendFactorSource1Alpha;
        case BlendFactor::InvSrc1Alpha: return MTLBlendFactorOneMinusSource1Alpha;
        default: return MTLBlendFactorOne;
        }
    }

    MTLBlendOperation convertBlendOp(BlendOp op)
    {
        switch (op)
        {
        case BlendOp::Add: return MTLBlendOperationAdd;
        case BlendOp::Subtract: return MTLBlendOperationSubtract;
        case BlendOp::ReverseSubtract: return MTLBlendOperationReverseSubtract;
        case BlendOp::Min: return MTLBlendOperationMin;
        case BlendOp::Max: return MTLBlendOperationMax;
        default: return MTLBlendOperationAdd;
        }
    }

    MTLStencilOperation convertStencilOp(StencilOp op)
    {
        switch (op)
        {
        case StencilOp::Keep: return MTLStencilOperationKeep;
        case StencilOp::Zero: return MTLStencilOperationZero;
        case StencilOp::Replace: return MTLStencilOperationReplace;
        case StencilOp::IncrementAndClamp: return MTLStencilOperationIncrementClamp;
        case StencilOp::DecrementAndClamp: return MTLStencilOperationDecrementClamp;
        case StencilOp::Invert: return MTLStencilOperationInvert;
        case StencilOp::IncrementAndWrap: return MTLStencilOperationIncrementWrap;
        case StencilOp::DecrementAndWrap: return MTLStencilOperationDecrementWrap;
        default: return MTLStencilOperationKeep;
        }
    }
}
