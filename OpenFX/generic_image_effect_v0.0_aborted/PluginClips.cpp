#include "PluginClips.h"

// Define constants
const char* const PluginClips::CLIP_SOURCE = kOfxImageEffectSimpleSourceClipName;
const char* const PluginClips::CLIP_OUTPUT = kOfxImageEffectOutputClipName;
const char* const PluginClips::CLIP_MASK = "Mask";

void PluginClips::defineBaseClips(OFX::ImageEffectDescriptor& p_Desc, bool p_SupportsTiles) {
    // Source clip
    OFX::ClipDescriptor* srcClip = p_Desc.defineClip(CLIP_SOURCE);
    srcClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(p_SupportsTiles);
    srcClip->setIsMask(false);

    // Output clip
    OFX::ClipDescriptor* dstClip = p_Desc.defineClip(CLIP_OUTPUT);
    dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    dstClip->setSupportsTiles(p_SupportsTiles);
}

void PluginClips::defineMaskClip(OFX::ImageEffectDescriptor& p_Desc, bool p_SupportsTiles) {
    // Create the optional mask clip
    OFX::ClipDescriptor* maskClip = p_Desc.defineClip(CLIP_MASK);
    maskClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    maskClip->addSupportedComponent(OFX::ePixelComponentAlpha);
    maskClip->setTemporalClipAccess(false);
    maskClip->setSupportsTiles(p_SupportsTiles);
    maskClip->setOptional(true);
    maskClip->setIsMask(true);
}

OFX::ClipDescriptor* PluginClips::defineCustomClip(
    OFX::ImageEffectDescriptor& p_Desc, 
    const char* clipName, 
    bool p_SupportsTiles, 
    bool isOptional,
    bool isMask
) {
    OFX::ClipDescriptor* clip = p_Desc.defineClip(clipName);
    clip->addSupportedComponent(OFX::ePixelComponentRGBA);
    clip->setTemporalClipAccess(false);
    clip->setSupportsTiles(p_SupportsTiles);
    clip->setOptional(isOptional);
    clip->setIsMask(isMask);
    return clip;
}