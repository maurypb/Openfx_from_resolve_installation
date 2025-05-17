#include "BlurPlugin.h"
#include "BlurPluginParameters.h"
#include "PluginClips.h"
#include "PluginParameters.h"
#include "GenericPluginFactory.h"
#include "PluginDefines.h"

// Define the blur plugin with one macro
DEFINE_PLUGIN(
    BlurPlugin,                    // Effect class
    BlurPluginParameters,          // Parameters class
    "com.maury.gaussianBlur",      // Identifier
    1,                             // Major version
    5                              // Minor version
)

// That's all that's needed! The macro expands to:
// 
// class BlurPluginFactory : public GenericPluginFactory<BlurPlugin, BlurPluginParameters> {
// public:
//     BlurPluginFactory() : GenericPluginFactory<BlurPlugin, BlurPluginParameters>(
//         "com.Maury.GaussianBlur", 1, 5) {}
// };
// static BlurPluginFactory g_BlurPluginFactory;
// 
// void DefinePlugins(OFX::PluginFactoryArray& p_FactoryArray) {
//     p_FactoryArray.push_back(&g_BlurPluginFactory);
// }
//
// The GenericPluginMain.cpp file contains:
// void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& p_FactoryArray) {
//     DefinePlugins(p_FactoryArray);
// }