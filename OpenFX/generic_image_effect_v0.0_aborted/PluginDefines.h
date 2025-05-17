#pragma once

// Macros for plugin factory definition and registration

// Define a plugin factory class and create an instance of it
#define DEFINE_PLUGIN(ClassName, ParamClassName, Identifier, Major, Minor) \
class ClassName##Factory : public GenericPluginFactory<ClassName, ParamClassName> { \
public: \
    ClassName##Factory() : GenericPluginFactory<ClassName, ParamClassName>( \
        Identifier, Major, Minor) {} \
}; \
static ClassName##Factory g_##ClassName##Factory; \
\
void DefinePlugins(OFX::PluginFactoryArray& p_FactoryArray) { \
    p_FactoryArray.push_back(&g_##ClassName##Factory); \
}