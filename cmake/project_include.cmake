# Injected into the external projects through CMAKE_PROJECT_INCLUDE, so it
# applies to every target without modifying the vendored sources.

if(MSVC)
    # Windows wheels bundle msvcp140.dll unmangled (see the delvewheel repair
    # step). If the process already loaded an older copy, binaries built with
    # VS 2022 17.10+ crash in std::mutex unless the constexpr constructor is
    # disabled; this is Microsoft's documented mitigation.
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()
