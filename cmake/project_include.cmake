# Injected into the external projects through CMAKE_PROJECT_INCLUDE, so it
# applies to every target without modifying the vendored sources.

if(MSVC)
    # Windows wheels bundle msvcp140.dll unmangled (see the delvewheel repair
    # step). If the process already loaded an older copy, binaries built with
    # VS 2022 17.10+ crash in std::mutex unless the constexpr constructor is
    # disabled; this is Microsoft's documented mitigation.
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# Upstream force-includes <functional> into OpenJTalk's MeCab sources on MSVC
# with $<$<COMPILE_LANGUAGE:CXX>:/FIfunctional>. Visual Studio generators
# evaluate target-wide options once for all of a target's sources, so its C
# sources get the option too and fail with STL1003. Move it onto the C++
# sources, once every target of the project exists.
function(nsp_openjtalk_cxx_only_force_include)
    set(target nemo_speech_openjtalk_frontend)
    if(NOT TARGET ${target})
        return()
    endif()
    get_target_property(options ${target} COMPILE_OPTIONS)
    if(NOT options MATCHES "/FIfunctional")
        return()
    endif()
    list(FILTER options EXCLUDE REGEX "/FIfunctional")
    set_property(TARGET ${target} PROPERTY COMPILE_OPTIONS "${options}")
    get_target_property(sources ${target} SOURCES)
    list(FILTER sources INCLUDE REGEX "\\.cpp$")
    set_property(SOURCE ${sources} TARGET_DIRECTORY ${target}
        APPEND PROPERTY COMPILE_OPTIONS /FIfunctional)
endfunction()

if(MSVC AND CMAKE_GENERATOR MATCHES "^Visual Studio"
   AND CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    cmake_language(DEFER CALL nsp_openjtalk_cxx_only_force_include)
endif()
