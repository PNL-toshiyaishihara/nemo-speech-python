# Injected into the external projects through CMAKE_PROJECT_INCLUDE, so it
# applies to every target without modifying the vendored sources.

if(MSVC)
    # Windows wheels bundle msvcp140.dll unmangled (see the delvewheel repair
    # step). If the process already loaded an older copy, binaries built with
    # VS 2022 17.10+ crash in std::mutex unless the constexpr constructor is
    # disabled; this is Microsoft's documented mitigation.
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# Open JTalk's MeCab (the Japanese TTS tokenizer) uses std::binary_function,
# which C++17 removed. Upstream restores it for MSVC only, and in a way that
# breaks the Visual Studio generator. Fixed up once every target exists.
function(nsp_fix_openjtalk_frontend)
    set(target nemo_speech_openjtalk_frontend)
    if(NOT TARGET ${target})
        return()
    endif()
    # libc++ (macOS) keeps it behind an opt-in macro.
    if(APPLE)
        set_property(TARGET ${target} APPEND PROPERTY
            COMPILE_DEFINITIONS _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
    endif()
    # Upstream force-includes <functional> with
    # $<$<COMPILE_LANGUAGE:CXX>:/FIfunctional>. Visual Studio generators
    # evaluate target-wide options once for all of a target's sources, so the C
    # sources get it too and fail with STL1003. Move it onto the C++ sources.
    if(CMAKE_GENERATOR MATCHES "^Visual Studio")
        get_target_property(options ${target} COMPILE_OPTIONS)
        if(options MATCHES "/FIfunctional")
            list(FILTER options EXCLUDE REGEX "/FIfunctional")
            set_property(TARGET ${target} PROPERTY COMPILE_OPTIONS "${options}")
            get_target_property(sources ${target} SOURCES)
            list(FILTER sources INCLUDE REGEX "\\.cpp$")
            set_property(SOURCE ${sources} TARGET_DIRECTORY ${target}
                APPEND PROPERTY COMPILE_OPTIONS /FIfunctional)
        endif()
    endif()
endfunction()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    cmake_language(DEFER CALL nsp_fix_openjtalk_frontend)
endif()
