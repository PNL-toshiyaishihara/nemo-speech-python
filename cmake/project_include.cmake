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

# Every buildsystem target of `dir` and its subdirectories.
function(nsp_collect_targets dir out_var)
    get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(subdir IN LISTS subdirs)
        nsp_collect_targets("${subdir}" sub_targets)
        list(APPEND targets ${sub_targets})
    endforeach()
    set(${out_var} "${targets}" PARENT_SCOPE)
endfunction()

# A path for comparing source files: absolute, symlinks resolved, and
# case-folded where the file system ignores case.
function(nsp_source_key path base_dir out_var)
    file(REAL_PATH "${path}" key BASE_DIRECTORY "${base_dir}")
    if(WIN32 OR APPLE)
        string(TOLOWER "${key}" key)
    endif()
    set(${out_var} "${key}" PARENT_SCOPE)
endfunction()

# NSP_SOURCE_OVERRIDES ("upstream;patched;...", from the superbuild; see
# cmake/patch_nemo_speech.cmake): compile each patched copy in place of the
# upstream file, in every target. A file no target compiles is an error, so a
# patch can never be silently left out.
function(nsp_apply_source_overrides)
    if(NOT NSP_SOURCE_OVERRIDES)
        return()
    endif()
    list(LENGTH NSP_SOURCE_OVERRIDES count)
    math(EXPR odd "${count} % 2")
    if(odd)
        message(FATAL_ERROR "NSP_SOURCE_OVERRIDES must hold upstream/patched pairs: ${NSP_SOURCE_OVERRIDES}")
    endif()
    nsp_collect_targets("${CMAKE_SOURCE_DIR}" targets)
    math(EXPR last "${count} - 1")
    foreach(i RANGE 0 ${last} 2)
        math(EXPR j "${i} + 1")
        list(GET NSP_SOURCE_OVERRIDES ${i} upstream)
        list(GET NSP_SOURCE_OVERRIDES ${j} patched)
        nsp_source_key("${upstream}" "${CMAKE_SOURCE_DIR}" upstream_key)
        set(replaced 0)
        foreach(target IN LISTS targets)
            get_target_property(sources ${target} SOURCES)
            if(NOT sources)
                continue()
            endif()
            get_target_property(target_dir ${target} SOURCE_DIR)
            set(new_sources "")
            set(changed OFF)
            foreach(source IN LISTS sources)
                if(NOT source MATCHES "^\\$<")
                    nsp_source_key("${source}" "${target_dir}" source_key)
                    if(source_key STREQUAL upstream_key)
                        set(source "${patched}")
                        set(changed ON)
                        math(EXPR replaced "${replaced} + 1")
                    endif()
                endif()
                list(APPEND new_sources "${source}")
            endforeach()
            if(changed)
                set_property(TARGET ${target} PROPERTY SOURCES "${new_sources}")
                message(STATUS "nemo-speech-python: ${target} compiles ${patched}")
            endif()
        endforeach()
        if(replaced EQUAL 0)
            message(FATAL_ERROR "no target compiles ${upstream}, which patches/NeMo-Speech.cpp patches")
        endif()
    endforeach()
endfunction()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    cmake_language(DEFER CALL nsp_fix_openjtalk_frontend)
    cmake_language(DEFER CALL nsp_apply_source_overrides)
endif()
