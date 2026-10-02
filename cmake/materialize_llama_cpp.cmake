# Materialize llama.cpp with NeMo-Speech.cpp's patches/ series applied, then
# this repository's own patches/llama.cpp series, for
# NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR.
#
# Adapted from NeMo-Speech.cpp cmake/llama_cpp.cmake (Copyright (c) 2026 NVIDIA
# CORPORATION & AFFILIATES, Apache-2.0); modified as described below.
#
# Upstream can do this itself, but it normalizes each patch with file(WRITE),
# which emits CRLF on Windows. With core.autocrlf=input or false the exported
# tree is LF, so every patch fails to apply. Here the tree and the patches both
# come from git objects with EOL conversion disabled, so their line endings
# always agree regardless of the user's git configuration. Without git metadata
# (an sdist) the files on disk are used as they are. This repository's patches
# are always read from disk; .gitattributes keeps them LF.

# Patch file names listed in `patch_dir`/series, in apply order.
function(nsp_patch_series patch_dir out_var)
    file(STRINGS "${patch_dir}/series" lines)
    set(patches "")
    foreach(line IN LISTS lines)
        string(REGEX REPLACE "#.*$" "" line "${line}")
        string(STRIP "${line}" line)
        if(NOT line STREQUAL "")
            list(APPEND patches "${line}")
        endif()
    endforeach()
    set(${out_var} "${patches}" PARENT_SCOPE)
endfunction()

# True when `dir` is itself the top of a working git checkout. An sdist has no
# git metadata, and may be unpacked inside some unrelated repository.
function(nsp_is_git_toplevel dir out_var)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${dir}" rev-parse --show-toplevel
        OUTPUT_VARIABLE toplevel RESULT_VARIABLE rc
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(result OFF)
    if(rc EQUAL 0)
        file(REAL_PATH "${toplevel}" toplevel)
        file(REAL_PATH "${dir}" dir)
        if(toplevel STREQUAL dir)
            set(result ON)
        endif()
    endif()
    set(${out_var} ${result} PARENT_SCOPE)
endfunction()

function(nsp_materialize_llama_cpp nemo_dir extra_patch_dir dest_dir)
    find_package(Git REQUIRED)
    set(src "${nemo_dir}/llama.cpp")
    set(git "${GIT_EXECUTABLE}" -c core.autocrlf=false -c core.eol=lf)
    nsp_patch_series("${nemo_dir}/patches" patches)
    nsp_patch_series("${extra_patch_dir}" extra_patches)

    nsp_is_git_toplevel("${src}" src_is_git)
    nsp_is_git_toplevel("${nemo_dir}" nemo_is_git)
    set(use_git OFF)
    if(src_is_git AND nemo_is_git)
        set(use_git ON)
    endif()

    # Stamp: the llama.cpp commit plus the exact patch bytes.
    set(work "${dest_dir}.work")
    file(REMOVE_RECURSE "${work}")
    file(MAKE_DIRECTORY "${work}/patches/upstream" "${work}/patches/extra" "${work}/tree")
    if(use_git)
        execute_process(COMMAND ${git} -C "${src}" rev-parse HEAD
            OUTPUT_VARIABLE stamp OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    else()
        file(SHA256 "${src}/CMakeLists.txt" stamp)
    endif()
    set(apply "")
    foreach(patch IN LISTS patches)
        set(patch_file "${work}/patches/upstream/${patch}")
        list(APPEND apply "${patch_file}")
        if(use_git)
            # Raw blob bytes: no EOL conversion or filters.
            execute_process(COMMAND ${git} -C "${nemo_dir}" show "HEAD:patches/${patch}"
                OUTPUT_FILE "${patch_file}" COMMAND_ERROR_IS_FATAL ANY)
        else()
            file(COPY_FILE "${nemo_dir}/patches/${patch}" "${patch_file}")
        endif()
        file(SHA256 "${patch_file}" digest)
        string(APPEND stamp "-${digest}")
    endforeach()
    foreach(patch IN LISTS extra_patches)
        set(patch_file "${work}/patches/extra/${patch}")
        list(APPEND apply "${patch_file}")
        file(COPY_FILE "${extra_patch_dir}/${patch}" "${patch_file}")
        file(SHA256 "${patch_file}" digest)
        string(APPEND stamp "-${digest}")
    endforeach()
    string(SHA256 stamp "${stamp}")

    set(stamp_file "${dest_dir}.stamp")
    if(EXISTS "${stamp_file}" AND EXISTS "${dest_dir}/ggml/CMakeLists.txt")
        file(READ "${stamp_file}" previous)
        if(previous STREQUAL stamp)
            file(REMOVE_RECURSE "${work}")
            return()
        endif()
    endif()

    message(STATUS "Applying ${nemo_dir}/patches and ${extra_patch_dir} to llama.cpp in ${dest_dir}")
    set(tree "${work}/tree")
    if(use_git)
        execute_process(
            COMMAND ${git} -C "${src}" archive --format=tar -o "${work}/tree.tar" HEAD
                    -- . ":(exclude)models" ":(exclude)docs" ":(exclude)media"
            COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf "${work}/tree.tar"
            WORKING_DIRECTORY "${tree}" COMMAND_ERROR_IS_FATAL ANY)
    else()
        file(GLOB entries RELATIVE "${src}" "${src}/*")
        foreach(entry IN LISTS entries)
            if(NOT entry MATCHES "^(\\.git|models|docs|media)$")
                file(COPY "${src}/${entry}" DESTINATION "${tree}")
            endif()
        endforeach()
    endif()

    # The build tree usually sits inside another work tree; keep git from
    # discovering it so patch paths resolve relative to the copy.
    foreach(patch_file IN LISTS apply)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${work}"
                    ${git} apply --whitespace=nowarn "${patch_file}"
            WORKING_DIRECTORY "${tree}"
            RESULT_VARIABLE rc ERROR_VARIABLE error)
        if(NOT rc EQUAL 0)
            get_filename_component(name "${patch_file}" NAME)
            message(FATAL_ERROR "${name} does not apply to llama.cpp:\n${error}")
        endif()
    endforeach()

    # Copy only changed files so rebuilds recompile only what a patch touched.
    file(GLOB_RECURSE new_files RELATIVE "${tree}" LIST_DIRECTORIES false "${tree}/*")
    foreach(path IN LISTS new_files)
        get_filename_component(dir "${dest_dir}/${path}" DIRECTORY)
        file(MAKE_DIRECTORY "${dir}")
        file(COPY_FILE "${tree}/${path}" "${dest_dir}/${path}" ONLY_IF_DIFFERENT)
    endforeach()
    file(GLOB_RECURSE old_files RELATIVE "${dest_dir}" LIST_DIRECTORIES false "${dest_dir}/*")
    foreach(path IN LISTS old_files)
        if(NOT EXISTS "${tree}/${path}")
            file(REMOVE "${dest_dir}/${path}")
        endif()
    endforeach()
    file(REMOVE_RECURSE "${work}")
    file(WRITE "${stamp_file}" "${stamp}")
endfunction()
