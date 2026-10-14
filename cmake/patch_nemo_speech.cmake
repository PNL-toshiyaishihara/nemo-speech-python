# Patched copies of NeMo-Speech.cpp sources for this repository's
# patches/NeMo-Speech.cpp series.
#
# vendor/NeMo-Speech.cpp itself is never modified. Each patch may only modify
# existing files; this writes the patched files under `dest_dir` and returns
# pairs of (upstream file, patched copy). The superbuild hands the pairs to
# NeMo-Speech.cpp's configuration, where cmake/project_include.cmake puts the
# patched copy in place of the upstream file in every target that compiles it.
#
# As for llama.cpp (cmake/materialize_llama_cpp.cmake), the upstream files come
# from git objects with EOL conversion disabled when vendor/NeMo-Speech.cpp is a
# checkout, so they match the LF patches whatever core.autocrlf is set to; in an
# sdist the files on disk are used.

include("${CMAKE_CURRENT_LIST_DIR}/materialize_llama_cpp.cmake")  # nsp_patch_series, nsp_is_git_toplevel

# Sets `out_overrides` to "upstream;patched;upstream;patched;..." (absolute
# paths) and `out_patches` to the applied patch names.
function(nsp_patch_nemo_speech_sources nemo_dir patch_dir dest_dir out_overrides out_patches)
    set(${out_overrides} "" PARENT_SCOPE)
    set(${out_patches} "" PARENT_SCOPE)
    if(NOT EXISTS "${patch_dir}/series")
        return()
    endif()
    nsp_patch_series("${patch_dir}" patches)
    if(NOT patches)
        file(REMOVE_RECURSE "${dest_dir}")
        return()
    endif()
    find_package(Git REQUIRED)
    set(git "${GIT_EXECUTABLE}" -c core.autocrlf=false -c core.eol=lf)
    nsp_is_git_toplevel("${nemo_dir}" use_git)

    # Files the series touches, from the "+++ b/<path>" lines.
    set(files "")
    foreach(patch IN LISTS patches)
        file(STRINGS "${patch_dir}/${patch}" lines REGEX "^(\\+\\+\\+|---) ")
        foreach(line IN LISTS lines)
            if(line MATCHES "^--- /dev/null")
                message(FATAL_ERROR "${patch}: patches/NeMo-Speech.cpp may only modify existing files")
            elseif(line MATCHES "^\\+\\+\\+ b/(.+)$")
                list(APPEND files "${CMAKE_MATCH_1}")
            endif()
        endforeach()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${patch_dir}/${patch}")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${patch_dir}/series")
    list(REMOVE_DUPLICATES files)

    # Upstream files into a scratch tree, then the series on top of them.
    set(work "${dest_dir}.work")
    file(REMOVE_RECURSE "${work}")
    foreach(path IN LISTS files)
        get_filename_component(dir "${work}/${path}" DIRECTORY)
        file(MAKE_DIRECTORY "${dir}")
        if(use_git)
            execute_process(COMMAND ${git} -C "${nemo_dir}" show "HEAD:${path}"
                OUTPUT_FILE "${work}/${path}" RESULT_VARIABLE rc ERROR_VARIABLE error)
            if(NOT rc EQUAL 0)
                message(FATAL_ERROR "cannot read ${path} from ${nemo_dir}:\n${error}")
            endif()
        else()
            file(COPY_FILE "${nemo_dir}/${path}" "${work}/${path}")
        endif()
    endforeach()
    foreach(patch IN LISTS patches)
        # The build tree may sit inside another work tree; keep git from
        # discovering it so patch paths resolve relative to the scratch tree.
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${work}"
                    ${git} apply --whitespace=nowarn "${patch_dir}/${patch}"
            WORKING_DIRECTORY "${work}"
            RESULT_VARIABLE rc ERROR_VARIABLE error)
        if(NOT rc EQUAL 0)
            message(FATAL_ERROR "${patch} does not apply to NeMo-Speech.cpp:\n${error}")
        endif()
    endforeach()

    # Copy only changed files, so rebuilds recompile only what changed, and
    # drop copies a previous series left behind.
    set(overrides "")
    foreach(path IN LISTS files)
        get_filename_component(dir "${dest_dir}/${path}" DIRECTORY)
        file(MAKE_DIRECTORY "${dir}")
        file(COPY_FILE "${work}/${path}" "${dest_dir}/${path}" ONLY_IF_DIFFERENT)
        list(APPEND overrides "${nemo_dir}/${path}" "${dest_dir}/${path}")
    endforeach()
    file(GLOB_RECURSE old_files RELATIVE "${dest_dir}" LIST_DIRECTORIES false "${dest_dir}/*")
    foreach(path IN LISTS old_files)
        if(NOT path IN_LIST files)
            file(REMOVE "${dest_dir}/${path}")
        endif()
    endforeach()
    file(REMOVE_RECURSE "${work}")
    message(STATUS "Patched NeMo-Speech.cpp sources (${patches}): ${files}")

    set(${out_overrides} "${overrides}" PARENT_SCOPE)
    set(${out_patches} "${patches}" PARENT_SCOPE)
endfunction()
