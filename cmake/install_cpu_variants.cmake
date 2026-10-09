# cmake -P helper of the NeMo-Speech.cpp install step (GGML_CPU_VARIANTS_PROXY).
#
# Copies ggml's CPU backend variants (ggml-cpu-haswell.dll,
# libggml-cpu-alderlake.so, ...) from the build tree into the install prefix,
# where the ggml-cpu proxy finds them beside itself.
#
# Inputs: NSP_SOURCE_DIR (the build tree's output directory), NSP_CONFIG,
# NSP_PATTERN (file name glob), NSP_DESTINATION.

# Multi-config generators (Visual Studio) add a directory per configuration.
set(_dir "${NSP_SOURCE_DIR}/${NSP_CONFIG}")
if(NOT IS_DIRECTORY "${_dir}")
    set(_dir "${NSP_SOURCE_DIR}")
endif()
file(GLOB _variants "${_dir}/${NSP_PATTERN}")
if(NOT _variants)
    message(FATAL_ERROR "no CPU backend variants (${NSP_PATTERN}) in ${_dir}")
endif()
file(MAKE_DIRECTORY "${NSP_DESTINATION}")
foreach(_variant IN LISTS _variants)
    get_filename_component(_name "${_variant}" NAME)
    message(STATUS "Installing: ${NSP_DESTINATION}/${_name}")
    file(COPY_FILE "${_variant}" "${NSP_DESTINATION}/${_name}")
endforeach()
