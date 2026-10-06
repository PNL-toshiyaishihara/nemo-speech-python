# Superbuild for the nemo_speech Python package.
#
# NeMo-Speech.cpp is built as its own top-level CMake project (its CMake
# assumes CMAKE_SOURCE_DIR is its own root, so it cannot be add_subdirectory'd).
# A private static SentencePiece is built first so a wheel build needs neither
# vcpkg nor a system package. The runtime libraries are then installed into
# nemo_speech/lib, where the ctypes bindings load them.
#
# Upstream options pass through: any -DNEMO_SPEECH_*, -DGGML_*, -DCMAKE_CUDA_*
# or -DCMAKE_OSX_* given to this project (e.g. via the CMAKE_ARGS environment
# variable of `pip install`) is forwarded to NeMo-Speech.cpp unchanged.
cmake_minimum_required(VERSION 3.26)
project(nemo_speech_python LANGUAGES C CXX)

include(ExternalProject)

set(NSP_UPSTREAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/NeMo-Speech.cpp")
set(NSP_SENTENCEPIECE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/sentencepiece")
# Short paths: the patched llama.cpp copy and MSBuild intermediates nest deeply.
set(NSP_STAGE_DIR "${CMAKE_BINARY_DIR}/stage")
set(NSP_PACKAGE_DIR "nemo_speech")

foreach(_nsp_required IN ITEMS
        "${NSP_UPSTREAM_DIR}/CMakeLists.txt"
        "${NSP_UPSTREAM_DIR}/llama.cpp/CMakeLists.txt"
        "${NSP_SENTENCEPIECE_DIR}/CMakeLists.txt")
    if(NOT EXISTS "${_nsp_required}")
        message(FATAL_ERROR
            "missing ${_nsp_required}; initialize the submodules:\n"
            "  git submodule update --init vendor/NeMo-Speech.cpp vendor/sentencepiece\n"
            "  git -C vendor/NeMo-Speech.cpp submodule update --init llama.cpp")
    endif()
endforeach()

# ---- Wheel defaults (override with -D / CMAKE_ARGS) ----
# Only components with Python bindings are built. GGML_NATIVE=OFF keeps the CPU
# backend portable (AVX2 baseline on x86-64) instead of tuning for the build host.
# The Japanese/Mandarin TTS tokenizers stay off: upstream compiles absolute
# data paths into the library, which do not survive relocation into a wheel.
set(NEMO_SPEECH_BUILD_ASR         ON  CACHE BOOL "Build automatic speech recognition")
set(NEMO_SPEECH_BUILD_DIAR        ON  CACHE BOOL "Build speaker diarization")
set(NEMO_SPEECH_BUILD_TTS         ON  CACHE BOOL "Build text-to-speech")
set(NEMO_SPEECH_BUILD_NMT         ON  CACHE BOOL "Build text translation")
set(NEMO_SPEECH_BUILD_CLI         OFF CACHE BOOL "Build the nemo-speech CLI")
set(NEMO_SPEECH_BUILD_MIC_CAPTURE OFF CACHE BOOL "Build microphone capture")
set(NEMO_SPEECH_GGML_PATCHED      ON  CACHE BOOL "Apply upstream patches/ to ggml and llama.cpp")
set(NEMO_SPEECH_CUBLAS_SHIM       OFF CACHE BOOL "Use upstream's drop-in cuBLAS shim (CUDA)")
set(GGML_NATIVE                   OFF CACHE BOOL "ggml: tune for the build host CPU")
if(APPLE)
    # Metal is available on every supported Mac. ggml's own thread pool
    # replaces OpenMP, which Apple clang lacks (a Homebrew libomp would pin
    # the wheel to the build machine's macOS version).
    set(GGML_METAL                ON  CACHE BOOL "ggml: use Metal")
    set(GGML_OPENMP               OFF CACHE BOOL "ggml: use OpenMP")
endif()

# ---- Build variant ----
# NEMO_SPEECH_VARIANT (environment; see pyproject.toml) names a published
# variant and gives it a local version label. pyproject.toml has already turned
# on its backend; here unknown labels are refused so no wheel is mislabelled.
set(NSP_VARIANT "$ENV{NEMO_SPEECH_VARIANT}")
if(NOT NSP_VARIANT MATCHES "^(|vulkan|cu[0-9]+)$")
    message(FATAL_ERROR
        "unknown NEMO_SPEECH_VARIANT '${NSP_VARIANT}' (expected vulkan, cu<major><minor> or unset)")
endif()
if(NSP_VARIANT)
    set(NSP_VARIANT_NAME "${NSP_VARIANT}")
else()
    set(NSP_VARIANT_NAME default)
endif()

# ---- CUDA ----
# CUDA wheels bundle the real cuBLAS/cuBLASLt and the CUDA runtime (the repair
# step copies them in), so a wheel needs nothing beyond the NVIDIA driver.
if(GGML_CUDA)
    # Upstream's drop-in cuBLAS shim must export everything the pinned ggml-cuda
    # calls, or the resulting libraries fail to load. Before v0.2.0 it lacked
    # cublasSetWorkspace_v2 and cublasSgemmBatched.
    if(NEMO_SPEECH_CUBLAS_SHIM)
        file(READ "${NSP_UPSTREAM_DIR}/kernels/ver_cublas.map" _nsp_shim_map)
        foreach(_nsp_symbol IN ITEMS cublasSetWorkspace_v2 cublasSgemmBatched)
            if(NOT _nsp_shim_map MATCHES "[ \t\n]${_nsp_symbol};")
                message(FATAL_ERROR
                    "NEMO_SPEECH_CUBLAS_SHIM=ON: upstream's cuBLAS shim does not export "
                    "${_nsp_symbol}, which the pinned ggml-cuda calls. Build without the "
                    "shim; the wheel then bundles cuBLAS.")
            endif()
        endforeach()
    endif()

    # The bundled CUDA libraries are NVIDIA redistributables under the CUDA
    # Toolkit EULA (not Apache-2.0); ship the EULA with them.
    find_package(CUDAToolkit REQUIRED)
    if(NSP_VARIANT MATCHES "^cu([0-9]+)([0-9])$")
        set(_nsp_want "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
        if(NOT "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}" STREQUAL _nsp_want)
            message(FATAL_ERROR
                "NEMO_SPEECH_VARIANT=${NSP_VARIANT} needs CUDA ${_nsp_want}, "
                "found CUDA ${CUDAToolkit_VERSION}")
        endif()
    elseif(NSP_VARIANT)
        message(FATAL_ERROR "NEMO_SPEECH_VARIANT=${NSP_VARIANT} cannot be built with GGML_CUDA")
    endif()
    if(CUDAToolkit_LIBRARY_ROOT)
        set(_nsp_cuda_root "${CUDAToolkit_LIBRARY_ROOT}")
    else()
        get_filename_component(_nsp_cuda_root "${CUDAToolkit_BIN_DIR}" DIRECTORY)
    endif()
    set(NSP_CUDA_EULA "")
    # Windows installs and the cuda-documentation RPM put it in the toolkit
    # root; the per-library RPMs ship it under /usr/share/licenses.
    set(_nsp_cuda_rpm "${CUDAToolkit_VERSION_MAJOR}-${CUDAToolkit_VERSION_MINOR}")
    foreach(_nsp_candidate IN ITEMS
            "${_nsp_cuda_root}/LICENSE"
            "${_nsp_cuda_root}/EULA.txt"
            "/usr/share/licenses/libcublas-${_nsp_cuda_rpm}/LICENSE")
        if(EXISTS "${_nsp_candidate}")
            set(NSP_CUDA_EULA "${_nsp_candidate}")
            break()
        endif()
    endforeach()
    if(NOT NSP_CUDA_EULA)
        message(FATAL_ERROR
            "CUDA Toolkit EULA not found under ${_nsp_cuda_root}; it must ship with "
            "the bundled CUDA libraries")
    endif()
endif()

# ---- Patched llama.cpp ----
# Applied here rather than by upstream; see cmake/materialize_llama_cpp.cmake.
set(NSP_LLAMA_CPP_ARGS "")
if(NEMO_SPEECH_GGML_PATCHED AND NOT NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR)
    include(cmake/materialize_llama_cpp.cmake)
    set(NSP_PATCHED_LLAMA_CPP_DIR "${CMAKE_BINARY_DIR}/llama.cpp")
    # patches/llama.cpp holds this repository's own fixes on top of upstream's.
    set(NSP_EXTRA_PATCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama.cpp")
    nsp_materialize_llama_cpp(
        "${NSP_UPSTREAM_DIR}" "${NSP_EXTRA_PATCH_DIR}" "${NSP_PATCHED_LLAMA_CPP_DIR}")
    list(APPEND NSP_LLAMA_CPP_ARGS "-DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR=${NSP_PATCHED_LLAMA_CPP_DIR}")
    # Re-run configuration when either series changes.
    foreach(_nsp_patch_dir IN ITEMS "${NSP_UPSTREAM_DIR}/patches" "${NSP_EXTRA_PATCH_DIR}")
        nsp_patch_series("${_nsp_patch_dir}" _nsp_patches)
        list(TRANSFORM _nsp_patches PREPEND "${_nsp_patch_dir}/")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${_nsp_patch_dir}/series" ${_nsp_patches})
    endforeach()
endif()

# ---- Arguments shared by both external projects ----
if(CMAKE_BUILD_TYPE)
    set(NSP_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
else()
    set(NSP_BUILD_TYPE Release)
endif()
set(NSP_COMMON_ARGS
    "-DCMAKE_BUILD_TYPE=${NSP_BUILD_TYPE}"
    "-DCMAKE_PROJECT_INCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/project_include.cmake")
foreach(_nsp_var IN ITEMS CMAKE_TOOLCHAIN_FILE CMAKE_C_COMPILER CMAKE_CXX_COMPILER
                          CMAKE_MAKE_PROGRAM)
    if(DEFINED ${_nsp_var} AND NOT CMAKE_GENERATOR MATCHES "^Visual Studio")
        list(APPEND NSP_COMMON_ARGS "-D${_nsp_var}=${${_nsp_var}}")
    endif()
endforeach()

# ---- Option forwarding ----
# CMAKE_OSX_* (architectures, deployment target) reach both projects; upstream
# options only NeMo-Speech.cpp.
set(NSP_FORWARD_ARGS "")
get_cmake_property(_nsp_cache_vars CACHE_VARIABLES)
foreach(_nsp_var IN LISTS _nsp_cache_vars)
    if(NOT _nsp_var MATCHES "^(NEMO_SPEECH_|GGML_|CMAKE_CUDA_|CMAKE_OSX_|CUDAToolkit_)")
        continue()
    endif()
    get_property(_nsp_type CACHE "${_nsp_var}" PROPERTY TYPE)
    if(_nsp_type STREQUAL "INTERNAL" OR _nsp_type STREQUAL "STATIC")
        continue()
    endif()
    # Lists (e.g. CMAKE_CUDA_ARCHITECTURES=86;89) survive via LIST_SEPARATOR.
    string(REPLACE ";" "|" _nsp_value "$CACHE{${_nsp_var}}")
    if(_nsp_type STREQUAL "UNINITIALIZED")
        set(_nsp_arg "-D${_nsp_var}=${_nsp_value}")
    else()
        set(_nsp_arg "-D${_nsp_var}:${_nsp_type}=${_nsp_value}")
    endif()
    if(_nsp_var MATCHES "^CMAKE_OSX_")
        list(APPEND NSP_COMMON_ARGS "${_nsp_arg}")
    else()
        list(APPEND NSP_FORWARD_ARGS "${_nsp_arg}")
    endif()
endforeach()

# ---- SentencePiece (static, private to the ASR library) ----
ExternalProject_Add(nsp_sentencepiece
    SOURCE_DIR "${NSP_SENTENCEPIECE_DIR}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/spm"
    INSTALL_DIR "${NSP_STAGE_DIR}/sentencepiece"
    LIST_SEPARATOR |
    CMAKE_ARGS
        ${NSP_COMMON_ARGS}
        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DSPM_ENABLE_SHARED=OFF
        -DSPM_ENABLE_TCMALLOC=OFF
        -DSPM_BUILD_TEST=OFF
        -DSPM_ABSL_PROVIDER=internal
        -DSPM_PROTOBUF_PROVIDER=internal
        # v0.2.0 declares cmake_minimum_required(3.1), which CMake 4 rejects.
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG> --parallel
    INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR> --config $<CONFIG>)

# ---- NeMo-Speech.cpp ----
ExternalProject_Add(nsp_nemo_speech
    SOURCE_DIR "${NSP_UPSTREAM_DIR}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/ns"
    INSTALL_DIR "${NSP_STAGE_DIR}/nemo-speech"
    DEPENDS nsp_sentencepiece
    LIST_SEPARATOR |
    CMAKE_ARGS
        ${NSP_COMMON_ARGS}
        ${NSP_FORWARD_ARGS}
        ${NSP_LLAMA_CPP_ARGS}
        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_INSTALL_BINDIR=bin
        # Linux picks up the static archive from this prefix; Windows and macOS
        # resolve the same library through CMAKE_PREFIX_PATH.
        "-DNEMO_SPEECH_DEPENDENCY_PREFIX=${NSP_STAGE_DIR}"
        "-DCMAKE_PREFIX_PATH=${NSP_STAGE_DIR}/sentencepiece"
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG> --parallel
    # Start from an empty prefix so libraries of a previous configuration
    # (e.g. a since-disabled cuBLAS shim) never reach the wheel.
    INSTALL_COMMAND "${CMAKE_COMMAND}" -E rm -rf <INSTALL_DIR>
            COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR> --config $<CONFIG>
    # Rebuild on every outer build so edits under vendor/ are picked up.
    BUILD_ALWAYS TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE)

# ---- Package layout ----
set(NSP_UPSTREAM_PREFIX "${NSP_STAGE_DIR}/nemo-speech")
if(WIN32)
    # The MSVC runtime that upstream stages for its own packages is left to the
    # system Visual C++ Redistributable (bundling unmangled copies into a
    # process that may already hold another version is unsafe).
    install(DIRECTORY "${NSP_UPSTREAM_PREFIX}/bin/"
        DESTINATION "${NSP_PACKAGE_DIR}/lib"
        FILES_MATCHING PATTERN "*.dll"
        PATTERN "msvcp*" EXCLUDE
        PATTERN "vcruntime*" EXCLUDE
        PATTERN "concrt*" EXCLUDE
        PATTERN "vccorlib*" EXCLUDE
        PATTERN "vcomp*" EXCLUDE)
else()
    # One copy per library under its SONAME / install name; see the script.
    install(CODE "
        set(NSP_LIB_SOURCE_DIR \"${NSP_UPSTREAM_PREFIX}/lib\")
        set(NSP_LIB_DESTINATION \"${NSP_PACKAGE_DIR}/lib\")")
    install(SCRIPT cmake/install_shared_libs.cmake)
endif()

if(NSP_CUDA_EULA)
    install(FILES "${NSP_CUDA_EULA}"
        DESTINATION "${NSP_PACKAGE_DIR}/licenses/third_party/cuda"
        RENAME EULA.txt)
endif()

# Build description for nemo_speech.build_info().
include(cmake/build_info.cmake)
nsp_write_build_info("${CMAKE_BINARY_DIR}/_build_info.json")
install(FILES "${CMAKE_BINARY_DIR}/_build_info.json" DESTINATION "${NSP_PACKAGE_DIR}")

# Model catalog used by nemo_speech.models (same file the upstream CLI reads).
install(FILES "${NSP_UPSTREAM_DIR}/models/index.json"
    DESTINATION "${NSP_PACKAGE_DIR}"
    RENAME model-index.json)

install(DIRECTORY "${NSP_UPSTREAM_PREFIX}/share/licenses/nemo-speech/"
    DESTINATION "${NSP_PACKAGE_DIR}/licenses")
install(FILES "${NSP_SENTENCEPIECE_DIR}/LICENSE"
    DESTINATION "${NSP_PACKAGE_DIR}/licenses/third_party/sentencepiece")
install(FILES "${NSP_SENTENCEPIECE_DIR}/third_party/absl/LICENSE"
    DESTINATION "${NSP_PACKAGE_DIR}/licenses/third_party/sentencepiece"
    RENAME absl-LICENSE)
install(FILES "${NSP_SENTENCEPIECE_DIR}/third_party/darts_clone/LICENSE"
    DESTINATION "${NSP_PACKAGE_DIR}/licenses/third_party/sentencepiece"
    RENAME darts-clone-LICENSE)
install(FILES "${NSP_SENTENCEPIECE_DIR}/third_party/protobuf-lite/LICENSE"
    DESTINATION "${NSP_PACKAGE_DIR}/licenses/third_party/sentencepiece"
    RENAME protobuf-lite-LICENSE)
