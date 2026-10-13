# nemo_speech/_build_info.json: how a wheel was built, read back by
# nemo_speech.build_info() and by the release-notes generator.
#
# Source revisions come from git when the vendored trees are checkouts, and
# otherwise from vendor/revisions.json, which the sdist workflow records
# (an sdist carries no git metadata).

include("${CMAKE_CURRENT_LIST_DIR}/materialize_llama_cpp.cmake")  # nsp_is_git_toplevel

# {"commit": ..., "describe": ...} of one checkout, or {} without git metadata.
function(nsp_source_revision dir out_var)
    set(json "{}")
    nsp_is_git_toplevel("${dir}" is_git)
    if(is_git)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${dir}" rev-parse HEAD
            OUTPUT_VARIABLE commit OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${dir}" describe --tags --always
            OUTPUT_VARIABLE describe OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(JSON json SET "${json}" commit "\"${commit}\"")
        string(JSON json SET "${json}" describe "\"${describe}\"")
    endif()
    set(${out_var} "${json}" PARENT_SCOPE)
endfunction()

# CMAKE_CUDA_ARCHITECTURES as ggml-cuda compiles them. It replaces some plain
# architectures with their architecture-specific "a" forms, whose code runs on
# exactly that compute capability: 12X upstream, and also 100 and 110 with
# NeMo-Speech.cpp's patch series. The rule is read from the ggml-cuda
# CMakeLists.txt this build uses, so it follows patch updates.
function(nsp_built_cuda_architectures out_var)
    if(NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR)
        set(llama_cpp "${NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR}")
    elseif(NSP_PATCHED_LLAMA_CPP_DIR)
        set(llama_cpp "${NSP_PATCHED_LLAMA_CPP_DIR}")
    else()
        set(llama_cpp "${NSP_UPSTREAM_DIR}/llama.cpp")
    endif()
    set(ggml_cuda "${llama_cpp}/ggml/src/ggml-cuda/CMakeLists.txt")
    file(STRINGS "${ggml_cuda}" rule REGEX "if \\(ARCH MATCHES \"")
    list(LENGTH rule count)
    if(NOT count EQUAL 1 OR NOT rule MATCHES "MATCHES \"([^\"]+)\"")
        message(FATAL_ERROR
            "cannot find ggml-cuda's architecture rewrite in ${ggml_cuda}; "
            "update nsp_built_cuda_architectures in cmake/build_info.cmake")
    endif()
    set(pattern "${CMAKE_MATCH_1}")
    set(requested "$CACHE{CMAKE_CUDA_ARCHITECTURES}")
    set(built "")
    foreach(arch IN LISTS requested)
        if(arch MATCHES "${pattern}")
            string(REGEX REPLACE "^([0-9]+)" "\\1a" arch "${arch}")
        endif()
        list(APPEND built "${arch}")
    endforeach()
    set(${out_var} "${built}" PARENT_SCOPE)
endfunction()

function(nsp_json_bool value out_var)
    if(value)
        set(${out_var} true PARENT_SCOPE)
    else()
        set(${out_var} false PARENT_SCOPE)
    endif()
endfunction()

# Writes the build description to `out_file`.
function(nsp_write_build_info out_file)
    find_package(Git QUIET)
    if(EXISTS "${CMAKE_SOURCE_DIR}/vendor/revisions.json")
        file(READ "${CMAKE_SOURCE_DIR}/vendor/revisions.json" sources)
    else()
        set(sources "{}")
        foreach(entry IN ITEMS
                "NeMo-Speech.cpp|${NSP_UPSTREAM_DIR}"
                "llama.cpp|${NSP_UPSTREAM_DIR}/llama.cpp"
                "sentencepiece|${NSP_SENTENCEPIECE_DIR}")
            string(REPLACE "|" ";" entry "${entry}")
            list(GET entry 0 name)
            list(GET entry 1 dir)
            nsp_source_revision("${dir}" revision)
            if(revision STREQUAL "{}")
                message(WARNING
                    "No revision for ${name}: ${dir} is not a git checkout and "
                    "vendor/revisions.json is missing; build_info() will lack it")
            endif()
            string(JSON sources SET "${sources}" "${name}" "${revision}")
        endforeach()
    endif()

    set(info "{}")
    string(JSON info SET "${info}" version "\"${SKBUILD_PROJECT_VERSION_FULL}\"")
    string(JSON info SET "${info}" variant "\"${NSP_VARIANT_NAME}\"")
    string(JSON info SET "${info}" system "\"${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}\"")
    string(JSON info SET "${info}" compiler
        "\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\"")

    set(backends "{}")
    foreach(backend IN ITEMS CUDA VULKAN METAL)
        string(TOLOWER "${backend}" key)
        nsp_json_bool("${GGML_${backend}}" flag)
        string(JSON backends SET "${backends}" "${key}" "${flag}")
    endforeach()
    string(JSON info SET "${info}" backends "${backends}")
    # The CPU backend is built per CPU generation and chosen at run time.
    nsp_json_bool("${GGML_CPU_VARIANTS_PROXY}" cpu_variants)
    string(JSON info SET "${info}" cpu_variants "${cpu_variants}")

    set(components "{}")
    foreach(component IN ITEMS ASR DIAR TTS NMT)
        string(TOLOWER "${component}" key)
        nsp_json_bool("${NEMO_SPEECH_BUILD_${component}}" flag)
        string(JSON components SET "${components}" "${key}" "${flag}")
    endforeach()
    string(JSON info SET "${info}" components "${components}")

    # Optional TTS tokenizers; their data ships in nemo_speech/data.
    set(tokenizers "{}")
    foreach(language IN ITEMS JA ZH)
        string(TOLOWER "${language}" key)
        nsp_json_bool("${NEMO_SPEECH_TTS_WITH_${language}}" flag)
        string(JSON tokenizers SET "${tokenizers}" "${key}" "${flag}")
    endforeach()
    string(JSON info SET "${info}" tts_tokenizers "${tokenizers}")

    if(GGML_CUDA)
        set(cuda "{}")
        string(JSON cuda SET "${cuda}" toolkit "\"${CUDAToolkit_VERSION}\"")
        # What is compiled, not what was requested.
        nsp_built_cuda_architectures(archs)
        string(JSON cuda SET "${cuda}" architectures "\"${archs}\"")
        nsp_json_bool("${NEMO_SPEECH_CUBLAS_SHIM}" shim)
        string(JSON cuda SET "${cuda}" cublas_shim "${shim}")
        string(JSON info SET "${info}" cuda "${cuda}")
    endif()

    nsp_json_bool("${NEMO_SPEECH_GGML_PATCHED}" patched)
    string(JSON info SET "${info}" llama_cpp_patched "${patched}")
    string(JSON info SET "${info}" sources "${sources}")
    file(WRITE "${out_file}" "${info}\n")
endfunction()
