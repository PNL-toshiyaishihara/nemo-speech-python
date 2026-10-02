# install(SCRIPT) helper for ELF and Mach-O platforms.
#
# Wheels cannot store symlinks, so the libfoo.so -> libfoo.so.1 -> libfoo.so.1.2
# chains of a CMake install would be duplicated as full copies. Install each
# library once, under the name its dependents record: the SONAME
# (libfoo.so.1) on Linux, the install name (libfoo.1.dylib) on macOS.
# Unversioned libraries are installed as they are.
#
# Inputs (set via install(CODE) before this script): NSP_LIB_SOURCE_DIR,
# NSP_LIB_DESTINATION (relative to the install prefix).

set(_dest "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${NSP_LIB_DESTINATION}")
file(MAKE_DIRECTORY "${_dest}")
if(APPLE)
    set(_versioned "^(lib[^.]+)\\.[0-9]+\\.dylib$")
    set(_unversioned "^(lib[^.]+)\\.dylib$")
    file(GLOB _libs "${NSP_LIB_SOURCE_DIR}/*.dylib")
else()
    set(_versioned "^(lib.+)\\.so\\.[0-9]+$")
    set(_unversioned "^(lib.+)\\.so$")
    file(GLOB _libs "${NSP_LIB_SOURCE_DIR}/*.so" "${NSP_LIB_SOURCE_DIR}/*.so.*")
endif()

set(_stems_with_version "")
foreach(_lib IN LISTS _libs)
    get_filename_component(_name "${_lib}" NAME)
    if(_name MATCHES "${_versioned}")
        list(APPEND _stems_with_version "${CMAKE_MATCH_1}")
    endif()
endforeach()

foreach(_lib IN LISTS _libs)
    get_filename_component(_name "${_lib}" NAME)
    set(_install OFF)
    if(_name MATCHES "${_versioned}")
        set(_install ON)
    elseif(_name MATCHES "${_unversioned}")
        list(FIND _stems_with_version "${CMAKE_MATCH_1}" _index)
        if(_index EQUAL -1)
            set(_install ON)
        endif()
    endif()
    if(_install)
        file(REAL_PATH "${_lib}" _real)
        message(STATUS "Installing: ${_dest}/${_name}")
        file(COPY_FILE "${_real}" "${_dest}/${_name}")
    endif()
endforeach()
