# Test declarations are included from tests/CMakeLists.txt so executable paths and
# CTest working directories stay under build/tests.
#
# Tests are programs in the ninfer_tests bundle (cmake/NinferBundles.cmake): one executable instead
# of one per test, each carrying its own copy of the kernel image. Run one by hand with
# `ninfer_tests <name> [args...]`. STANDALONE keeps a test in its own executable, for a test whose
# link options would leak into every other program (symbol wrapping) or that a script invokes as
# an executable path.
include(${PROJECT_SOURCE_DIR}/cmake/NinferBundles.cmake)

function(ninfer_test_includes target)
  ninfer_internal_includes(${target})
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

# The runtime DLL directories a test process needs, as CTest ENVIRONMENT_MODIFICATION entries:
# cudart64_13, cublas64_13 and cublasLt64_13 live in the toolkit's bin/x64 (not bin/), and zlib in
# the vcpkg tree's bin. They are reached through PATH, which a launcher that is not a native Windows
# shell (Git Bash) hands over mangled wherever a directory contains a space, and which a test started
# by hand may not have at all -- CTest then reports 0xc0000135 (STATUS_DLL_NOT_FOUND) for every test,
# which reads as a failure of the test rather than of its environment. Appending to whatever PATH the
# launcher already had keeps ctest correct from any shell. Empty off Windows.
function(ninfer_test_dll_path out_var)
  set(_mods "")
  if(WIN32)
    get_filename_component(_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    set(_dirs "${_cuda_bin}/x64" "${_cuda_bin}")
    foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
      if(EXISTS "${_prefix}/bin")
        list(APPEND _dirs "${_prefix}/bin")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _dirs)
    foreach(_dir IN LISTS _dirs)
      file(TO_NATIVE_PATH "${_dir}" _native)
      list(APPEND _mods "PATH=path_list_append:${_native}")
    endforeach()
  endif()
  set(${out_var} "${_mods}" PARENT_SCOPE)
endfunction()

function(ninfer_add_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "NEEDS_SOURCE_DIR;STANDALONE" ""
    "SOURCES;LIBRARIES;TEST_ARGS")
  if(arg_STANDALONE)
    add_executable(${name} ${arg_SOURCES})
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
    set(command ${name})
  else()
    ninfer_bundle_program(ninfer_tests ${name} SOURCES ${arg_SOURCES} LIBRARIES ${arg_LIBRARIES})
    ninfer_bundle_command(command ninfer_tests ${name})
  endif()
  ninfer_test_includes(${name})
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  # TEST_ARGS lets a test that takes a configuration on the command line run something this box can
  # actually fit. Without it a test defaults to whatever its argv fallbacks are, which for the
  # real-model tests is a maximum layout that no single 24 GB card can hold -- so the test fails
  # permanently and stops being read. The executable still accepts any other configuration when run
  # by hand.
  add_test(NAME ${name} COMMAND ${command} ${arg_TEST_ARGS})
  # The runtime DLLs a test needs -- cudart64_13, cublas64_13 and cublasLt64_13 from the toolkit's
  # bin/x64, zlib from the vcpkg tree -- are found through PATH, which a launcher that is not a
  # native Windows shell (Git Bash) hands over mangled wherever a directory contains a space, and
  # which a test started by hand from anywhere else may not have at all. CTest would then report
  # 0xc0000135 (STATUS_DLL_NOT_FOUND) for every test, which reads as a failure of the test rather
  # than of its environment. Give each test the directories itself, appended to whatever PATH the
  # launcher already had, so ctest is correct from any shell.
  ninfer_test_dll_path(_ninfer_path_mods)
  if(_ninfer_path_mods)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT_MODIFICATION "${_ninfer_path_mods}")
  endif()
endfunction()

# Apply these to the translation unit containing the oracle, including shared
# test support libraries. Executable options do not propagate into those libraries.
function(ninfer_op_oracle_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninfer_add_op_test name)
  ninfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  ninfer_op_oracle_options(${name})
endfunction()
