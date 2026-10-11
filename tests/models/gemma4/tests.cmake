ninfer_add_test(ninfer_gemma4_kernels_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kernels.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_gemma4_kernels_test PROPERTIES SKIP_RETURN_CODE 77)
