# Qwen3.8-Flash-Next (qwen4exp). Host only: the n-gram hash against its fixture, and the n-gram
# volume reader on a synthetic volume. GPU, real artifact (skips without
# NINFER_TEST_QWEN4EXP_ARTIFACT): the whole forward pass, teacher-forced scoring and block taps.
ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_ngram_volume_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_volume.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)
set_tests_properties(ninfer_qwen4_exp_forward_real_test PROPERTIES SKIP_RETURN_CODE 77)

# Host only: the expert cache (LFRU against the replay tool's decisions, frames, the controller and
# the warm-start state file) and the startup memory plans (RAM ledger, VRAM sizing, control law).
ninfer_add_test(ninfer_qwen4_exp_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_cache.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_memory_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_memory_plan.cpp"
  LIBRARIES ninfer_model_loading ninfer_core)

# GPU, real artifact (skips without NINFER_TEST_QWEN4EXP_ARTIFACT): the M1 gates through the public
# Engine (placement invariance across cold, warm, warm-started and small expert caches; the RAM
# ledger's refusal; the elastic KV pool's grow/shrink cycle within the VRAM budget) and, with
# --bench, the decode measurements.
ninfer_add_test(ninfer_qwen4_exp_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_qwen4_exp_engine_real_test PROPERTIES SKIP_RETURN_CODE 77)
