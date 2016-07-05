library(testthat)
if (identical(Sys.getenv("TRAVIS"), "true") || identical(Sys.getenv("R_EXPENSIVE_TEST_OK"), "true")) {
  #test_check("mlr", filter = "stack_NA")
  test_check("mlr", filter = "stack_boost")
  test_check("mlr", filter = "stack_base")
  test_check("mlr", filter = "aggregatePredictions")
  test_check("mlr", filter = "stack_re")
}
