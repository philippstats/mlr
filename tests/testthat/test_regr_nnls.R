context("regr_nnls")

test_that("regr_nnls", {
  lrn = makeLearner("regr.nnls")
  m = train(lrn, subsetTask(regr.task, regr.train.inds))
  expect_that(length(m$learner.model$x), 14)

  p = predict(m, subsetTask(regr.task, regr.test.inds))
})