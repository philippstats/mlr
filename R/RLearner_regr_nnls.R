makeRLearner.regr.nnls = function() {
  makeRLearnerRegr(
    cl = "regr.nnls",
    package = "nnls",
    par.set = makeParamSet(),
    properties = c("numerics", "factors"),
    name = "Non-Negative Least Squares",
    short.name = "nnls",
    note = "Dummy variables are generated for factor features. No intercept available"
  )
}

trainLearner.regr.nnls = function(.learner, .task, .subset, .weights = NULL) {
  task = createDummyFeatures(.task)
  data = getTaskData(task, .subset, target.extra = TRUE)
  nnls::nnls(A = as.matrix(data$data), b = as.numeric(data$target))
}

predictLearner.regr.nnls = function(.learner, .model, .newdata, ...) {
  coef = coef(.model$learner.model)
  X = createDummyFeatures(.newdata)
  as.vector(coef %*% t(X))
}
