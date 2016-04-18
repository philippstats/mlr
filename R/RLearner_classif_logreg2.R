#' @export
makeRLearner.classif.logreg2 = function() {
  makeRLearnerClassif(
    cl = "classif.logreg2",
    package = "stats",
    par.set = makeParamSet(),
    properties = c("twoclass", "numerics", "factors", "prob", "weights", "missings"),
    name = "Logistic Regression",
    short.name = "logreg",
    note = 'Delegates to `glm` with `family = binomial(link = "logit")`.'
  )
}

#' @export
trainLearner.classif.logreg2 = function(.learner, .task, .subset, .weights = NULL,  ...) {
  f = getTaskFormula(.task)
  stats::glm(f, data = getTaskData(.task, .subset), model = FALSE, family = "binomial", na.action = na.omit, ...)
}

#' @export
predictLearner.classif.logreg2 = function(.learner, .model, .newdata, ...) {
  x = predict(.model$learner.model, newdata = .newdata, type = "response", ...)
  levs = .model$task.desc$class.levels
  if (.learner$predict.type == "prob") {
    propVectorToMatrix(x, levs)
  } else {
    levs = .model$task.desc$class.levels
    p = as.factor(ifelse(x > 0.5, levs[2L], levs[1L]))
    unname(p)
  }
}