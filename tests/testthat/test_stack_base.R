context("stack_base")

checkStack = function(task, method, base, super, bms.pt, sm.pt, use.feat) {
  base = lapply(base, makeLearner, predict.type = bms.pt)
  if (method %in% c("aggregate", "ensembleselection")) {
    super = NULL
  } else {
    super = makeLearner(super, predict.type = sm.pt)
  }
  if (method == "ensembleselection" && bms.pt == "response" && inherits(task, "ClassifTask")) return()

  stk = makeStackedLearner(id = "stack", base.learner = base, super.learner = super, 
    method = method, use.feat = use.feat, predict.type = sm.pt, save.preds = T)
  tr = train(stk, task)
  pr = predict(tr, task)

  if (sm.pt == "prob") {
    expect_equal(ncol(pr$data[,grepl("prob", colnames(pr$data))]), length(getTaskClassLevels(task)))
  }

  if (method %nin% c("superlearner", "ensembleselection")) {
    expect_equal(
      lapply(getStackedBaseLearnerPredictions(tr), function(x) getPredictionResponse(x)),
      lapply(getStackedBaseLearnerPredictions(tr, newdata = getTaskData(task)), function(x) getPredictionResponse(x))
    )
  }
}

test_that("Base functions", {
  tasks = list(binaryclass.task, multiclass.task, regr.task)
  for (task in tasks) {
    td = getTaskDescription(task)
    if (inherits(task, "ClassifTask")) {
      pts = c("response", "prob")
      base = c("classif.rpart", "classif.randomForest", "classif.kknn")
      super = "classif.randomForest"
    } else {
      pts = "response"
      base = c("regr.rpart", "regr.lm", "regr.kknn")
      super = "regr.randomForest"
    }
    for (method in c("aggregate", "superlearner", "ensembleselection")) {
      ufs = if (method %in% c("aggregate", "ensembleselection")) FALSE else c(FALSE, TRUE)
      for (use.feat in ufs) {
        for (sm.pt in pts) {
          for (bms.pt in pts) {
            #cat(td$type, td$id, method, use.feat, sm.pt, bms.pt, fill = TRUE)
            checkStack(task, method, base, super, bms.pt, sm.pt, use.feat)
          }
        }
      }
    }
  }
})