context("stack_base")

checkStack = function(task, method, base, super, bms.pt, sm.pt, use.feat) {
  base = lapply(base, makeLearner, predict.type = bms.pt)
  if (method %in% c("average", "hill.climb")) {
    super = NULL
  } else {
    super = makeLearner(super, predict.type = sm.pt)
    # sm.pt = NULL
  }
  if (method == "hill.climb" && bms.pt == "response" && inherits(task, "ClassifTask")) return()

  stk = makeStackedLearner(id = "stack", base, super, method = method, use.feat = use.feat, predict.type = sm.pt)
  tr = train(stk, task)
  pr = predict(tr, task)

  if (sm.pt == "prob") {
    expect_equal(ncol(pr$data[,grepl("prob", colnames(pr$data))]), length(getTaskClassLevels(task)))
  }

  if (method %nin% c("stack.cv", "hill.climb")) {
    expect_equal(
      getStackedBaseLearnerPredictions(tr),
      getStackedBaseLearnerPredictions(tr, newdata = getTaskData(task))
    )
  }
}

test_that("Stacking works", {
  tasks = list(binaryclass.task, multiclass.task, regr.task)
  for (task in tasks) {
    td = getTaskDescription(task)
    if (inherits(task, "ClassifTask")) {
      pts = c("response", "prob")
      base = c("classif.rpart", "classif.lda", "classif.svm")
      super = "classif.randomForest"
    } else {
      pts = "response"
      base = c("regr.rpart", "regr.lm", "regr.svm")
      super = "regr.randomForest"
    }
    for (method in c("average", "stack.cv", "stack.nocv", "hill.climb")) {
      ufs = if (method %in% c("average", "hill.climb")) FALSE else c(FALSE, TRUE)
      for (use.feat in ufs) {
        for (sm.pt in pts) {
          for (bms.pt in pts) {
            cat(td$type, td$id, method, use.feat, sm.pt, bms.pt, fill = TRUE)
            #messagef(method, base, super, bms.pt, sm.pt, use.feat)
            checkStack(task, method, base, super, bms.pt, sm.pt, use.feat)
          }
        }
      }
    }
  }
})

test_that("Stacking works with wrapped learners (#687)", {
  base = c("classif.rpart")
  lrns = lapply(base, makeLearner)
  lrns = lapply(lrns, setPredictType, "prob")
  lrns[[1]] = makeFilterWrapper(lrns[[1]], fw.abs = 2)
  m = makeStackedLearner(id = "stack", base.learners = lrns, predict.type = "prob", method = "hill.climb")
})

test_that("Parameters for hill climb works", {
  tsk = binaryclass.task
  lrns = list(
    makeLearner("classif.ksvm", predict.type = "prob"),
    makeLearner("classif.randomForest", predict.type = "prob"),
    makeLearner("classif.kknn", id = "classif.knn2", predict.type = "prob", k = 2),
    makeLearner("classif.kknn", id = "classif.knn3", predict.type = "prob", k = 3),
    makeLearner("classif.kknn", id = "classif.knn4", predict.type = "prob", k = 4),
    makeLearner("classif.kknn", id = "classif.knn5", predict.type = "prob", k = 5)
  )
  for (init in c(1, 5)) {
    for(bagprob in c(0.5, 1)) {
      for (replace in c(TRUE, FALSE)) {
        for (bagtime in c(1, 2, 10)) {
          messagef("This is: %s, %s, %s, %s", init, bagprob, replace, bagtime)
          m = makeStackedLearner(id = "stack", base.learners = lrns, predict.type = "prob", 
            method = "hill.climb", parset = list(init = init, bagprob = bagprob, 
            bagtime = bagtime, replace = replace, metric = mmce))
          tmp = train(m, tsk)
          res = predict(tmp, tsk)
          max.selected = (init + length(lrns)) * bagtime
          n.selected = sum(tmp$learner.model$freq)
          
          expect_equal(anyNA(tmp$learner.model$bls.performance), FALSE)
          expect_equal(anyNA(res$data$response), FALSE)
          expect_true(n.selected <= max.selected)
        }
      }
    }
  } 
  # use other metric (auc)
  messagef("Testing metric: auc")
  m = makeStackedLearner(id = "stack", base.learners = lrns, predict.type = "prob", method = "hill.climb",
    parset = list(replace = TRUE, bagprob = 0.7, bagtime = 3, init = 2, metric = auc))
  tmp = train(m, tsk)
  res = predict(tmp, tsk)
  expect_equal(anyNA(tmp$learner.model$bls.performance), FALSE)
  expect_equal(anyNA(res$data$response), FALSE)
})

