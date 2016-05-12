context("stack_NAs")

test_that("Error handling if learner crashs (classif)", {
  configureMlr(on.learner.error = "quiet")
  data = iris
  data$newfeat = 1 # will make LDA crash
  mc.task = makeClassifTask(data = data, target = "Species")
  
  data = getTaskData(pid.task)[1:100, ]
  data$newfeat = 1 # will make LDA crash
  sp.task = makeClassifTask(data = data, target = "diabetes")
  tasks = list(mc.task, sp.task)
  pts = c("prob", "response")
  
  bls = list(
   makeLearner("classif.rpart"),
   makeLearner("classif.lda"),
   makeLearner("classif.randomForest"))
  
  # average
  for (tsk in tasks) {
    for (spt in pts) {
      for (bpt in pts) {
        bls = lapply(bls, setPredictType, bpt)
        sta = makeStackedLearner(bls, predict.type = spt, method = "average")
        r=resample(sta, tsk, cv2)
        expect_that(r$aggr, is_a("numeric"))
        expect_that(r, is_a("ResampleResult"))
      }
    }
  }
  
  #stack.cv
  for (spt in pts) {
    for (bpt in pts) {
      bls = lapply(bls, setPredictType, bpt)
      slr = makeLearner("classif.logreg")
      stc = makeStackedLearner(bls, slr, predict.type = spt, method = "stack.cv")
      r = resample(stc, sp.task, cv2, extract = function(x) x$learner.model)
      expect_that(r$aggr, is_a("numeric"))
      expect_that(r, is_a("ResampleResult")) 
    }
  }
  
  #hill.climb
  bagtimes = c(1, 2, 5)
  for (spt in pts) {
    for (bt in bagtimes) {
      bls = lapply(bls, setPredictType, "prob")
      stc = makeStackedLearner(bls, predict.type = spt, method = "hill.climb2",
        parset = list(bagtime = bt))
      r = resample(stc, sp.task, cv2)
      expect_that(r$aggr, is_a("numeric")) 
      expect_that(r, is_a("ResampleResult")) 
    }
  }
})
