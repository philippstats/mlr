context("stack_NAs")

test_that("Error handling if learner crashs (classif)", {
  configureMlr(on.learner.error = "warn", on.learner.warning = "quiet")
  iris = iris
  iris$newfeat = 1 # will make LDA crash
  mc.task = makeClassifTask(data = iris, target = "Species")
  
  pid = getTaskData(pid.task)[1:100, ]
  pid$newfeat = 1 # will make LDA crash
  sp.task = makeClassifTask(data = pid, target = "diabetes")
  tasks = list(mc.task, sp.task)
  pts = c("prob", "response")

  bls = list(
   makeLearner("classif.rpart"),
   makeLearner("classif.lda"),
   makeLearner("classif.randomForest"))
 
  # average
  context("average")
  #load_all("mlr"); tsk = mc.task; spt = "prob"; bpt =  "response"
  for (tsk in tasks) {
    for (spt in pts) {
      for (bpt in pts) {
        messagef("> %s, %s, %s", getTaskId(tsk), spt, bpt)
        configureMlr(on.learner.error = "quiet", on.learner.warning = "quiet")
        bls = lapply(bls, setPredictType, bpt)
        sta = makeStackedLearner(id = "stack", bls, predict.type = spt, method = "average")
        #debugonce(trainLearner.StackedLearner)
        #debugonce(averageBaseLearners)
        #m = train(sta, tsk)
        #debugonce(predictLearner.StackedLearner)
        #p = predict(m, subsetTask(tsk, 95:105))
        r = resample(sta, tsk, cv2, models = TRUE)
        expect_equal(length(r$models[[1]]$learner.model$base.models), 2) # check if only 2 models got returned
        expect_that(r$aggr, is_a("numeric"))
        expect_that(r, is_a("ResampleResult"))
      }
    }
  }
  
  #stack.cv
  context("stack.cv")
  for (spt in pts) {
    for (bpt in pts) {
      messagef("> %s, %s", spt, bpt)
      configureMlr(on.learner.error = "quiet", on.learner.warning = "quiet")
      bls = lapply(bls, setPredictType, bpt)
      slr = makeLearner("classif.kknn")
      stc = makeStackedLearner(id = "stack", bls, slr, predict.type = spt, method = "stack.cv")
      #debugonce(trainLearner.StackedLearner)
      #debugonce(stackCV)
      #m = train(stc, tsk)
      #expect_equal(length(m$learner.model$base.models), 2)
      #expect_equal(length(m$learner$base.learners), 3)
      #debugonce(predictLearner.StackedLearner)
      #p = predict(m, subsetTask(tsk, 1:5))
      r = resample(stc, tsk, cv2, model = TRUE)
      expect_equal(length(r$models[[1]]$learner.model$base.models), 2) # check if only 2 models got returned
      expect_that(r$aggr, is_a("numeric"))
      expect_that(r, is_a("ResampleResult")) 
    }
  }
 
  #hill.climb
  context("hill.climb")
  bagtimes = c(1, 2, 5)
  for (tsk in tasks) {
    for (spt in pts) {
      for (bt in bagtimes) {
        configureMlr(on.learner.error = "quiet", on.learner.warning = "quiet")
        messagef("> %s, %s, %s", getTaskId(tsk), spt, bt)
        bls = lapply(bls, setPredictType, "prob")
        stc = makeStackedLearner(id = "stack", bls, predict.type = spt, method = "hill.climb",
          parset = list(bagtime = bt))
        r = resample(stc, tsk, cv2, model = TRUE)
        expect_equal(length(r$models[[1]]$learner.model$base.models), 2) # check if only 2 models got returned
        expect_that(r$aggr, is_a("numeric")) 
        expect_that(r, is_a("ResampleResult")) 
        #Sys.sleep(2)
      }
    }
  }
})
