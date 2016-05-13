#library(devtools); library(dplyr); library(checkmate); library(testthat)
#setwd("~/Documents/Rdevelop")
#rm(list = ls())
#load_all("mlr")
#
#source('~/Documents/Rdevelop/mlr/tests/testthat/helper_objects.R')
#li = setdiff(ls(), c("binaryclass.task", "multiclass.task", "multiclass.small.task", "regr.num.task", "regr.task"))
#rm(list = li); rm(li)
#task = tsk = pid.task
#task = tsk = iris.task

#bpt = "prob"
#spt = "prob"

context("stack_boost")

test_that("Parameters for makeBoostedStackingLearner (classif)", {
  tasks_classif = list(binaryclass.task, multiclass.task) 
  ctrl = makeTuneControlRandom(maxit = 3L)
  pts = c("prob", "response")

  #spt = "prob"; bpt = "prob"
  for (tsk in tasks_classif) {
    for (spt in pts) {
      for (bpt in pts) {
        #context(paste(tsk$task.desc$id, spt, bpt))
        lrns = list(
          makeLearner("classif.gbm"),
          makeLearner("classif.randomForest"))
        lrns = lapply(lrns, setPredictType, bpt)
        mm = makeModelMultiplexer(lrns)
        ps = makeModelMultiplexerParamSet(mm,
          #makeNumericParam("sigma", lower = -5, upper = 5, trafo = function(x) 2^x),
          #makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x),
          makeIntegerParam("n.trees", lower = 1L, upper = 500L),
          makeIntegerParam("interaction.depth", lower = 1L, upper = 10L),
          makeIntegerParam("ntree", lower = 1L, upper = 500L),
          makeIntegerParam("mtry", lower = 1L, upper = getTaskNFeats(tsk)))
        stb = makeBoostedStackingLearner(model.multiplexer = mm, 
          predict.type = spt, 
          resampling = cv5,
          mm.ps = ps, 
          measures = mmce, 
          control = ctrl, 
          niter = 2L)
        n.obs = getTaskSize(tsk)
        train.set = 1:as.integer(n.obs * 0.8)
        test.set =  as.integer(n.obs * 0.8 + 1):n.obs
        #model = train(stb, subsetTask(tsk, subset = train.set))
        #res = predict(model, subsetTask(tsk, subset = test.set))
        #res
        #performance(res)
        
        r = resample(stb, task = tsk, resampling = cv2, models = TRUE, show.info = TRUE)
        expect_is(r$aggr, "numeric")
        if (spt == "prob") {
          p = getPredictionProbabilities(r$pred, cl = tsk$task.desc$class.levels)
          expect_that(dim(p), is_identical_to(c(getTaskSize(tsk), length(levels(getTaskTargets(tsk))))))
        } else {
          p = getPredictionResponse(r$pred)
          expect_class(p, "factor")
          expect_equal(length(p), (getTaskSize(tsk)))
        }
      }
    }
  }
})
  
#test_that("Parameters for boost.stack model (regr)", {
#
#  tasks_regr = list(regr.num.task, regr.task)
#  lrns = list(
#    #makeLearner("classif.ksvm", kernel = "rbfdot"),
#    makeLearner("regr.gbm"),
#    makeLearner("regr.randomForest"))
#  mm = makeModelMultiplexer(lrns)
#  ctrl = makeTuneControlRandom(maxit = 3L)
#  for (tsk in tasks_regr) {
#    ps = makeModelMultiplexerParamSet(mm,
#      #makeNumericParam("sigma", lower = -5, upper = 5, trafo = function(x) 2^x),
#      #makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x),
#      makeIntegerParam("n.trees", lower = 1L, upper = 500L),
#      makeIntegerParam("interaction.depth", lower = 1L, upper = 10L),
#      makeIntegerParam("ntree", lower = 1L, upper = 500L),
#      makeIntegerParam("mtry", lower = 1L, upper = getTaskNFeats(tsk)))
#    stb = makeBoostedStackingLearner(model.multiplexer = mm, 
#      predict.type = "response", #because regr  
#      resampling = cv3,
#      mm.ps = ps, 
#      measures = mse, 
#      control = ctrl, 
#      niter = 2L)
#      n.obs = getTaskSize(tsk)
#    n.obs = getTaskSize(tsk)
#    train.set = sort(sample(n.obs, 0.7 * n.obs))
#    test.set =  setdiff(1:n.obs, train.set)
#    #model = train(stb, subsetTask(tsk, subset = train.set))
#    #pre = predict(model, subsetTask(tsk, subset = test.set))
#    r = resample(stb, task = tsk, resampling = cv3)
#    expect_is(r, "ResampleResult")
#  }  
#})
