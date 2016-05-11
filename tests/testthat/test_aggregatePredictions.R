context("aggregatePredictions")
'
source("~/Documents/Rdevelop/mlr/tests/testthat/helper_objects.R")
keepPatternEnv("task")
setwd("~/Documents/Rdevelop")
library(devtools)
load_all("mlr")
'

# classif
pts = c("prob", "response")
tasks = list(binaryclass.task, multiclass.task)

for (tsk in tasks) {
  for (pt in pts) {
    for (spt in pts) {
      #messagef("This is %s, %s, %s \n", tsk$task.desc$id, pt, spt)
      lrn0 = makeLearner("classif.rpart", predict.type = pt)
      lrn1 = makeLearner("classif.kknn", predict.type = pt)
      lrn2 = makeLearner("classif.kknn", k = 10, predict.type = pt)
  
      tr0 = train(lrn0, tsk)
      tr1 = train(lrn1, tsk)
      tr2 = train(lrn2, tsk)
      pr0 = predict(tr0, tsk)
      pr1 = predict(tr1, tsk)
      pr2 = predict(tr2, tsk)
      
      pred.list = list(pr0, pr1, pr2)
      p = aggregatePredictions(pred.list, spt = spt)
      p #%>% print
      perf = performance(p) #%>% print
      expect_that(perf < 1/2, is_true())
    }
  }
}

# regr
tasks = list(regr.task, regr.small.task, regr.num.task)
lrn0 = makeLearner("regr.rpart")
lrn1 = makeLearner("regr.kknn")
lrn2 = makeLearner("regr.kknn", k = 10, predict.type = pt)

for (tsk in tasks) {
  #messagef("This is %s\n", tsk$task.desc$id)
  tr0 = train(lrn0, tsk)
  tr1 = train(lrn1, tsk)
  tr2 = train(lrn2, tsk)
  pr0 = predict(tr0, tsk)
  pr1 = predict(tr1, tsk)
  pr2 = predict(tr2, tsk)
  
  pred.list = list(pr0, pr1, pr2)
  p = aggregatePredictions(pred.list)
  p #%>% print
  perf = performance(p) #%>% print
  expect_that(perf > 0.1, is_true())
}
