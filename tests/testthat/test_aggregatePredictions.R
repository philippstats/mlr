context("aggregatePredictions")

# classif
pts = c("prob", "response")
tasks = list(binaryclass.task, multiclass.task)

for (tsk in tasks) {
  for (pt in pts) {
    lrn0 = makeLearner("classif.rpart", predict.type = pt)
    lrn1 = makeLearner("classif.kknn", predict.type = pt)
    lrn2 = makeLearner("classif.svm", predict.type = pt)

    tr0 = train(lrn0, tsk)
    tr1 = train(lrn1, tsk)
    tr2 = train(lrn2, tsk)
    pr0 = predict(tr0, tsk)
    pr1 = predict(tr1, tsk)
    pr2 = predict(tr2, tsk)
    
    pred.list = list(pr0, pr1, pr2)
    p = aggregatePredictions(pred.list)
    perf = performance(p) #%>% print
    expect_that(perf < 1/3, is_true())
  }
}

# regr
tasks = list(regr.task, regr.small.task, regr.num.task)
lrn0 = makeLearner("regr.rpart")
lrn1 = makeLearner("regr.kknn")
lrn2 = makeLearner("regr.svm")

for (tsk in tasks) {
  tr0 = train(lrn0, tsk)
  tr1 = train(lrn1, tsk)
  tr2 = train(lrn2, tsk)
  pr0 = predict(tr0, tsk)
  pr1 = predict(tr1, tsk)
  pr2 = predict(tr2, tsk)
  
  pred.list = list(pr0, pr1, pr2)
  p = aggregatePredictions(pred.list)
  perf = performance(p) #%>% print
  expect_that(perf > 0.1, is_true())
}
