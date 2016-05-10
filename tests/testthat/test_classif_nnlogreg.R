'library(devtools)
load_all("mlr")
lrn = makeLearner("classif.nnlogreg", predict.type = "prob")
debugonce(mlr:::trainLearner.classif.nnlogreg)
debugonce(mlr:::nnlogreg)
tr = train(lrn, subsetTask(pid.task, 1:400))
trX
debugonce(mlr:::predictLearner.classif.nnlogreg)
pr = predict(tr, subsetTask(pid.task, 401:765))
performance(pr)

.newdata = getTaskData(pid.task, 401:4, target.extra = TRUE)$data



lrn0 = makeLearner("classif.logreg", predict.type = "prob")
tr0 = train(lrn0, subsetTask(pid.task, 1:400))
debugonce(mlr:::predictLearner.classif.logreg)
pr0 = predict(tr0, subsetTask(pid.task, 401:405))
pr0


data = getTaskData(pid.task)
d = data$diabetes
d
as.numeric(d)
f = formula  = getTaskFormula(pid.task, explicit.features = TRUE)
f

data = mtcars
data$am = factor(data$am)
data$cyl = factor(data$cyl)
data$gear = factor(data$gear)
data$carb = factor(data$carb)

mt.task = makeClassifTask(data = data, target = "am")

lrn = makeLearner("classif.nnlogreg", predict.type = "prob")
tr = train(lrn, subsetTask(mt.task, 1:32))
pr = predict(tr, subsetTask(mt.task, 26:32))
re = resample(tr)


data = getTaskData(pid.task)
data$pregnant = factor(data$pregnant)
p.task = makeClassifTask(data = data, target = "diabetes")

lrn = makeLearner("classif.nnlogreg", predict.type = "response")
tr = train(lrn, subsetTask(p.task, 1:500))

#debugonce(mlr:::predictLearner.classif.nnlogreg)
pr = predict(tr, subsetTask(p.task, 501:765))
pr
performance(pr)
r = resample(lrn, p.task, cv10, models = T)
'

context("classif_nnlogreg")

test_that("classif_nnlogreg", {
  # create data with factors from pid.task
  data = getTaskData(pid.task)
  data$pregnant = factor(data$pregnant)
  p.task = makeClassifTask(data = data, target = "diabetes")
  
  lrn = makeLearner("classif.nnlogreg")

  pts = c("prob", "response")
  for (pt in pts) {
    lrn = setPredictType(lrn, pt)
    m = train(lrn, subsetTask(p.task, 1:500))
    p = predict(tr, subsetTask(p.task, 501:765))
    perf = performance(p)
    # approximate value if prediction might be useful
    expect_that(perf < 1/3, is_true())
    r = resample(lrn, p.task, cv10, models = TRUE)
    # Typical error in resample (is catched by try):
    # Error in solve.default(IB) : Lapack routine dgesv: system is exactly singular: U[16,16] = 0
    expect_that(r$aggr < 1/3, is_true())
    #lapply(r$models, function(x) anyNA(x$learner.model$beta))
  }
})
