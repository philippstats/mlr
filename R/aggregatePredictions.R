'library(parallelMap)
parallelStartMulticore(2)    # start in socket mode and create 2 processes on localhost
f = function(i) i + 5     # define our job
y = parallelMap(f, 1:2)   # like Rs Map but in parallel
parallelStop() 

for (i in seq_along(bls)) {
    bl = bls[[i]]
    model = train(bl, task)
    base.models[[i]] = model
    #
    pred = predict(model, task = task)
    probs[[i]] = getResponse(pred, full.matrix = TRUE)
}

library(mlr)
tsk = subsetTask(iris.task, subset = c(46:50, 70:100, 130:150))
tsk$env$data[1,1] = NA


data = getTaskData(pid.task)
data$CONT = 1
tsk = makeClassifTask(data = data, target = "diabetes"); rm(data)
tsk = subsetTask(tsk, 1:50)

tsk = bh.task

pt = "response"
lrn0 = makeLearner("classif.rpart", predict.type = pt)
lrn1 = makeLearner("classif.kknn", predict.type = pt)
lrn2 = makeLearner("classif.svm", predict.type = pt)

lrn0 = makeLearner("regr.rpart", predict.type = pt)
lrn1 = makeLearner("regr.kknn", predict.type = pt)
lrn2 = makeLearner("regr.svm", predict.type = pt)

rm(tr0); rm(tr1)
tr0 = train(lrn0, tsk)
tr1 = train(lrn1, tsk)
tr2 = train(lrn2, tsk)

rm(pr0); rm(pr1)
pr0 = predict(tr0, tsk)
pr1 = predict(tr1, tsk)
pr2 = predict(tr2, tsk)
pr0
pr1
pr2

rm(pred.list)
pred.list = list(pr0, pr1, pr2)
pred.list'

#' Aggregate Predictions
#' 
#' Aggregate predicitons results by averaging (for \code{regr}, and  \code{classif} with prob) or mode ( \code{classif} with response). 
#' (works for regr, classif, multiclass)
#' 
#' @param pred.list [list of \code{Predictions}]\cr
#' @export


aggregatePredictions = function(pred.list) {
  # Check if "equal"
  x = lapply(pred.list, function(x) getTaskDescription(x))
  task.unequal = unlist(lapply(2:length(x), function(i) !all.equal(x[[1]], x[[i]])))
  if (any(task.unequal)) stopf("Task descriptions in prediction '1' and '%s' differ. This is not possible!", which(task.unequal)[1])

  x = lapply(pred.list, function(x) x$predict.type)
  pts.unequal = unlist(lapply(2:length(x), function(i) !all.equal(x[[1]], x[[i]])))
  if (any(pts.unequal)) stopf("Predict type in prediction '1' and '%s' differ. This is not possible!",  which(pts.unequal)[1])
  
  x = unlist(lapply(pred.list, function(x) checkIfNullOrAnyNA(x$data$response)))
  if (any(x)) messagef("Prediction '%s' is broken and will be removed.", which(x))
  pred.list = pred.list[!x]
  
  # Body
  pred1 = pred.list[[1]]
  type = getTaskType(pred1)
  td = getTaskDescription(pred1)
  rn = row.names(pred1$data)
  id = pred1$data$id
  tr = pred1$data$truth
  pt = pred1$predict.type
  ti = NA
  pred.length = length(pred.list) # FIXME if any failures
  
  # Reduce results
  if (type == "classif") {
    if (pt == "prob") {
      preds = lapply(pred.list, getPredictionProbabilities, cl = td$class.levels)
      y = Reduce("+", preds)/pred.length
    } else {
      preds = as.data.frame(lapply(pred.list, getPredictionResponse))
      y = factor(apply(preds, 1L, computeMode), td$class.levels)
    }
    return(mlr:::makePrediction(task.desc = td, rn, id = id, truth = tr, predict.type = pt, predict.threshold = NULL, y, time = ti))
  } else { # type = "regr"
    preds = lapply(pred.list, getPredictionResponse)
    y = Reduce("+", preds)/pred.length
    return(mlr:::makePrediction(task.desc = td, rn, id = id, truth = tr, predict.type = pt, predict.threshold = NULL, y, time = ti))
  }
}
