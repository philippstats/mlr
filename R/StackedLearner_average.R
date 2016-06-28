# simple averaging of baselearner predictions without weights
averageBaseLearners = function(learner, task) {
  id = learner$id
  save.on.disc = learner$save.on.disc
  save.preds = learner$save.preds 
  bls = learner$base.learners
  bls.names = names(bls)

  # parallelMap: train, predict
  parallelLibrary("mlr", master = FALSE, level = "mlr.stacking", show.info = FALSE)
  exportMlrOptions(level = "mlr.stacking")
  show.info = getMlrOption("show.info")
  results = parallelMap(doTrainPredict, bls, more.args = list(task, show.info, id, save.on.disc), impute.error = function(x) x, level = "mlr.stacking")

  base.models = lapply(results, function(x) x[["base.models"]])
  pred.list = lapply(results, function(x) x[["pred"]])
  names(base.models) = bls.names
  names(pred.list) = bls.names

  ##FIXME: Ok way to remove bls1?
  #broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  ##broke.idx.pd1 = which(unlist(lapply(pred.data, function(x) anyNA(x)))) # if models is FailesModels and NAs are returend in a Prediction
  ##broke.idx.pd2 = which(unlist(lapply(pred.data, function(x) class(x) %nin% c("numeric", "factor", "data.frame")))) # if model fails and error message is returned it is not class numeric (regr, binary classif) nor data.frame (multiclassif)
  ##broke.idx = unique(c(broke.idx.pd1, broke.idx.pd2))
  #broke.idx = broke.idx.bm
  #
  #if (length(broke.idx) > 0) {
  #  messagef("Base Learner %s is broken and will be removed\n", bls.names[broke.idx])
  #  base.models = base.models[-broke.idx]
  #  pred.list = pred.list[-broke.idx]
  #  #pred.data = pred.data[-broke.idx]
  #}
  
  # return
  if (save.preds == FALSE) pred.list = NULL
  list(method = "average", base.models = base.models, super.model = NULL,
    pred.train = pred.list)
}
