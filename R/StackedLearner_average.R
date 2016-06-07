# super simple averaging of base-learner predictions without weights. we should beat this
averageBaseLearners = function(learner, task) {
  bls = learner$base.learners
  id = learner$id
  save.on.disc = learner$save.on.disc
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stacking", show.info = FALSE)
  exportMlrOptions(level = "mlr.stacking")
  show.info = getMlrOption("show.info")
  results = parallelMap(doTrainPredict, bls, more.args = list(task, show.info, id, save.on.disc), impute.error = function(x) x, level = "mlr.stacking")

  #base.models = lapply(results, function(x) x[["base.models"]])
  base.models = lapply(results, function(x) x[["base.models"]])
  pred.data = lapply(results, function(x) try(getResponse(x[["pred"]], full.matrix = TRUE), silent = TRUE)) #QUEST: Return Predictions instead of data.frame!?
  
  #names(base.models) = names(bls)
  names(base.models) = names(bls)
  names(pred.data) = names(bls)
  
  #FIXME: I don't know if it is the nicest way to remove bls 
  #broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pd1 = which(unlist(lapply(pred.data, function(x) anyNA(x)))) # if models is FailesModels and NAs are returend in a Prediction
  broke.idx.pd2 = which(unlist(lapply(pred.data, function(x) class(x) %nin% c("numeric", "factor", "data.frame")))) # if model fails and error message is returned it is not class numeric (regr, binary classif) nor data.frame (multiclassif)
  #broke.idx = unique(broke.idx.bm, broke.idx.pd)
  broke.idx = unique(c(broke.idx.pd1, broke.idx.pd2))
  
  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    base.models = base.models[-broke.idx]
    pred.data = pred.data[-broke.idx]
  }
  
  #list(method = "average", base.models = base.models, super.model = NULL,
  list(method = "average", base.models = base.models, super.model = NULL,
       pred.train = pred.data)
}
