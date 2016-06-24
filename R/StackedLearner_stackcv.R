#' Stacking method stackCV uses super.learner to obtain level 1 data and uses inner cross-validation for prediction.
#'
#' @param learner [\{code(StackedLearner)}]
#' @template arg_task

stackCV = function(learner, task) {
  # 
  td = getTaskDescription(task)
  type = getPreciseTaskType(td)
  bls = learner$base.learners
  bls.names = names(bls)
  bpt = unique(extractSubList(bls, "predict.type"))
  use.feat = learner$use.feat
  id = learner$id
  save.on.disc = learner$save.on.disc
  
  # 
  rin = makeResampleInstance(learner$resampling, task = task)
  # resampling, train (parallelMap)
  parallelLibrary("mlr", master = FALSE, level = "mlr.stacking", show.info = FALSE)
  exportMlrOptions(level = "mlr.stacking")
  show.info = getMlrOption("show.info")
  results = parallelMap(doTrainResample, bls, more.args = list(task, rin, 
    measures = getDefaultMeasure(task), show.info, id, save.on.disc), 
    impute.error = function(x) x, level = "mlr.stacking")
  
  base.models = lapply(results, function(x) x[["base.models"]])
  pred.data = lapply(results, function(x) try(getResponse(x[["resres"]]$pred, 
    full.matrix = FALSE), silent = TRUE)) # mulitclass: all; classif: only pos

  names(pred.data) = bls.names
  names(base.models) = bls.names
  
  # Remove broken models/predictions
  #broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pd1 = which(unlist(lapply(pred.data, function(x) anyNA(x))))
  broke.idx.pd2 = which(unlist(lapply(pred.data, function(x) class(x) %nin% c("numeric", "factor", "data.frame"))))
  #broke.idx = unique(broke.idx.bm, broke.idx.pd)
  broke.idx = unique(c(broke.idx.pd1, broke.idx.pd2))
  
  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    base.models = base.models[-broke.idx]
    pred.data = pred.data[-broke.idx]
  }

  # remove 1st prediction for multiclassif due to multicollinearity reason (alternative: use getResponse TRUE, and always remove first row)
  if (type == "multiclassif" && bpt == "prob") { #FIXME: only for "stats" methods
    pred.data = lapply(pred.data, function(x) x[, -1])
  }
  # add true value
  tn = getTaskTargetNames(task)
  pred.data[[tn]] = results[[1]]$resres$pred$data$truth
  # convert list to data.frame
  pred.data = as.data.frame(pred.data)

  if (use.feat) {
    # add data with normal features IN CORRECT ORDER
    org.feat = getTaskData(task)
    org.feat = org.feat[, !colnames(org.feat) %in% tn, drop = FALSE]
    pred.data = cbind(pred.data, org.feat)
  } 
  super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data, target = tn)
  #print(head(super.task))
  messagef("[Super Learner] Train %s with %s features on %s observations", learner$super.learner$id, getTaskNFeats(super.task), getTaskSize(super.task))
  super.model = train(learner$super.learner, super.task)
  # return
  list(method = "stack.cv", base.models = base.models,
       super.model = super.model, pred.train = pred.data)
}
