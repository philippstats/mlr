# stacking where we crossval the training set with the base learners, then super-learn on that
stackCV = function(learner, task) {
  
  td = getTaskDescription(task)
  type = ifelse(td$type == "regr", "regr",
                ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))
  bls = learner$base.learners
  bpt = unique(extractSubList(bls, "predict.type"))
  use.feat = learner$use.feat
  # cross-validate all base learners and get a prob vector for the whole dataset for each learner
  rin = makeResampleInstance(learner$resampling, task = task)
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stacking", show.info = FALSE)
  exportMlrOptions(level = "mlr.stacking")
  results = parallelMap(doResampleTrain, bls, more.args = list(task, rin), impute.error = function(x) x, level = "mlr.stacking")
  
  base.models = lapply(results, function(x) x[["base.models"]])
  pred.data = lapply(results, function(x) try(getResponse(x[["resres"]]$pred, full.matrix = FALSE))) # mulitclass: all; classif: only pos

  if (type == "multiclassif" && bpt == "prob") { #FIXME: only for "stats" methods
    pred.data = lapply(pred.data, function(x) x[, -1])
  }
                         
  names(pred.data) = names(bls)
  names(base.models) = names(bls)
  tn = getTaskTargetNames(task)
  pred.data[[tn]] = results[[1]]$resres$pred$data$truth
  
  # Remove FailureModels which would occur problems later
  broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pd = which(unlist(lapply(pred.data, function(x) anyNA(x))))
  broke.idx = unique(broke.idx.bm, broke.idx.pd)
  
  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    base.models = base.models[-broke.idx]
    pred.data = pred.data[-broke.idx]
  }
  # convert list to
  if (type == "regr" | type == "classif") {
    pred.data = as.data.frame(pred.data)
  } else {
    pred.data = as.data.frame(lapply(pred.data, function(X) X)) #X[,-ncol(X)]))
  }
  if (use.feat) {
    # add data with normal features IN CORRECT ORDER
    org.feat = getTaskData(task)#[test.inds, ]
    org.feat = org.feat[, !colnames(org.feat) %in% tn, drop = FALSE]
    pred.data = cbind(pred.data, org.feat)
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data, target = tn)
  } else {
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data, target = tn)
  }
  #message(getTaskDescription(task))
  #message(na_count(getTaskData(super.task)))
  messagef("Super learner '%s' will be trained with %s features and %s observations", learner$super.learner$id, getTaskNFeats(super.task), getTaskSize(super.task))
  super.model = train(learner$super.learner, super.task)
  
  list(method = "stack.cv", base.models = base.models,
       super.model = super.model, pred.train = pred.data)
}
