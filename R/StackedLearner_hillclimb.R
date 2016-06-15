hillclimbBaseLearners = function(learner, task, replace = TRUE, init = 1, bagprob = 1, bagtime = 1,
  metric = NULL, maxiter = NULL, tolerance = 1e-8, ...) {
  # checks, inits
  assertFlag(replace)
  assertInt(init, lower = 1, upper = length(learner$base.learners)) #807
  assertNumber(bagprob, lower = 0, upper = 1)
  assertInt(bagtime, lower = 1)
  if (is.null(metric)) metric = getDefaultMeasure(task)
  assertClass(metric, "Measure")
  
  td = getTaskDescription(task)
  type = getPreciseTaskType(task) # "regr", "classif", "multiclassif"
  bls = learner$base.learners
  id = learner$id
  save.on.disc = learner$save.on.disc
  if (is.null(maxiter)) maxiter = length(bls)
  assertInt(maxiter, lower = 1)
  assertNumber(tolerance)

  if (type != "regr") {
    if (any(extractSubList(bls, "predict.type") == "response"))
      stop("Hill climbing algorithm only takes probability predict type for classification.")
  }
  # body
  rin = makeResampleInstance(learner$resampling, task = task)
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stacking", show.info = FALSE)
  exportMlrOptions(level = "mlr.stacking")
  show.info = getMlrOption("show.info")
  results = parallelMap(doTrainResample, bls, more.args = list(task, rin, measures = metric, show.info, id, save.on.disc), 
      impute.error = function(x) x, level = "mlr.stacking")
  
  base.models = lapply(results, function(x) x[["base.models"]])
  resres = lapply(results, function(x) x[["resres"]])
  pred.list = lapply(resres, function(x) x[["pred"]])
  bls.performance = sapply(resres, function(x) x$aggr) # only use
#browser()  
  names(base.models) = names(bls)
  names(resres) = names(bls) 
  names(pred.list) = names(bls)
  names(bls.performance) = names(bls) # this will not be removed below!
  
  # Remove FailureModels which would occur problems later #FIXME!?
  #broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pl = which(unlist(lapply(pred.list, function(x) anyNA(x$data))))# FIXME?!
  broke.idx.rr = which(unlist(lapply(resres, function(x) is.na(x$aggr[1]))))
  #broke.idx = unique(c(broke.idx.bm, broke.idx.rr, broke.idx.pl))
  broke.idx = unique(c(broke.idx.rr, broke.idx.pl))

  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    resres = resres[-broke.idx]
    #pred.data = pred.data[-broke.idx]
    base.models = base.models[-broke.idx]
    pred.list = pred.list[-broke.idx]
  }

  ensel = applyEnsembleSelection(pred.list = pred.list,
    bls.length = length(base.models), bls.names = names(base.models), 
    bls.performance = bls.performance, parset = list(replace = replace, 
    init = init, bagprob = bagprob, bagtime = bagtime, maxiter = maxiter, 
    metric = metric, tolerance = tolerance))
  
  # TODO current.pred gleich freq*bls gleich 
  # pred.list is list of Predictions, no data.frame, but I would say that is not needed
  list(method = "hill.climb", base.models = base.models, super.model = NULL,
    pred.train = pred.list, bls.performance = bls.performance, 
    weights = ensel$weights, freq = ensel$freq, freq.list = ensel$freq.list)
}