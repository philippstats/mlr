
#' Retrain super learner on new test task
#' 
#' @param obj [\code{ResampleResult}]\cr Object using \code{StackedLearner} as Learner.
#' @param super.learner [\code{Learner}]\cr New \code{super.learner} to apply.

#'
#retrainSuperLearner = function(models, super.learner, type, target, pred.train.is.pred = is.pred) {
#  assertClass(models, "list")
#  assertClass(super.learner, "Learner")
#  # train new super.learner (we need level1 data and then train the super.learner model)
#  #train.level1 = extractLevel1Task(obj)
#  train.level1 = extractLevel1Task(models, type, target, pred.train.is.pred, as.task = TRUE)
#  train.superlearner = vector("list", length = length(train.level1))
#  for (i in seq_along(train.level1)) {
#    train.superlearner[[i]] = train(super.learner, train.level1[[i]])
#  }
#  train.superlearner
#}
#'

#' Extract level 1 data from stacking resample and create a task
#' 
#' @param obj [\code{ResampleResult}]\cr Object using \code{StackedLearner} as Learner.
#' @param as.task [\code{logical(1)}]\cr Specify if task or data.frame should be returned. 


#extractLevel1Task = function(models, type, target, pred.train.is.pred, as.task = TRUE) {
#  #assertClass(obj, "ResampleResult")
#  assertLogical(as.task)
#  #type = obj$models[[1]]$task.desc$type
#  #target = obj$models[[1]]$task.desc$target
#
#  #if (is.null(obj$models)) 
#  #  stopf("Resample needs models. Use argument model = TRUE in your resample function.")
#  datas = lapply(models, function(x) x$learner.model$pred.train)
#  if (pred.train.is.pred) {
#    for (f in folds)
#    datas = lapply(seq_len(length(datas)), function(x) getResponse(datas[[x]], full.matrix = TRUE))
#    if (type == "prob") { #remove first column for predict.type="prob"
#      datas = lapply(datas, function(x) x[, -1])
#    }
#  }
#  tasks = lapply(datas, function(d) createTask(type, d, target))
#  if (as.task) {
#    return(tasks)
#  } else {
#    return(datas)
#  }
#}

#' Create a Classif or Regr Task
#' 
#' @param type [\code{character(1)}]\cr Use "classif" for Classification and "regr" for regression. 
#' @param data [\code{data.frame}]\cr  Data to use.
#' @param target [\code{character(1)}]\cr Target to use.
 
createTask = function(type, data, target, id = deparse(substitute(data))) {
  if (type == "classif") {
    task = makeClassifTask(id = id, data = data, target = target)
  } else {
    task = makeRegrTask(id = id, data = data, target = target)
  }
  task
}

#' Get precise type of the task, which is "classif", "multiclassif" or "regr".
#' @template arg_task_or_desc
#' @export
getPreciseTaskType = function(x) {
  if (any(class(x) == "Task"))
    x = getTaskDescription(x)
  type = getTaskType(x)
  if (type == "classif" & length(x$class.levels) > 2)
    type = "multiclassif"
  type
}


#' @export
print.RecombinedResampleResult = function(x, ...) {
  cat("Recombined Resample Result\n")
  catf("Task: %s", x$task.id)
  catf("Learner: %s", x$learner.id)
  m = x$measure.test[, -1L, drop = FALSE]
  Map(function(name, x, aggr) {
    catf("%s.aggr: %.2f", name, aggr)
    catf("%s.mean: %.2f", name, mean(x, na.rm = TRUE))
    catf("%s.sd: %.2f", name, sd(x, na.rm = TRUE))
  }, name = colnames(m), x = m, aggr = x$aggr)
  catf("Runtime: %g", x$runtime)
  invisible(NULL)
}

#' Create Predictions for Testing Set
#' 
#' @param i number of fold
#' @param bls base.learner to use
#' @param test.idx idx for subsetting
#' @param task task
#' @param save.on.disc wether model are presend in bls or must be loaded using readRDS

createTestPreds = function(i, bls, test.idx, task, save.on.disc) {
    bls.len = length(bls)
    if (save.on.disc) {
    # This only works if outer resampling is Holdout (save model does not 
    # get infos about the fold figure, therefore only one fold is allowed): 
    if (i != 1) {
      stopf("Using 'save.on.disc = TRUE' and outer resampling strategies others 
        than Holdout is not supported. Switch save.on.disc to FALSE or use Holdout.")
    }
    all.preds = vector("list", length = bls.len)
    for (b in seq_len(bls.len)) { # do it seqentially
      bm = readRDS(bls[[b]]) # i is always 1
      all.preds[[b]] = predict(bm, subsetTask(task, test.idx))
      names(all.preds)[b] = bm$learner$id
      rm(bm)
    }
  } else { # save.on.disc = FALSE
    all.preds = lapply(seq_len(bls.len), function(b) predict(bls[[b]], subsetTask(task, test.idx)))
    all.preds.names = unlist(lapply(seq_len(bls.len), function(b) bls[[b]]$learner$id))
    names(all.preds) = all.preds.names
  }
  all.preds
}



#' Create new parset
#' 
#' @param org.parset orginal/old parset from obj
#' @param new.parset parameters which should be updated
 
createNewParset = function(org.parset, new.parset) {
  org.keep = setdiff(names(org.parset), names(new.parset))
  org.parset = org.parset[org.keep]
  final.parset = c(org.parset, new.parset)
  allowed =  c("replace", "init", "bagprob", "bagtime", "maxiter", "tolerance", "metric")
  unallowed = setdiff(names(new.parset), allowed) 
  if (length(unallowed) > 0) 
    stopf("'%s' is no an allowed argument for parset.", unallowed)
  final.parset
}



