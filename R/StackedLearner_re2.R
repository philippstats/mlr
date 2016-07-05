#' Create a classif or regr Task
#' 
#' @param type [\code{character(1)}]\cr Use "classif" for classification and "regr" for regression. 
#' @param data [\code{data.frame}]\cr  Data to use.
#' @param target [\code{character(1)}]\cr Target to use.
#' @param id [\code{character(1)}] Task id.
 
createTask = function(type, data, target, id = deparse(substitute(data))) {
  if (type == "classif") {
    task = makeClassifTask(id = id, data = data, target = target)
  } else if (type == "regr") {
    task = makeRegrTask(id = id, data = data, target = target)
  }
  task
}

#' Get precise type of the task, i.e. distinguish between "classif" (binary target), "multiclassif" and all other types.
#' 
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

#' Create predictions for testing set (with base model or saved one in RDS).
#' 
#' @param i Current fold number.
#' @param bls base.learner to use.
#' @param test.idx idx for subsetting.
#' @param task task.
#' @param save.on.disc wether model are present in \code{bls} or must be loaded using readRDS.

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



#' Create new parset.
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
