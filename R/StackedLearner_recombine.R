#' Recombine stacking with new super learner (not yet implemented for "hill.climb")
#' 
#' Instead of train a new super learner in a resample just use recombine 
#' which reuse the already done work from resample, i.e. reuse base models, 
#' reuse level 1 data. This function does not support resample objects with single 
#' broken base models.
#' 
#' @param id [\code{character(1)}]\n Id.
#' @param obj [\code{ResampleResult]\n Object using \code{StackedLearner} as Learner.
#' @param super.learner [\code{Learner}]\n New \code{super.learner} to apply.
#' @template arg_task
#' @template arg_measures
#' @export

recombine = function(id = NULL, obj, super.learner = NULL, task, measures, parset) {
  ### checks 
  if (is.null(id))
    id = paste("recombined", super.learner$id, sep = ".")
  assertClass(id, "character")
  assertClass(obj, "ResampleResult")
  #assertChoice(method, c("stack.cv", "hill.climb"))
  if (!is.null(super.learner)) 
    assertClass(super.learner, "Learner")
  assertClass(task, "Task") # check if tasks from obj and tasks fits
  lapply(measures, function(x) assertClass(x, "Measure"))
  
  # method
  method = obj$models[[1]]$learner$method
  if (method == "stack.cv" & any(class(super.learner) != "Learner"))
    stopf("Method 'stack.cv' needs 'super.learner'")
  #FIXME parset
  
  ### pre
  type = getTaskType(task) 
  tn = getTaskTargetNames(task)
  bls.length = length(obj$models[[1]]$learner.model$base.models)
  bls.names = names(obj$models[[1]]$learner.model$base.models)
  folds = length(obj$models)
  bms.pt = unique(unlist(lapply(seq_len(bls.length), function(x) obj$models[[1]]$learner.model$base.models[[x]]$learner$predict.type)))
  task.size = getTaskSize(task)
  
  time1 = Sys.time()
  #
  if (method == "stack.cv") {
    assertCharacter(method, "stack.cv")
    ### get TEST level 1 data, i.e. apply bls from traning on testing data
    # get base models
    base.models = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$base.models)
    # get test idxs
    train.idxs = lapply(seq_len(folds), function(x) obj$models[[x]]$subset)
    test.idxs = lapply(seq_len(folds), function(x) setdiff(1:task.size, train.idxs[[x]]))
    # train base models to obtain level 1 data for test parts
    test.level1.preds = vector("list", length = folds)
    for (i in seq_len(folds)) {
      idxs = test.idxs[[i]]
      preds = lapply(seq_len(length(base.models[[i]])), function(b) predict(base.models[[i]][[b]], subsetTask(task, idxs)))
      test.level1 = lapply(seq_len(length(preds)), function(x) getResponse(preds[[x]], full.matrix = TRUE))
      if (bms.pt == "prob") { #remove first column for predict.type="prob"
        test.level1 = lapply(test.level1, function(x) x[, -1])
      }
      names(test.level1) = bls.names
      test.level1[[tn]] = getTaskTargets(task)[idxs]
      
      test.level1.data = as.data.frame(test.level1)
      test.level1.task = createTask(type, data = test.level1.data, target = tn)
      test.level1.preds[[i]] =  test.level1.task
    }
    
    ### apply super.learner learner on TRAIN level 1 data to obtain models
    train.superlearner = retrainSuperLearner(obj, super.learner) # FIXME not whole obj
    ### apply models from above on TEST level 1 data
    test.superlearner.preds = vector("list", length = folds)
    for (i in seq_len(folds)) {
      test.superlearner.preds[[i]] = predict(train.superlearner[[i]], test.level1.preds[[i]])
    }
    final.preds = test.superlearner.preds
    
    m = lapply(test.superlearner.preds, function(x) performance(x, measures = measures))

  } else { # end stack.cv / start hill.climb
    assertCharacter(method, pattern = "hill.climb")
    org.parset = obj$models[[1]]$learner$parset
    use.org = setdiff(names(org.parset), names(parset))
    org.parset = org.parset[use.org]
    parset = c(org.parset, parset)
    
    preds = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$pred.train)
    perfs = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$bls.performance)
#browser()    
    # get base models
    base.models = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$base.models)
    # get test idxs
    train.idxs = lapply(seq_len(folds), function(x) obj$models[[x]]$subset)
    test.idxs = lapply(seq_len(folds), function(x) setdiff(1:task.size, train.idxs[[x]]))
    ## predict on new test data
    # train base models to obtain level 1 data for test parts
    test.level1.preds = vector("list", length = folds)
    for (i in seq_len(folds)) {
      idxs = test.idxs[[i]]
      tmp.preds = lapply(seq_len(length(base.models[[i]])), function(b) predict(base.models[[i]][[b]], subsetTask(task, idxs)))
      thenames = unlist(lapply(seq_len(length(base.models[[i]])), function(b) base.models[[i]][[b]]$learner$id))
      names(tmp.preds) = thenames
      test.level1.preds[[i]] = tmp.preds
      }
    final.preds = vector("list", length = folds)
    ### train with new parameters
    for (i in seq_len(folds)) {
      ensel = applyEnsembleSelection(bls.length = bls.length, 
        bls.names = bls.names, pred.list = preds[[i]], 
        bls.performance = perfs[[i]], parset = parset)
      
      freq = ensel$freq
      current.pred.list = test.level1.preds[[i]]
      #names(current.pred.list) = bls.names
      current.pred.list = expandPredList(current.pred.list, freq = freq)
      final.preds[[i]] = aggregatePredictions(pred.list = current.pred.list)
    }
    m = lapply(final.preds, function(x) performance(x, measures = measures))
    
  }
      

  ### measures and runtime
  measure.test = as.data.frame(do.call(rbind, m))
  measure.test = cbind(iter = 1:NROW(measure.test), measure.test)
  aggr = colMeans(measure.test[, -1, drop = FALSE])
  names(aggr) = paste0(names(aggr), ".test.mean")
  
  time2 = Sys.time()
  runtime = as.numeric(difftime(time2, time1, "sec"))
  
  ### return
  X = list(learner.id = id, task.id = tsk$task.desc$id, measure.test = measure.test, aggr = aggr, pred = final.preds, 
       runtime = runtime)
  class(X) = "RecombinedResampleResult"
  X
}

#' Retrain super learner on new test task
#' 
#' @param obj [\code{ResampleResult]\n Object using \code{StackedLearner} as Learner.
#' @param super.learner [\code{Learner}]\n New \code{super.learner} to apply.

retrainSuperLearner = function(obj, super.learner) {
  assertClass(obj, "ResampleResult")
  assertClass(super.learner, "Learner")
  # train new super.learner (we need level1 data and then train the super.learner model)
  train.level1 = extractLevel1Task(obj)
  train.superlearner = vector("list", length = length(train.level1))
  for (i in seq_along(train.level1)) {
    train.superlearner[[i]] = train(super.learner, train.level1[[i]])
  }
  train.superlearner
}


#' Extract level 1 data from stacking resample and create a task
#' 
#' @param obj [\code{ResampleResult]\n Object using \code{StackedLearner} as Learner.
#' @param as.task [\code{logical(1)]\n Specify if task or data.frame should be returned 

extractLevel1Task = function(obj, as.task = TRUE) {
  assertClass(obj, "ResampleResult")
  assertLogical(as.task)
  type = obj$models[[1]]$task.desc$type
  target = obj$models[[1]]$task.desc$target
  
  if (is.null(obj$models)) 
    stopf("Resample needs models. Use argument 'model = TRUE' in your resample function.")
  datas = lapply(obj$models, function(x) x$learner.model$pred.train)
  tasks = lapply(datas, function(d) createTask(type, d, target))
  if (as.task) {
    return(tasks)
  } else {
    return(datas)
  }
}

#' Create a Classif or Regr Task
#' 
#' @param type [\code{character(1)]\n Use "classif" for Classification and "regr" for regression. 
#' @param data [\code{data.frame]\n  Data to use.
#' @param target [\code{character(1)]\n Target to use.
 
createTask = function(type, data, target) {
  if (type == "classif") {
    task = makeClassifTask(data = data, target = target)
  } else {
    task = makeRegrTask(data = data, target = target)
  }
  task
}

getPreciseTaskType = function(task) {
  type = getTaskType(task)
  if (type == "classif" & length(task$task.desc$class.levels) > 2)
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