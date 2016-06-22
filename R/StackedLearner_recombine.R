#' Recombine stacking with new super.learner or new parset for "ens.sel". This is in alpha.
#' 
#' Instead of compute a whole new resampling procedure just use \code{recombine}. 
#' \code{recombine} reuse the already done work from \code{resample}, i.e. 
#' reuse already fitted base models and reuse level 1 data. Note: This function 
#' does not support resample objects with single broken base models (no error 
#' handling). Moreover models need to present (i.e. save.preds = TRUE in 
#' akeStackedLearner). 
#' This function does three things internally to obtain the new predictions.
#' 1. Obtain Level 1 Data for training set. This is needed to train new superlearner 
#' or new ensemble selection.
#' 2. Train superlearner or ensemble selection using level 1 data.
#' 3. Create Level 1 Data for the test set (may be several test sets in case of CV),
#' i.e. predict with saved models and test data. Returned Predictions are Level 1 Data.
#' 4. Apply model from (2) on Level 1 Data from (3).
#' 
#' @param id [\code{character(1)}]\cr Id.
#' @param obj [\code{ResampleResult}]\cr Object using \code{StackedLearner} as Learner.
#' @param super.learner [\code{Learner}]\cr New \code{super.learner} to apply.
#' @param use.feat [\code{logical(1)}]\cr Whether the original features should also be passed to the super learner.
#' @param parset [\code{list}]\cr List containing parameter for \code{hill.climb}. See \code{\link{makeStackedLearner}}.
#' @template arg_task
#' @template arg_measures
#' @export
#' @examples 
#' tsk = pid.task
#' bls = list(makeLearner("classif.kknn", id = "k1"), 
#'   makeLearner("classif.randomForest", id = "f1"),
#'   makeLearner("classif.rpart", id = "r1", minsplit = 5),
#'   makeLearner("classif.rpart", id = "r2", minsplit = 10),
#'   makeLearner("classif.rpart", id = "r3", minsplit = 15),
#'   makeLearner("classif.rpart", id = "r4", minsplit = 20),
#'   makeLearner("classif.rpart", id = "r5", minsplit = 25)
#' )
#' bls = lapply(BLS, function(x) setPredictType(x, predict.type = "prob"))
#' ste = makeStackedLearner(id = "stack", bls, resampling = cv3, 
#'   predict.type = "prob", method = "hill.climb", parset = list(init = 1, 
#'   bagprob = 0.5, bagtime = 3, metric = mmce))
#' resres = resample(ste, tsk, cv5, models = TRUE) 
#' re2 = recombine(obj = resres, task = tsk, parset = list(init = 2))
#' re3 = recombine(obj = resres, task = tsk, measures = list(mmce), parset = list(prob = .2))
#' re3 = recombine(obj = resres, task = tsk, measures = mmce, parset = list(prob = .2))
#' re4 = recombine(obj = resres, task = tsk, measures = list(acc), parset = list(bagtime = 10))
#' re5 = recombine(obj = resres, task = tsk, measures = list(mmce, acc), parset = list(init = 2, prob = .7, bagtime = 10))
#' 
#' sapply(list(resres, re2, re3, re4, re5), function(x) x$runtime)
#' sapply(list(resres, re2, re3, re4, re5), function(x) x$aggr)



recombine = function(id = NULL, obj, task, measures = NULL, super.learner = NULL, use.feat = NULL, parset = NULL) {
  ### checks 
  if (is.null(id))
    id = paste("recombined", super.learner$id, collapse = ".")
  assertClass(id, "character")
  assertClass(obj, "ResampleResult")
  assertClass(task, "Task") # check if tasks from obj and tasks fits
  if (is.null(measures))
    measures = list(getDefaultMeasure(task))
  if (class(measures) == "Measure")
    measures = list(measures)
  lapply(measures, function(x) assertClass(x, "Measure"))
  if (!is.null(super.learner)) {
    assertClass(super.learner, "Learner")
    if (is.null(use.feat)) {
      use.feat = FALSE
    }
    assertLogical(use.feat)  
    assertClass(parset, "NULL")
  }
  if (!is.null(parset)) {
    assertClass(parset, "list")
  }

  # method
  org.method = obj$models[[1]]$learner$method
  if (!is.null(super.learner)) {
    assertClass(super.learner, "Learner")
    method = "stack.cv"
  } else {
    assertClass(parset, "list")
    method = "hill.climb"
  }
  if (org.method == "stack.cv" & method == "hill.climb") {
    stopf("Ensemble Selection cannot be applied on top of method 'stack.cv'.")
  }
  ### 
  type = getTaskType(task) 
  tn = getTaskTargetNames(task)
  bls.length = length(obj$models[[1]]$learner.model$base.models)
  bls.names = names(obj$models[[1]]$learner.model$base.models)
  folds = length(obj$models)
  #bms.pt = unique(unlist(lapply(seq_len(bls.length), function(x) obj$models[[1]]$learner.model$base.models[[x]]$learner$predict.type)))
  bms.pt = unique(unlist(lapply(seq_len(bls.length), function(x) obj$models[[1]]$learner$base.learners[[x]]$predict.type)))
  if (length(bms.pt) > 1) stopf("All Base Learner must be of the same predict.type.")
  task.size = getTaskSize(task)
  save.on.disc = obj$models[[1]]$learner$save.on.disc
  
  time1 = Sys.time()
  
  # get base models (for save.on.disc=FALSE: WrappedModels; for TRUE: names directing to RData on disc.)
  base.models = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$base.models)
  # get train and test idxs
  train.idxs = lapply(seq_len(folds), function(x) obj$models[[x]]$subset)
  test.idxs = lapply(seq_len(folds), function(x) setdiff(seq_len(task.size), train.idxs[[x]]))
  #
  test.level1.preds = vector("list", length = folds)
  #
  if (method == "stack.cv") { ### stack.cv ###
    ### get TEST level 1 preds, i.e. apply bls models from training on testing data
    # saved as "test.level1.preds"
    for (i in seq_len(folds)) {
      test.idx = test.idxs[[i]]
      preds = createTestPreds(i, base.models[[i]], test.idx = test.idx, task, save.on.disc)
      #preds = lapply(seq_len(length(base.models[[i]])), function(b) predict(base.models[[i]][[b]], subsetTask(task, idxs)))
      test.level1 = lapply(seq_len(length(preds)), function(x) getResponse(preds[[x]], full.matrix = TRUE))
      names(test.level1) = bls.names
      if (bms.pt == "prob") { #remove first column for predict.type="prob"
        test.level1 = lapply(test.level1, function(x) x[, -1])
      }
      if (use.feat) {
        feat = getTaskData(task, subset = test.idx, target.extra = TRUE)$data
        test.level1[["feat"]] = feat
      }
      test.level1[[tn]] = getTaskTargets(task)[test.idx]
      test.level1.data = as.data.frame(test.level1)
      test.level1.task = createTask(type, data = test.level1.data, target = tn)
      test.level1.preds[[i]] = test.level1.task
      rm(preds, test.level1.data, test.level1.task)
    }
    ### apply super.learner on TRAIN level 1 tasks to obtain models
    # data saved as "train.level1.datas" (one list entry per fold)
    # task saved as "train.level1.tasks" (-"-)

    # If pred.train are a data.frames (i.e. method=stack.cv) then train.level1.datas
    # are already a list of data.frame. 
    # If pred.train are predictions (i.e. method=hill.climb) then they must be 
    # converted using getResponse.
    train.level1.datas = lapply(obj$models, function(x) x$learner.model$pred.train)
    if (org.method == "hill.climb") { # pred.train are predictions, no data.frame
      data.list = vector("list", length = folds)
      for (f in seq_len(folds)) {
        train.idx = train.idxs[[f]]
        tmp = lapply(seq_len(bls.length), function(x) getResponse(train.level1.datas[[f]][[x]], full.matrix = TRUE))
        names(tmp) = bls.names
        if (bms.pt == "prob") { # remove first column for predict.type="prob" (i.e. kind of classif)
          tmp = lapply(tmp, function(x) x[, -1])
        }
        if (use.feat) {
        feat = getTaskData(task, subset = train.idx, target.extra = TRUE)$data
        tmp[["feat"]] = feat
        }
        tmp[[tn]] = getTaskTargets(task)[train.idx]
        data.list[[f]] = as.data.frame(tmp)
      }
      train.level1.datas = data.list
    }
    train.level1.tasks = lapply(train.level1.datas, function(d) createTask(type, d, target = tn))
    for (f in seq_len(folds)) {
      train.level1.tasks[[f]]$task.desc$id = paste0("fold", f)
    }
    rm(train.level1.datas)
    # fit super.learner on every fold
    #-train.supermodel = benchmark(super.learner, tasks = train.level1.tasks, resampling = makeResampleDesc("Holdout", predict = "train", split = 1), measures = measures)
    #+ here tuneParam could be applied
    train.supermodel = lapply(seq_len(folds), function(x) train(super.learner, task = train.level1.tasks[[x]]))
    ### apply train.supermodel from line above on TEST level 1 preds 
    train.preds = lapply(seq_len(folds), function(i) predict(train.supermodel[[i]], test.level1.preds[[i]]))
    res.model = train.supermodel
  } else { 
    ### end stack.cv / start hill.climb ###
    # use settings from new parset
    parset = createNewParset(org.parset = obj$models[[1]]$learner$parset, new.parset = parset)
    ### get TRAIN level 1 preds and perfs
    # saved as "train.level1.preds"
    # saved as "train.level1.perfs"
    train.level1.preds = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$pred.train)
    train.level1.perfs = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$bls.performance)
    # 
    train.preds = vector("list", length = folds)
    res.model = vector("list", length = folds)
    for (i in seq_len(folds)) {
      ### get TEST level 1 preds, i.e. apply bls models from training on testing data
      # saved as "test.level1.preds"
      test.level1.preds[[i]] = createTestPreds(i, base.models[[i]], test.idx = test.idxs[[i]], task, save.on.disc)
      ### train with new parameters
      res.model[[i]] = applyEnsembleSelection(bls.length = bls.length, 
        bls.names = bls.names, pred.list = train.level1.preds[[i]], 
        bls.performance = train.level1.perfs[[i]], parset = parset)
      freq = res.model[[i]]$freq
      current.pred.list = test.level1.preds[[i]]
      #names(current.pred.list) = bls.names
      current.pred.list = expandPredList(current.pred.list, freq = freq)
      train.preds[[i]] = aggregatePredictions(pred.list = current.pred.list)
    }
  }
  ### measures and runtime
  m = lapply(train.preds, function(x) performance(x, measures = measures))
  measure.test = as.data.frame(do.call(rbind, m))
  measure.test = cbind(iter = 1:NROW(measure.test), measure.test)
  aggr = colMeans(measure.test[, -1, drop = FALSE])
  names(aggr) = paste0(names(aggr), ".test.mean")
  
  time2 = Sys.time()
  runtime = as.numeric(difftime(time2, time1, "sec"))
  
  ### return
  X = list(learner.id = id, task.id = task$task.desc$id, measure.test = measure.test, aggr = aggr, train.preds = train.preds, 
       runtime = runtime, super.learner = super.learner, parset = parset, res.model = res.model)
  class(X) = "RecombinedResampleResult"
  return(X)
}


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