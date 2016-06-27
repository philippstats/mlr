#' Rerun a alreday done outer resample for a StackedLearner again with new settings.

#' Instead of computing a whole new resampling procedure just use \code{resampleStackedLearnerAgain}. 
#' \code{resampleStackedLearnerAgain} reuses the already done work in a \code{ResampleResult}, i.e. 
#' reuse fitted base models (needed for level 1 test data) and reuse level 1 training data. 
#'Note: This function does not support resample objects with single broken base models (no error 
#' handling implemented). Moreover models need to present (i.e. save.preds = TRUE in 
#' \code{makeStackedLearner}). When using \code{save.on.disc = TRUE} in makeStackedLearner 
#' resampling procedures with holdout are allowed only (model names are not unique 
#' regarding CV fold number).
#' This function does four things internally to obtain the new predictions.
#' \describe{
#' \item{1.}{Extract level 1 train data.}
#' \item{2.}{Fit new super learner or apply new ensemble selection setting using level 1 train data.}
#' \item{3.}{Use saved base models on test data to predict level 1 test data}
#' \item{4.}{Apply model from (2) on level 1 test data from (3) to obtain final prediction}
#' }
#' For method stack.cv \code{super.learner} and \code{use.feat} need to be set. 
#' For \code{hill.climb} \code{parset} need to be set. 
#' Method \code{average} is not supported (use no inner resampling).
#' @param id [\code{character(1)}]\cr Unique ID for object
#' @param obj [\code{ResampleResult}]\cr Object using \code{StackedLearner} as learner.
#' @param super.learner [\code{Learner}]\cr New \code{super.learner} to apply.
#' @param use.feat [\code{logical(1)}]\cr Whether the original features should be passed to the super learner.
#' @param parset [\code{list}]\cr List containing parameters for \code{hill.climb}. See \code{\link{makeStackedLearner}}.
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
#' bls = lapply(bls, function(x) setPredictType(x, predict.type = "prob"))
#' ste = makeStackedLearner(id = "stack", bls, resampling = cv3, 
#'   predict.type = "prob", method = "hill.climb", parset = list(init = 1, 
#'   bagprob = 0.5, bagtime = 3, metric = mmce))
#' resres = resample(ste, tsk, cv2, models = TRUE) 
#' re2 = resampleStackedLearnerAgain(obj = resres, task = tsk, parset = list(init = 2))
#' re3 = resampleStackedLearnerAgain(obj = resres, task = tsk, measures = list(mmce), parset = list(bagprob = .2))
#' re3 = resampleStackedLearnerAgain(obj = resres, task = tsk, measures = mmce, parset = list(bagprob = .2))
#' re4 = resampleStackedLearnerAgain(obj = resres, task = tsk, measures = list(acc), parset = list(bagtime = 10))
#' re5 = resampleStackedLearnerAgain(obj = resres, task = tsk, measures = list(mmce, acc), parset = list(init = 2, bagprob = .7, bagtime = 10))
#' 
#' sapply(list(resres, re2, re3, re4, re5), function(x) x$runtime)
#' sapply(list(resres, re2, re3, re4, re5), function(x) x$aggr)



resampleStackedLearnerAgain = function(id = NULL, obj, task, measures = NULL, super.learner = NULL, use.feat = NULL, parset = NULL) {
  ### checks 
  if (is.null(id))
    id = paste("recombined", super.learner$id, collapse = ".") #TODO what about ES uniq name
  assertClass(id, "character")
  assertClass(obj, "ResampleResult")
  assertClass(task, "Task") # TODO check if tasks from obj and tasks fits
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
  #
  type = getTaskType(task) 
  tn = getTaskTargetNames(task)
  bls.length = length(obj$models[[1]]$learner.model$base.models)
  bls.names = names(obj$models[[1]]$learner.model$base.models)
  bms.pt = unique(unlist(lapply(seq_len(bls.length), function(x) obj$models[[1]]$learner$base.learners[[x]]$predict.type)))
  folds = length(obj$models)
  #bms.pt = unique(unlist(lapply(seq_len(bls.length), function(x) obj$models[[1]]$learner.model$base.models[[x]]$learner$predict.type)))
  if (length(bms.pt) > 1) stopf("All Base Learner must be of the same predict.type.")
  task.size = getTaskSize(task)
  save.on.disc = obj$models[[1]]$learner$save.on.disc
  
  time1 = Sys.time()
  
  # get base models (for save.on.disc=FALSE: WrappedModels; for TRUE: names directing to RData on disc.)
  base.models_f = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$base.models)
  # get train and test idxs
  train.idxs = lapply(seq_len(folds), function(x) obj$models[[x]]$subset)
  test.idxs = lapply(seq_len(folds), function(x) setdiff(seq_len(task.size), train.idxs[[x]]))
  #
  test.level1.task_f = vector("list", length = folds)
  #
  ### 
  ### 
  ### stack.cv ###
  ### 
  ### 
  if (method == "stack.cv") { 
    ### get TEST level 1 preds, i.e. apply bls models from training on testing data
    # saved as "test.level1.preds"
    for (f in seq_len(folds)) {
      test.idx = test.idxs[[f]]
      pred.list = createTestPreds(f, base.models_f[[f]], test.idx = test.idx, task, save.on.disc)
      #preds = lapply(seq_len(length(base.models[[i]])), function(b) predict(base.models[[i]][[b]], subsetTask(task, idxs)))
      pred.data.list = lapply(seq_len(length(pred.list)), function(x) getPredictionDataNonMulticoll(pred.list[[x]]))
      names(pred.data.list) = bls.names
      if (use.feat) {
        feat = getTaskData(task, subset = test.idx, target.extra = TRUE)$data
        pred.data.list[["feat"]] = feat
      }
      pred.data.list[[tn]] = getTaskTargets(task)[test.idx]
      pred.data = as.data.frame(pred.data.list)
      test.level1.task_f[[f]]  = createTask(type, data = pred.data, target = tn)
      rm(pred.data.list, pred.data)
    }
    ### apply super.learner on TRAIN level 1 tasks to obtain models
    # data saved as "train.level1.datas" (one list entry per fold)
    # task saved as "train.level1.tasks" (-"-)

    # If pred.train are a data.frames (i.e. method=stack.cv) then train.level1.datas
    # are already a list of data.frame. 
    # If pred.train are predictions (i.e. method=hill.climb) then they must be 
    # converted using getResponse.
#browser()
    train.level1.preds_f = lapply(obj$models, function(x) x$learner.model$pred.train)
    #if (org.method == "hill.climb") { # pred.train are predictions, no data.frame
    train.level1.task_f = vector("list", length = folds)
    for (f in seq_len(folds)) {
      train.idx = train.idxs[[f]]
      pred.data.list = lapply(seq_len(bls.length), function(x) getPredictionDataNonMulticoll(train.level1.preds_f[[f]][[x]])) # TODO naming
      names(pred.data.list) = bls.names #TODO/FIME save as above: write a fct
      if (use.feat) {
      feat = getTaskData(task, subset = train.idx, target.extra = TRUE)$data
      pred.data.list[["feat"]] = feat
      }
      pred.data.list[[tn]] = getTaskTargets(task)[train.idx]
      pred.data = as.data.frame(pred.data.list) # TODO: naming
      train.level1.task_f[[f]] = createTask(type, pred.data, target = tn, id = paste0("fold", f))
    }
#browser()
    # fit super.learner on every fold
    #-train.supermodel = benchmark(super.learner, tasks = train.level1.tasks, resampling = makeResampleDesc("Holdout", predict = "train", split = 1), measures = measures)
    #+ here tuneParam could be applied
    train.supermodel_f = lapply(seq_len(folds), function(f) train(super.learner, task = train.level1.task_f[[f]]))
    train.preds_f = lapply(seq_len(folds), function(f) predict(train.supermodel_f[[f]], task = train.level1.task_f[[f]]))
    ### apply train.supermodel from line above on TEST level 1 preds 
    test.preds_f = lapply(seq_len(folds), function(f) predict(train.supermodel_f[[f]], test.level1.task_f[[f]]))
    res.model_f = train.supermodel_f
    ### 
    ### 
    ### end stack.cv / start hill.climb ###
    ###
    ###
  } else if (method == "hill.climb") { 
    # use settings from new parset
    parset = createNewParset(org.parset = obj$models[[1]]$learner$parset, new.parset = parset)
    ### get TRAIN level 1 preds and perfs
    # saved as "train.level1.preds"
    # saved as "train.level1.perfs"
    train.level1.preds_f = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$pred.train)
    train.level1.perfs_f = lapply(seq_len(folds), function(x) obj$models[[x]]$learner.model$bls.performance)
    # 
    train.preds_f = test.preds_f = vector("list", length = folds)
    res.model_f = vector("list", length = folds)
    for (f in seq_len(folds)) {
      ### get TEST level 1 preds, i.e. apply bls models from training on testing data
      # saved as "test.level1.preds"
      test.level1.preds = createTestPreds(i, base.models_f[[f]], test.idx = test.idxs[[f]], task, save.on.disc)
      ### train with new parameters
      res.model_f[[f]] = applyEnsembleSelection(pred.list = train.level1.preds_f[[f]],
        bls.length = bls.length, bls.names = bls.names, 
        bls.performance = train.level1.perfs_f[[f]], parset = parset)
      freq = res.model_f[[f]]$freq
      # train prediction
      current.pred.list = train.level1.preds_f[[f]] # names(current.pred.list) = bls.names
      current.pred.list = expandPredList(current.pred.list, freq = freq)
      train.preds_f[[f]] = aggregatePredictions(pred.list = current.pred.list)
      # test prediction
      current.pred.list = expandPredList(test.level1.preds, freq = freq)
      test.preds_f[[f]] = aggregatePredictions(pred.list = current.pred.list)
    }
  }
  ### measures and runtime 
  m = lapply(train.preds_f, function(x) performance(x, measures = measures))
  measure.train = as.data.frame(do.call(rbind, m))
  measure.train = cbind(iter = 1:NROW(measure.train), measure.train)
  
  m = lapply(test.preds_f, function(x) performance(x, measures = measures))
  measure.test = as.data.frame(do.call(rbind, m))
  measure.test = cbind(iter = 1:NROW(measure.test), measure.test)
  aggr = colMeans(measure.test[, -1, drop = FALSE])
  names(aggr) = paste0(names(aggr), ".test.mean")
  
  time2 = Sys.time()
  runtime = as.numeric(difftime(time2, time1, "sec"))
  
  ### return
  X = list(learner.id = id, 
    task.id = task$task.desc$id, 
    measure.train = measure.train, 
    measure.test = measure.test, 
    aggr = aggr, 
    #train.preds = train.preds_f, 
    pred = test.preds_f, # not 1 ResampleRediction object as in resample but a list of single ResamplePredictions
    models = res.model_f,
    err.msgs = NULL, # no error handling so far
    extract = NULL, # only works it original resample has modesl = TRUE
    runtime = runtime, 
    # extra returns
    super.learner = super.learner, 
    use.feat = use.feat,
    parset = parset) 
  class(X) = c("RecombinedResampleResult", "ResampleResult")
  return(X)
}

