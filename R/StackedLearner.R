#' @title Create a stacked learner object.
#'
#' @description A stacked learner uses predictions of several base learners and fits
#' a super learner using these predictions as features in order to predict the outcome.
#' The following stacking methods are available:
#'
#'  \describe{
#'   \item{\code{average}}{Averaging of base learner predictions without weights.}
#'   \item{\code{stack.nocv}}{Fits the super learner, where in-sample predictions of the base learners are used.}
#'   \item{\code{stack.cv}}{Fits the super learner, where the base learner predictions are computed
#'   by crossvalidated predictions (the resampling strategy can be set via the \code{resampling} argument).}
#'   \item{\code{hill.climb}}{Select a subset of base learner predictions by hill climbing algorithm.}
#'   \item{\code{compress}}{Train a neural network to compress the model from a collection of base learners.}
#'  }
#'
#' @param base.learners [(list of) \code{\link{Learner}}]\cr
#'   A list of learners created with \code{makeLearner}.
#' @param super.learner [\code{\link{Learner} | character(1)}]\cr
#'   The super learner that makes the final prediction based on the base learners.
#'   If you pass a string, the super learner will be created via \code{makeLearner}.
#'   Not used for \code{method = 'average'}. Default is \code{NULL}.
#' @param predict.type [\code{character(1)}]\cr
#'   Sets the type of the final prediction for \code{method = 'average'}.
#'   For other methods, the predict type should be set within \code{super.learner}.
#'   If the type of the base learner prediction, which is set up within \code{base.learners}, is
#'   \describe{
#'    \item{\code{"prob"}}{then \code{predict.type = 'prob'} will use the average of all
#'    bease learner predictions and \code{predict.type = 'response'} will use
#'    the class with highest probability as final prediction.}
#'    \item{\code{"response"}}{then, for classification tasks with \code{predict.type = 'prob'},
#'    the final prediction will be the relative frequency based on the predicted base learner classes
#'    and classification tasks with \code{predict.type = 'response'} will use majority vote of the base
#'    learner predictions to determine the final prediction.
#'    For regression tasks, the final prediction will be the average of the base learner predictions.}
#'   }
#'
#' @param method [\code{character(1)}]\cr
#'   \dQuote{average} for averaging the predictions of the base learners,
#'   \dQuote{stack.nocv} for building a super learner using the predictions of the base learners,
#'   \dQuote{stack.cv} for building a super learner using crossvalidated predictions of the base learners.
#'   \dQuote{hill.climb} for averaging the predictions of the base learners, with the weights learned from
#'   hill climbing algorithm and
#'   \dQuote{compress} for compressing the model to mimic the predictions of a collection of base learners
#'   while speeding up the predictions and reducing the size of the model.
#'   Default is \dQuote{stack.nocv},
#' @param use.feat [\code{logical(1)}]\cr
#'   Whether the original features should also be passed to the super learner.
#'   Not used for \code{method = 'average'}.
#'   Default is \code{FALSE}.
#' @param resampling [\code{\link{ResampleDesc}}]\cr
#'   Resampling strategy for \code{method = 'stack.cv'} and \code{method = 'hill.climb'}.
#'   Currently only CV is allowed for resampling.
#'   The default \code{NULL} uses 5-fold CV.
#' @param parset the parameters for \code{hill.climb} method, including
#' \describe{
#'   \item{\code{replace}}{Whether a base learner can be selected more than once.}
#'   \item{\code{init}}{Number of best models being included before the selection algorithm.}
#'   \item{\code{bagprob}}{The proportion of models being considered in one round of selection.}
#'   \item{\code{bagtime}}{The number of rounds of the bagging selection.}
#'   \item{\code{metric}}{The result evaluation metric function taking two parameters \code{pred} and \code{true},
#'   the smaller the score the better.}
#' }
#' the parameters for \code{compress} method, including
#' \describe{
#'    \item{k}{the size multiplier of the generated data}
#'    \item{prob}{the probability to exchange values}
#'    \item{s}{the standard deviation of each numerical feature}
#' }
#' @examples
#'   # Classification
#'   data(iris)
#'   tsk = makeClassifTask(data = iris, target = "Species")
#'   base = c("classif.rpart", "classif.lda", "classif.svm")
#'   lrns = lapply(base, makeLearner)
#'   lrns = lapply(lrns, setPredictType, "prob")
#'   m = makeStackedLearner(base.learners = lrns, predict.type = "prob", method = "hill.climb", parset = list(init = 0, metric = mmce))
#'   tmp = train(m, tsk)
#'   res = predict(tmp, tsk)
#'
#'   # Regression
#'   data(BostonHousing, package = "mlbench")
#'   tsk = makeRegrTask(data = BostonHousing, target = "medv")
#'   base = c("regr.rpart", "regr.svm")
#'   lrns = lapply(base, makeLearner)
#'   m = makeStackedLearner(base.learners = lrns, predict.type = "response", method = "compress", parset = list(init = 0, metric = mmce))
#'   tmp = train(m, tsk)
#'   res = predict(tmp, tsk)
#' @export
makeStackedLearner = function(base.learners, super.learner = NULL, predict.type = NULL,
  method = "stack.nocv", use.feat = FALSE, resampling = NULL, parset = list()) {

  if (is.character(base.learners)) base.learners = lapply(base.learners, checkLearner)
  if (is.null(super.learner) && method == "compress") {
    super.learner = makeLearner(paste0(base.learners[[1]]$type,'.nnet'))
  }
  if (!is.null(super.learner)) {
    super.learner = checkLearner(super.learner)
    if (!is.null(predict.type)) super.learner = setPredictType(super.learner, predict.type)
  }

  baseType = unique(extractSubList(base.learners, "type"))
  assertChoice(method, c("average", "stack.nocv", "stack.cv", "hill.climb", "compress"))

  if (method %in% c("stack.cv", "hill.climb", "compress")) {
    if (is.null(resampling)) {
      resampling = makeResampleDesc("CV", iters = 5L, stratify = ifelse(baseType == "classif", TRUE, FALSE))
    } else {
      assertClass(resampling, "CVDesc")
    }
  } else {
    assertClass(resampling, "NULL")
  }

  pts = unique(extractSubList(base.learners, "predict.type"))
  if ("se"%in%pts | (!is.null(predict.type) && predict.type == "se") |
        (!is.null(super.learner) && super.learner$predict.type == "se"))
    stop("Predicting standard errors currently not supported.")
  if (length(pts) > 1L)
    stop("Base learner must all have the same predict type!")
  if ((method == "average" | method == "hill.climb") & (!is.null(super.learner) | is.null(predict.type)) )
    stop("No super learner needed for this method or the 'predict.type' is not specified.")
  if (method != "average" & method != "hill.climb" & is.null(super.learner))
    stop("You have to specify a super learner for this method.")
  #if (method != "average" & !is.null(predict.type))
  #  stop("Predict type has to be specified within the super learner.")
  if ((method == "average" | method == "hill.climb") & use.feat)
    stop("The original features can not be used for this method")
  #if (!inherits(resampling, "CVDesc")) # new 
  #  stop("Currently only CV is allowed for resampling!") # new

  # lrn$predict.type is "response" by default change it using setPredictType
  lrn =  makeBaseEnsemble(
    id = "stack",
    base.learners = base.learners,
    cl = "StackedLearner"
  )

  # get predict.type from super learner or from predict.type
  if (!is.null(super.learner)) {
    lrn = setPredictType(lrn, predict.type = super.learner$predict.type)
  } else {
    lrn = setPredictType(lrn, predict.type = predict.type)
  }

  lrn$fix.factors.prediction = TRUE
  lrn$use.feat = use.feat

  lrn$method = method
  lrn$super.learner = super.learner
  lrn$resampling = resampling
  lrn$parset = parset
  return(lrn)
}

# FIXME: see FIXME in predict.StackedLearner I don't know how to make it better.
#'
#' @title Returns the predictions for each base learner.
#'
#' @description Returns the predictions for each base learner.
#'
#' @param model [\code{WrappedModel}]\cr Wrapped model, result of train.
#' @param newdata [\code{data.frame}]\cr
#' New observations, for which the predictions using the specified base learners should be returned.
#' Default is \code{NULL} and extracts the base learner predictions that were made during the training.
#'
#' @details None.
#'
#' @export
getStackedBaseLearnerPredictions = function(model, newdata = NULL, type = "pred.data") {
  assertChoice(type, choices = c("pred.data", "pred"))
  # get base learner and predict type
  #bms = model$learner.model$base.models
  #method = model$learner.model$method

  if (is.null(newdata)) {
    pred.data = model$learner.model$pred.train
  } else {
    # get base learner and predict type
    bms = model$learner.model$base.models
    method = model$learner.model$method
    # if (model == "stack.cv") warning("Crossvalidated predictions for new data is not possible for this method.")
    # predict prob vectors with each base model
    pred = pred.data = vector("list", length(bms))
    for (i in seq_along(bms)) {
      pred[[i]] = predict(bms[[i]], newdata = newdata)
      pred.data[[i]] = getResponse(pred[[i]], full.matrix = ifelse(method %in% c("average","hill.climb"), TRUE, FALSE))
    }

    names(pred.data) = sapply(bms, function(X) X$learner$id) #names(.learner$base.learners)
    
    # FIXME I don
    broke.idx.pd = which(unlist(lapply(pred.data, function(x) checkIfNullOrAnyNA(x))))
    if (length(broke.idx.pd) > 0) {
      messagef("Base Learner %s is broken in 'getStackedBaseLearnerPredictions' and will be removed\n", names(bls)[broke.idx])
      pred.data = pred.data[-broke.idx.pd, drop = FALSE]
      pred = pred[-broke.idx.pd, drop = FALSE]
    }
  }
  if (type == "pred") {
    return(pred)
  } else {
    return(pred.data)
  }
}

checkIfNullOrAnyNA = function(x) {
  if (is.null(x)) return(TRUE)
  if (anyNA(x)) return(TRUE)
  else FALSE
}

#' @export
trainLearner.StackedLearner = function(.learner, .task, .subset, ...) {
  # reduce to subset we want to train ensemble on
  .task = subsetTask(.task, subset = .subset)
  switch(.learner$method,
    average = averageBaseLearners(.learner, .task),
    stack.nocv = stackNoCV(.learner, .task),
    stack.cv = stackCV(.learner, .task),
    # hill.climb = hillclimbBaseLearners(.learner, .task, ...)
    hill.climb = do.call(hillclimbBaseLearners, c(list(.learner, .task), .learner$parset)),
    compress = compressBaseLearners(.learner, .task, .learner$parset)
  )
}

# FIXME: if newdata is the same data that was also used by training, then getBaseLearnerPrediction
# won't use the crossvalidated predictions (for method = "stack.cv").
#' @export
predictLearner.StackedLearner = function(.learner, .model, .newdata, ...) {
  # FIXME actually only .learner$method is needed
  use.feat = .model$learner$use.feat

  # get predict.type from learner and super model (if available)
  sm.pt = .model$learner$predict.type

  # get of predict type base learner
  bms.pt = unique(extractSubList(.model$learner$base.learners, "predict.type"))

  # get task information (classif)
  td = .model$task.desc
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))

  # predict prob vectors with each base model
  if (.learner$method != "compress") {
    pred.data = getStackedBaseLearnerPredictions(model = .model, newdata = .newdata)
  } else {
    pred.data = .newdata
  }
  #
  # average/hill.climb
  #
  if (.learner$method %in% c("average", "hill.climb")) {
    if (.learner$method == "hill.climb") {
      model.weight = .model$learner.model$weights
      # model.weights need to be adjusted when base learner fail in prediction
      number.pd = length(pred.data)
      number.mw = length(model.weights)
      # TODO
    } else {
      #FIXME Alternatively: all models can be kept. and here is the error handling done
      model.weight = rep(1/length(pred.data), length(pred.data))
    }
    # --- bms.pt == "prob" ---
    if (bms.pt == "prob") {
      # if base learner predictions are probabilities for classification
      for (i in 1:length(pred.data))
        pred.data[[i]] = pred.data[[i]]*model.weight[i]
      pred.data = Reduce("+", pred.data)
      if (sm.pt == "prob") {
        # if super learner predictions should be probabilities
        return(as.matrix(pred.data))
      } else {
        # if super learner predictions should be responses
        return(factor(colnames(pred.data)[max.col(pred.data)], td$class.levels))
      }
    # --- bms.pt == "response" ---
    } else { 
      pred.data = as.data.frame(pred.data)
      # if base learner predictions are responses
      if (type == "classif" || type == "multiclassif") {
        # if base learner predictions are responses for classification
        if (sm.pt == "prob") {
          # if super learner predictions should be probabilities, iter over rows to get proportions
          # FIXME: this is very slow + CUMBERSOME. we also do it in more places
          # we need a bbmisc fun for counting proportions in rows or cols
          #pred.data = apply(pred.data, 1L, function(x) (table(factor(x, td$class.levels) )/length(x)))
          #return(setColNames(t(pred.data), td$class.levels))
          pred.data = rowiseRatio(pred.data, td$class.levels, model.weight)
          return(pred.data)
        } else {
          # if super learner predictions should be responses
          return(factor(apply(pred.data, 1L, computeMode), td$class.levels))
        }
      }
      if (type == "regr") {
        # if base learner predictions are responses for regression
        prob = Reduce("+", pred.data) / length(pred.data) #rowMeans(pred.data)
        return(prob)
      }
    }
  # 
  # compress
  #
  } else if (.learner$method == "compress") {
    sm = .model$learner.model$super.model

    pred.data = as.data.frame(pred.data)
    pred = predict(sm, newdata = pred.data)
    if (sm.pt == "prob") {
      return(as.matrix(getPredictionProbabilities(pred, cl = td$class.levels)))
    } else {
      return(pred$data$response)
    }
  #
  # stack.nocv stack.cv
  #
  } else { 
    pred.data = as.data.frame(pred.data)

    if (use.feat) {
      # feed pred.data into super model and we are done
      feat = .newdata[, colnames(.newdata) %nin% td$target, drop = FALSE]
      predData = cbind(pred.data, feat)
    } else {
      predData = pred.data
    }
    sm = .model$learner.model$super.model
    messagef("Super model '%s' will use %s features and %s observations for prediction", sm$id, NCOL(predData), NROW(predData))
    messagef("There are %s NA in 'predData'", sum(is.na(predData)))
    pred = predict(sm, newdata = predData)
    if (sm.pt == "prob") {
      return(as.matrix(getPredictionProbabilities(pred, cl = td$class.levels)))
    } else {
      return(pred$data$response)
    }
  }
}

# Sets the predict.type for the super learner of a stacked learner
#' @export
setPredictType.StackedLearner = function(learner, predict.type) {
  lrn = setPredictType.Learner(learner, predict.type)
  lrn$predict.type = predict.type
  if ("super.learner"%in%names(lrn)) lrn$super.learner$predict.type = predict.type
  return(lrn)
}

### helpers to implement different ensemble types ###

# super simple averaging of base-learner predictions without weights. we should beat this
averageBaseLearners = function(learner, task) {
  bls = learner$base.learners
  #base.models = pred.data = vector("list", length(bls))
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stack", show.info = FALSE)
  exportMlrOptions(level = "mlr.stack")
  results = parallelMap(doTrainPredict, bls, more.args = list(task), impute.error = function(x) x)
  base.models = lapply(results, function(x) x[["base.models"]])
  pred.data = lapply(results, function(x) try(getResponse(x[["pred"]], full.matrix = TRUE)))
  #for (i in seq_along(bls)) {
  #  bl = bls[[i]]
  #  model = train(bl, task) # FIXME: err is failuremodel
  #  #message(paste(">>>", bl$id, paste(class(model))))
  #  base.models[[i]] = model
  #  pred = predict(model, task = task)
  #  pred.data[[i]] = getResponse(pred, full.matrix = TRUE)
  #}
  names(base.models) = names(bls)
  names(pred.data) = names(bls)

  #FIXME: I don't know if it is the nicest way to remove bls 
  broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pd = which(unlist(lapply(pred.data, function(x) any(is.na(x)))))
  broke.idx = unique(broke.idx.bm, broke.idx.pd)

  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    base.models = base.models[-broke.idx, drop = FALSE]
    pred.data = pred.data[-broke.idx, drop = FALSE]
  }

  list(method = "average", base.models = base.models, super.model = NULL,
    pred.train = pred.data)
}

# stacking where we predict the training set in-sample, then super-learn on that
stackNoCV = function(learner, task) {
  td = getTaskDescription(task)
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))
  bls = learner$base.learners
  use.feat = learner$use.feat
  #base.models = pred.data = vector("list", length(bls))
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stack", show.info = FALSE)
  exportMlrOptions(level = "mlr.stack")
  results = parallelMap(doTrainPredict, bls, more.args = list(task), impute.error = function(x) x)
  base.models = lapply(results, function(x) x[["base.models"]])
  pred.data = lapply(results, function(x) try(getResponse(x[["pred"]], full.matrix = FALSE)))
  
  #for (i in seq_along(bls)) {
  #  bl = bls[[i]]
  #  model = train(bl, task)
  #  base.models[[i]] = model
  #  pred = predict(model, task = task)
  #  pred.data[[i]] = getResponse(pred, full.matrix = FALSE)
  #}
  names(base.models) = names(bls)
  names(pred.data) = names(bls)

  # Remove FailureModels which would occur problems later
  broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pd = which(unlist(lapply(pred.data, function(x) any(is.na(x)))))
  broke.idx = unique(broke.idx.bm, broke.idx.pd)

  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    base.models = base.models[-broke.idx]
    pred.data = pred.data[-broke.idx]
  }
  
  pred.train = pred.data

  if (type == "regr" | type == "classif") {
    pred.data = as.data.frame(pred.data)
  } else {
    pred.data = as.data.frame(lapply(pred.data, function(X) X)) #X[,-ncol(X)]))
  }

  # now fit the super learner for predicted_pred.data --> target
  pred.data[[td$target]] = getTaskTargets(task)
  if (use.feat) {
    # add data with normal features
    feat = getTaskData(task)
    feat = feat[, colnames(feat) %nin% td$target, drop = FALSE]
    pred.data = cbind(pred.data, feat)
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data,
      target = td$target)
  } else {
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data, target = td$target)
  }
  messagef("Super learner '%s' will be trained with %s features and %s observations", learner$super.learner$id, getTaskNFeats(super.task), getTaskSize(super.task))
  super.model = train(learner$super.learner, super.task)
  list(method = "stack.no.cv", base.models = base.models,
       super.model = super.model, pred.train = pred.train)
}

# stacking where we crossval the training set with the base learners, then super-learn on that
stackCV = function(learner, task) {

  td = getTaskDescription(task)
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))
  bls = learner$base.learners
  use.feat = learner$use.feat
  # cross-validate all base learners and get a prob vector for the whole dataset for each learner
  #base.models = pred.data = vector("list", length(bls))
  rin = makeResampleInstance(learner$resampling, task = task)
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stack", show.info = FALSE)
  exportMlrOptions(level = "mlr.stack")
  results = parallelMap(doResampleTrain, bls, more.args = list(task, rin), impute.error = function(x) x)

    #for (i in seq_along(bls)) {
  #  bl = bls[[i]]
  #  r = resample(bl, task, rin, show.info = FALSE) #, extract = function(x) class(x))
  #  pred.data[[i]] = getResponse(r$pred, full.matrix = FALSE)
  #  # also fit all base models again on the complete original data set
  #  base.models[[i]] = train(bl, task)
  #}
  base.models = lapply(results, function(x) x[["base.models"]])
  pred.data = lapply(results, function(x) try(getResponse(x[["resres"]]$pred, full.matrix = FALSE)))
  
  names(pred.data) = names(bls)
  names(base.models) = names(bls)

  # Remove FailureModels which would occur problems later
  broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pd = which(unlist(lapply(pred.data, function(x) any(is.na(x)))))
  broke.idx = unique(broke.idx.bm, broke.idx.pd)

  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    base.models = base.models[-broke.idx]
    pred.data = pred.data[-broke.idx]
  }
  
  if (type == "regr" | type == "classif") {
    pred.data = as.data.frame(pred.data)
  } else {
    pred.data = as.data.frame(lapply(pred.data, function(X) X)) #X[,-ncol(X)]))
  }

  # add true target column IN CORRECT ORDER
  tn = getTaskTargetNames(task)
  test.inds = unlist(rin$test.inds)

  pred.train = as.list(pred.data[order(test.inds), , drop = FALSE])

  pred.data[[tn]] = getTaskTargets(task)[test.inds]

  # now fit the super learner for predicted_pred.data --> target
  pred.data = pred.data[order(test.inds), , drop = FALSE]
  #na_count <-function (x) sapply(x, function(y) sum(is.na(y)))
  #message(na_count(pred.data))

  if (use.feat) {
    # add data with normal features IN CORRECT ORDER
    feat = getTaskData(task)#[test.inds, ]
    feat = feat[, !colnames(feat)%in%tn, drop = FALSE]
    pred.data = cbind(pred.data, feat)
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data, target = tn)
  } else {
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data, target = tn)
  }
  #message(getTaskDescription(task))
  #message(na_count(getTaskData(super.task)))
  messagef("Super learner '%s' will be trained with %s features and %s observations", learner$super.learner$id, getTaskNFeats(super.task), getTaskSize(super.task))
  super.model = train(learner$super.learner, super.task)

  list(method = "stack.cv", base.models = base.models,
       super.model = super.model, pred.train = pred.train)
}

hillclimbBaseLearners = function(learner, task, replace = TRUE, init = 0, bagprob = 1, bagtime = 1,
  metric = NULL, ...) {

  assertFlag(replace)
  assertInt(init, lower = 0, upper = length(learner$base.learners)) #807
  assertNumber(bagprob, lower = 0, upper = 1)
  assertInt(bagtime, lower = 1)
  if (init > 0 & class(metric) == "Measure")
    stop("'metric' only implemented for init = 0. Set 'metric = NULL' or 'init = 0'")


  td = getTaskDescription(task)
  type = ifelse(td$type == "regr", "regr",
                ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))
  if (is.null(metric)) {
    if (type == "regr") {
      metric = function(pred, true) mean((pred-true)^2)
    } else {
      metric = function(pred, true) {
        pred = colnames(pred)[max.col(pred)]
        tb = table(pred, true)
        return( 1- sum(diag(tb))/sum(tb) )
      }
    }
  #} else {
  #  assertClass(metric, "Measure")
  #  metric = metric$fun # new
  }

  bls = learner$base.learners
  if (type != "regr") {
    if (any(extractSubList(bls, "predict.type") == "response"))
      stop("Hill climbing algorithm only takes probability predict type for classification.")
  }
  # cross-validate all base learners and get a prob vector for the whole dataset for each learner
  #base.models = resres = 
  pred.data = vector("list", length(bls)) #new
  rin = makeResampleInstance(learner$resampling, task = task)
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stack", show.info = FALSE)
  exportMlrOptions(level = "mlr.stack")
  results = parallelMap(doResampleTrain, bls, more.args = list(task, rin), impute.error = function(x) x)
  
  resres = lapply(results, function(x) x[["resres"]])
  base.models = lapply(results, function(x) x[["base.models"]])
  #pred.data = lapply(results, function(x) try(getResponse(x[["resres"]]$pred, full.matrix = T)))
  # as before:
  for (i in seq_along(bls)) {
    r = resres[[i]]
    if (type == "regr") {
      pred.data[[i]] = matrix(getResponse(r$pred, full.matrix = TRUE), ncol = 1)
    } else {
      pred.data[[i]] = getResponse(r$pred, full.matrix = TRUE)
      colnames(pred.data[[i]]) = task$task.desc$class.levels
    }
  }

  #for (i in seq_along(bls)) {
  #  bl = bls[[i]]
  #  resres[[i]] = r = resample(bl, task, rin, show.info = FALSE) #new
  #  if (type == "regr") {
  #    pred.data[[i]] = matrix(getResponse(r$pred, full.matrix = TRUE), ncol = 1)
  #  } else {
  #    pred.data[[i]] = getResponse(r$pred, full.matrix = TRUE)
  #    colnames(pred.data[[i]]) = task$task.desc$class.levels
  #  }
  #  # also fit all base models again on the complete original data set
  #  base.models[[i]] = train(bl, task) #FIXME
  #
  #}
  names(resres) = names(bls)
  names(base.models) = names(bls) #new

  # Remove FailureModels which would occur problems later
  broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pd = which(unlist(lapply(pred.data, function(x) anyNA(x))))
  broke.idx = unique(broke.idx.bm, broke.idx.pd)

  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    resres = resres[-broke.idx, drop = FALSE]
    pred.data = pred.data[-broke.idx, drop = FALSE]
    base.models = base.models[-broke.idx, drop = FALSE]
  }
  # add true target column IN CORRECT ORDER
  tn = getTaskTargetNames(task)
  test.inds = unlist(rin$test.inds)

  # now start the hill climbing
  pred.data = lapply(pred.data, function(x) x[order(test.inds), , drop = FALSE])
  pred.data[[tn]] = getTaskTargets(task)[test.inds]
  pred.data[[tn]] = pred.data[[tn]][order(test.inds)]
  # pred.data = pred.data[order(test.inds), , drop = FALSE]
  m = length(resres)
  weights = rep(0, m)
  flag = TRUE
  for (bagind in 1:bagtime) {
    # bagging of models
    bagsize = ceiling(m*bagprob)
    bagmodel = sample(1:m, bagsize)
    bagweight = rep(0, m) #807

    # Initial selection of strongest learners
    inds = NULL
    if (init > 0) {
      score = rep(Inf, m)
      for (i in bagmodel) {
        if (class(metric) != "Measure") {
          score[i] = metric(pred.data[[i]], pred.data[[tn]])
        } else {
          assertClass(metric, "Measure")
          score[i] = metric$fun(task, model = base.models[[i]], pred = resres[[i]]$pred) #new
        }
      }
      inds = order(score)[1:init]
      bagweight[inds] = 1 #807
    }

    selection.size = init
    selection.ind = inds
    # current.prob = rep(0, nrow(pred.data))
    current.prob = matrix(0, nrow(pred.data[[1]]), ncol(pred.data[[1]]))
    old.score = Inf
    if (selection.size > 0) {
      current.prob = Reduce('+', pred.data[selection.ind])
      old.score = metric(current.prob/selection.size, pred.data[[tn]]) #todo-metric
    }
    flag = TRUE

    while (flag) {
      score = rep(Inf, m)
      for (i in bagmodel) {
        if (class(metric) == "function") { #new
          score[i] = metric((pred.data[[i]]+current.prob)/(selection.size+1), pred.data[[tn]]) #new
        } else { # new
          assertClass(metric, "Measure")
          score[i] = metric$fun(task, model = base.models[[i]], pred = resres[[i]]$pred) #new FIXME!
        }
      }
      inds = order(score)
      if (!replace) {
        ind = setdiff(inds, selection.ind)[1]
      } else {
        ind = inds[1]
      }

      new.score = score[ind]
      if (old.score - new.score < 1e-8) {
        flag = FALSE
      } else {
        current.prob = current.prob + pred.data[[ind]]
        weights[ind] = weights[ind] + 1
        selection.ind = c(selection.ind, ind)
        selection.size = selection.size + 1
        old.score = new.score
      }
    }
    weights = weights + bagweight #807
  }
  weights = weights/sum(weights)
  names(weights) = names(resres)
  
  list(method = "hill.climb", base.models = base.models, super.model = NULL,
       pred.train = pred.data, weights = weights)
}

### other helpers ###

# Returns response for correct usage in stackNoCV and stackCV and for predictions
# also used in average and hill.climb
# full.matrix only used for predict.type = "prob": only returns positive prob if FALSE, all pred.data otherwise
getResponse = function(pred, full.matrix = NULL) {
  # if classification with probabilities
  if (pred$predict.type == "prob") {
    if (full.matrix) {
      # return matrix of probabilities
      td = pred$task.desc
      predReturn = pred$data[, paste("prob", td$class.levels, sep = ".")]
      colnames(predReturn) = td$class.levels
      return(predReturn)
    } else {
      # return only vector of probabilities for binary classification
      return(getPredictionProbabilities(pred))
    }
  } else {
    # if regression task
    pred$data$response
  }
}

# Create a super learner task
makeSuperLearnerTask = function(type, data, target) {
  #na_count <-function (x) sapply(x, function(y) sum(is.na(y)))
  #print(na_count(data))
  #data = data[, colnames(unique(as.matrix(data), MARGIN = 2))] # may not be useful for small data sets with predict.type=response
  # FIX it for now:
  keep.idx = colSums(is.na(data)) == 0
  data = data[, keep.idx, drop = FALSE]
  messagef("Feature '%s' will be removed\n", names(data)[!keep.idx])
  #message((na_count(data)))
  if (type == "classif") {
    removeConstantFeatures(task = makeClassifTask(id = "level 1 data", 
      data = data, target = target, fixup.data = "no"))
  } else {
    removeConstantFeatures(task = makeRegrTask(id = "level 1 data", 
      data = data, target = target, fixup.data = "no"))

  }
}

# Count the ratio
rowiseRatio = function(pred.data, levels, model.weight = NULL) {
  m = length(levels)
  p = ncol(pred.data)
  if (is.null(model.weight)) {
    model.weight = rep(1/p, p)
  }
  mat = matrix(0,nrow(pred.data),m)
  for (i in 1:m) {
    ids = matrix(pred.data==levels[i], nrow(pred.data), p)
    for (j in 1:p)
      ids[,j] = ids[,j]*model.weight[j]
    mat[,i] = rowSums(ids)
  }
  colnames(mat) = levels
  return(mat)
}


# TODOs:
# - document + test + export
# - benchmark stuff on openml
# - allow base.learners to be character of learners (not only list of learners)
# - rename 'pred.data' in code into 'preds'
# - allow option to remove predictions for one class in multiclass tasks (to avoid collinearity)
# - DONE: return predictions from each single base learner
# - DONE: allow predict.type = "response" for classif using majority vote (for super learner predict type "response")
#   and using average for super learner predict type "prob".
# - DONE: add option to use normal features in super learner
# - DONE: super learner can also return predicted probabilites
# - DONE: allow regression as well

# getWeights

doTrainPredict = function(bls, task) {
    model = train(bls, task)
    pred = predict(model, task = task)
    list(base.models = model, pred = pred)
}

doResampleTrain = function(bls, task, rin) {
  r = resample(bls, task, rin, show.info = FALSE)
  model = train(bls, task)
  list(resres = r, base.models = model)
}