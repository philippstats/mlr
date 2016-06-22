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
#'   \item{\code{hill.climb}}{Select a subset of base learner predictions by hill climbing algorithm. (new implementation)}
#'   \item{\code{compress}}{Train a neural network to compress the model from a collection of base learners.}
#'  }
#'
#' @param id [\code{character(1)}]  Id string for object. Used to display object and for model saving. Default is "stack".
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
#'   \dQuote{stack.cv} for building a super learner using crossvalidated predictions of the base learners.
#'   \dQuote{hill.climb} for averaging the predictions of the base learners, with the weights learned from
#'   ensemble selection algorithm and 
#' @param use.feat [\code{logical(1)}]\cr
#'   Whether the original features should also be passed to the super learner.
#'   Only used for \code{method = 'stack.cv'}.
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
#'   \item{\code{metric}}{The result evaluation metric. Must be an object of type \code{Measure} from mlr.}
#'   \item{\code{tolerance}}{The tolerance when inner loop should stop.}
#' }
#' @param save.on.disc [\code{logical(1)}]\cr 
#'   If set to \code{TRUE}, base models are saved on disc at the working directory. 
#'   This setting saves memory when huge models are fitted. Later during prediction 
#'   this models are loaded. Models are saved with the name "saved.models<stack.id>.<base.learner.id>.RData".
#'   Note that is only works for train-predict procdureas as well as for resampling using holdout. 
#'   Applying outer cross validation will result in wrong predictions due to the 
#'   fact that model names does not seperate between different resample iterations.
#'   Default is \code{FALSE}
#' @param save.preds [\code{logical(1)}]\cr 
#'   If set to \code{FALSE} models will not contain predictions. This reduce the 
#'   object size. Note that function \code{recombine} does not work if saving 
#'   prediction is disabled. Default is \code{TRUE}.
#' @examples
#' \dontrun{
#'   # Classification
#'   data(iris)
#'   tsk = makeClassifTask(data = iris, target = "Species")
#'   base = c("classif.rpart", "classif.lda", "classif.svm")
#'   lrns = lapply(base, makeLearner)
#'   lrns = lapply(lrns, setPredictType, "prob")
#'   m = makeStackedLearner(base.learners = lrns, predict.type = "prob", method = "hill.climb", parset = list(init = 1, metric = mmce))
#'   tmp = train(m, tsk)
#'   res = predict(tmp, tsk)
#'
#'   # Regression
#'   data(BostonHousing, package = "mlbench")
#'   tsk = makeRegrTask(data = BostonHousing, target = "medv")
#'   base = c("regr.rpart", "regr.svm")
#'   lrns = lapply(base, makeLearner)
#'   m = makeStackedLearner(base.learners = lrns, predict.type = "response", method = "compress", parset = list(init = 1, metric = mae))
#'   tmp = train(m, tsk)
#'   res = predict(tmp, tsk)
#' }
#' @export
makeStackedLearner = function(id = "stack", base.learners, super.learner = NULL, 
  predict.type = NULL, method = "stack.nocv", use.feat = FALSE, resampling = NULL, 
  parset = list(), save.on.disc = FALSE, save.preds = TRUE) {
  # checking
  if (is.character(base.learners)) base.learners = lapply(base.learners, checkLearner)
  #if (is.null(super.learner) && method == "compress") {
  #  super.learner = makeLearner(paste0(base.learners[[1]]$type,'.nnet'))
  #}
  if (!is.null(super.learner)) {
    super.learner = checkLearner(super.learner)
    if (!is.null(predict.type)) super.learner = setPredictType(super.learner, predict.type)
  }

  baseType = unique(extractSubList(base.learners, "type"))
  assertChoice(method, c("average", "stack.nocv", "stack.cv", "hill.climb","compress"))
  assertCharacter(id, min.chars = 1)
  assertLogical(save.on.disc, len = 1)
  assertLogical(save.preds, len = 1)
  
  if (method %in% c("stack.cv", "hill.climb", "compress")) {
    if (is.null(resampling)) {
      resampling = makeResampleDesc("CV", iters = 5L, stratify = ifelse(baseType == "classif", TRUE, FALSE))
    } else {
      assertClass(resampling, "CVDesc")
    }
  } else {
    assertClass(resampling, "NULL")
  }

  bpt = unique(extractSubList(base.learners, "predict.type"))
  if ("se" %in% bpt | (!is.null(predict.type) && predict.type == "se") |
        (!is.null(super.learner) && super.learner$predict.type == "se"))
    stop("Predicting standard errors currently not supported.")
  if (length(bpt) > 1L)
    stop("Base learner must all have the same predict type!")
  if ((method %in% c("average", "hill.climb")) & (!is.null(super.learner) | is.null(predict.type)) )
    stop("No super learner needed for this method or the 'predict.type' is not specified.")
  if (method %nin% c("average", "hill.climb") & is.null(super.learner))
    stop("You have to specify a super learner for this method.")
  #if (method != "average" & !is.null(predict.type))
  #  stop("Predict type has to be specified within the super learner.")
  if ((method %in% c("average", "hill.climb")) & use.feat)
    stop("The original features can not be used for this method")
  #if (!inherits(resampling, "CVDesc")) # new 
  #  stop("Currently only CV is allowed for resampling!") # new
  # lrn$predict.type is "response" by default change it using setPredictType
  lrn =  makeBaseEnsemble(
    id = id,
    base.learners = base.learners,
    cl = "StackedLearner"
  )
  if (!is.null(super.learner)) {
    lrn = setPredictType(lrn, predict.type = super.learner$predict.type)
  } else {
    lrn = setPredictType(lrn, predict.type = predict.type)
  }

  lrn$fix.factors.prediction = TRUE
  lrn$use.feat = use.feat

  lrn$method = method
  lrn$name = "Stacked Learner"
  lrn$short.name = "stack"
  lrn$super.learner = super.learner
  lrn$resampling = resampling
  lrn$parset = parset
  lrn$save.on.disc = save.on.disc
  lrn$save.preds = save.preds
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
#' @param type [\code{character(1)}]\cr 
#'  \dQuote{pred.data} to obtain predictions as a vector/matrix.
#'  \dQuote{pred} to obtain predictions as \code{Prediction} object.
#' @details None.
#'
#' @export
getStackedBaseLearnerPredictions = function(model, newdata = NULL, type = "pred.data") {
  assertChoice(type, choices = c("pred.data", "pred"))
  stack.id = model$learner$id
  # checking
  if (is.null(newdata)) {
    pred.data = model$learner.model$pred.train
  } else {
    # get base learner and predict type
    method = model$learner.model$method
    if (method == "hill.climb") {
      used.bls = names(which(model$learner.model$freq > 0))
      #bms = model$learner.model$base.models[used.bls]
      bms = model$learner.model$base.models[used.bls]
    } else {
      #bms = model$learner.model$base.models
      bms = model$learner.model$base.models
    }
    # if (model == "stack.cv") warning("Crossvalidated predictions for new data is not possible for this method.") # and not needes
    # predict prob vectors with each base model
    pred = pred.data = vector("list", length(bms))
    if (model$learner$save.on.disc) {
      for (i in seq_along(bms)) {
        m = readRDS(bms[[i]])
        pred[[i]] = predict(m, newdata = newdata)
        pred.data[[i]] = getResponse(pred[[i]], full.matrix = ifelse(method %in% c("average", "hill.climb"), TRUE, FALSE))
      }
      bls.ids = sapply(bms, function(x) convertModelNameToBlsName(x, stack.id))
    } else {
      for (i in seq_along(bms)) {
        pred[[i]] = predict(bms[[i]], newdata = newdata)
        pred.data[[i]] = getResponse(pred[[i]], full.matrix = ifelse(method %in% c("average", "hill.climb"), TRUE, FALSE))
      }
      bls.ids = sapply(bms, function(X) X$learner$id) #names(.learner$base.learners)
    }
    #bls.ids = names(model$learner.model$base.models)
    #bls.ids = sapply(bms, function(X) X$learner$id) #names(.learner$base.learners)
    
    names(pred) = bls.ids  #names(.learner$base.learners)
    names(pred.data) = bls.ids
    
    # FIXME I don
    broke.idx.pd = which(unlist(lapply(pred.data, function(x) checkIfNullOrAnyNA(x))))
    if (length(broke.idx.pd) > 0) {
      messagef("Preds '%s' is broken in 'getStackedBaseLearnerPredictions' and will be removed\n", names(bls)[broke.idx])
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


# phr
# TODOS
# - presently the first predicions feature is removed for multiclass and prob 
#   cases. this is done due multicoll reasons (especiallly glm methods whic hare 
#   famous for stacking). allow that this first prediction is not removed. 
#   learnsers like rF can hadle such cases. 
