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
#' the parameters for \code{boost.stack} method, including
#' \describe{
#'    \item{mm.ps}{\code{ParamSet} for ModelMultiplexer}
#'    \item{control}{\code{TuneControl} for parameter tuning}
#'    \item{niter}{number of features to add}
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
  assertChoice(method, c("average", "stack.nocv", "stack.cv", "hill.climb", "compress", "boost.stack"))

  if (method %in% c("stack.cv", "hill.climb", "compress", "boost.stack")) {
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
  if (method %in% c("average", "hill.climb", "boost.stack") && (!is.null(super.learner) | is.null(predict.type)) )
    stop("No super learner needed for this method or the 'predict.type' is not specified.")
  if (method %nin% c("average", "hill.climb", "boost.stack") & is.null(super.learner))
    stop("You have to specify a super learner for this method.")
  #if (method != "average" & !is.null(predict.type))
  #  stop("Predict type has to be specified within the super learner.")
  if (method %in% c("average", "hill.climb") & use.feat)
    stop("The original features cannot be used for this method")
  #if (method == "boost.stack" & !is.null(use.feat))
  #  stop("Argument use.fest will be ignored for this method")
  #if (!inherits(resampling, "CVDesc")) # new 
  #  stop("Currently only CV is allowed for resampling!") # new

  if (method == "boost.stack") {
    lrn = makeModelMultiplexer(base.learners = base.learners) # class ModelMultiplexer & BaseEnsemble
    lrn$id = "stack"
    # FIXME: not nice
    class(lrn) = c("StackedLearner", class(lrn))
  } else {
    # lrn$predict.type is "response" by default change it using setPredictType
    lrn =  makeBaseEnsemble(
      id = "stack",
      base.learners = base.learners,
      cl = "StackedLearner")
  }

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
getStackedBaseLearnerPredictions = function(model, newdata = NULL) {
  # get base learner and predict type
  #bms = model$learner.model$base.models
  #method = model$learner.model$method

  if (is.null(newdata)) {
    probs = model$learner.model$pred.train
  } else {
    # get base learner and predict type
    bms = model$learner.model$base.models
    method = model$learner.model$method
    # if (model == "stack.cv") warning("Crossvalidated predictions for new data is not possible for this method.")
    # predict prob vectors with each base model
    probs = vector("list", length(bms))
    for (i in seq_along(bms)) {
      pred = predict(bms[[i]], newdata = newdata)
      probs[[i]] = getResponse(pred, full.matrix = ifelse(method %in% c("average","hill.climb"), TRUE, FALSE))
    }

    names(probs) = sapply(bms, function(X) X$learner$id) #names(.learner$base.learners)
  }
  return(probs)
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
    compress = compressBaseLearners(.learner, .task, .learner$parset),
    boost.stack = boostStack(.learner, .task)
  )
}

# FIXME: if newdata is the same data that was also used by training, then getBaseLearnerPrediction
# won't use the crossvalidated predictions (for method = "stack.cv").
#' @export
predictLearner.StackedLearner = function(.learner, .model, .newdata, ...) {
 # browser()
  use.feat = .model$learner$use.feat
  # get predict.type from learner and super model (if available)
  sm.pt = .model$learner$predict.type
  sm = .model$learner.model$super.model

  # get base learner and predict type
  bms.pt = unique(extractSubList(.model$learner$base.learners, "predict.type"))

  # get task information (classif)
  td = .model$task.desc
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))

  # predict prob vectors with each base model
  if (.learner$method %nin% c("compress", "boost.stack")) {
    probs = getStackedBaseLearnerPredictions(model = .model, newdata = .newdata)
  } else if (.learner$method == "compress"){ # FIXME: naming
    probs = .newdata
  } else { # boost.stack
  # FIXME multiclass, reg/or in data-version w/o Task:
    #new.task = makeClassifTask(data = .newdata) # needs target
    new.data = .newdata
  }
  if (.learner$method %in% c("average", "hill.climb")) {    # average/h.c
    if (.learner$method == "hill.climb") {                  # h.c
      model.weight = .model$learner.model$weights
    } else {                                                # average
      model.weight = rep(1/length(probs), length(probs))
    }
    if (bms.pt == "prob") {
      # if base learner predictions are probabilities for classification
      for (i in 1:length(probs))
        probs[[i]] = probs[[i]]*model.weight[i]
      prob = Reduce("+", probs)
      if (sm.pt == "prob") {
        # if super learner predictions should be probabilities
        return(as.matrix(prob))
      } else {
        # if super learner predictions should be responses
        return(factor(colnames(prob)[max.col(prob)], td$class.levels))
      }
    } else {
      probs = as.data.frame(probs)
      # if base learner predictions are responses
      if (type == "classif" || type == "multiclassif") {
        # if base learner predictions are responses for classification
        if (sm.pt == "prob") {
          # if super learner predictions should be probabilities, iter over rows to get proportions
          # FIXME: this is very slow + CUMBERSOME. we also do it in more places
          # we need a bbmisc fun for counting proportions in rows or cols
          #probs = apply(probs, 1L, function(x) (table(factor(x, td$class.levels) )/length(x)))
          #return(setColNames(t(probs), td$class.levels))
          probs = rowiseRatio(probs, td$class.levels, model.weight)
          return(probs)
        } else {
          # if super learner predictions should be responses
          return(factor(apply(probs, 1L, computeMode), td$class.levels))
        }
      }
      if (type == "regr") {
        # if base learner predictions are responses for regression
        prob = Reduce("+", probs) / length(probs) #rowMeans(probs)
        return(prob)
      }
    }
  } else if (.learner$method == "compress") { # compress
    probs = as.data.frame(probs)
    pred = predict(sm, newdata = probs)
    if (sm.pt == "prob") {
      return(as.matrix(getPredictionProbabilities(pred, cl = td$class.levels)))
    } else {
      return(pred$data$response)
    }
  } else if (.learner$method %in% c("stack.nocv", "stack.cv")) {
    probs = as.data.frame(probs)
    # feed probs into super model and we are done
    feat = .newdata[, colnames(.newdata) %nin% td$target, drop = FALSE]

    if (use.feat) {
      predData = cbind(probs, feat)
    } else {
      predData = probs
    }

    pred = predict(sm, newdata = predData)
    if (sm.pt == "prob") {
      return(as.matrix(getPredictionProbabilities(pred, cl = td$class.levels)))
    } else {
      return(pred$data$response)
    }
  } else { #### boost.stack ####
    niter = length(.model$learner.model$base.models)
    for (i in seq_len(niter)) {
      newest.pred = predict(.model$learner.model$base.models[[i]], newdata = new.data)
      #FIXME for pred with response (or forbid it!?)
      # new.feat = getResponse(newest.pres) # is nicer
      if (bms.pt == "prob") {
        new.feat = getPredictionProbabilities(newest.pred, cl = td$class.levels)
        new.data = makeDataWithNewFeat(data = new.data, 
          new.col = new.feat[, -NCOL(new.feat), drop = FALSE],
          feat.name = paste0("feat.", i))
      } else {
        new.feat = newest.pred$data$response
        new.data = makeDataWithNewFeat(data = new.data, 
          new.col = new.feat, feat.name = paste0("feat.", i))
      }
    }
#    if (sm.pt == "prob") {
      return(as.matrix(getPredictionProbabilities(newest.pred, cl = td$class.levels)))
#    } else {
#      return(newest.pred$data$response)
#    }
    #FIXME multiclass - should work now: check it!
  }
}

# Sets the predict.type for the super learner of a stacked learner
#' @export
setPredictType.StackedLearner = function(learner, predict.type) {
  lrn = setPredictType.Learner(learner, predict.type)
  lrn$predict.type = predict.type
  if ("super.learner" %in% names(lrn)) lrn$super.learner$predict.type = predict.type
  return(lrn)
}

### helpers to implement different ensemble types ###

# super simple averaging of base-learner predictions without weights. we should beat this
averageBaseLearners = function(learner, task) {
  bls = learner$base.learners
  base.models = probs = vector("list", length(bls))
  for (i in seq_along(bls)) {
    bl = bls[[i]]
    model = train(bl, task)
    message(bl$id)
    base.models[[i]] = model
    pred = predict(model, task = task)
    probs[[i]] = getResponse(pred, full.matrix = TRUE)
    message(paste0("loop>", round(mem_used()/1024/1024, 2), "-MB"))
  }
  message(paste0(round(mem_used()/1024/1024, 2), "-MB"))
  names(probs) = names(bls)
  list(method = "average", base.models = base.models, super.model = NULL,
       pred.train = probs)
}

# stacking where we predict the training set in-sample, then super-learn on that
stackNoCV = function(learner, task) {
  td = getTaskDescription(task)
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))
  bls = learner$base.learners
  use.feat = learner$use.feat
  base.models = probs = vector("list", length(bls))
  # FIXME: parallize it
  for (i in seq_along(bls)) {
    bl = bls[[i]]
    model = train(bl, task)
    base.models[[i]] = model
    pred = predict(model, task = task)
    probs[[i]] = getResponse(pred, full.matrix = FALSE)
  }
  names(probs) = names(bls)

  pred.train = probs

  if (type == "regr" | type == "classif") {
    probs = as.data.frame(probs)
  } else {
    probs = as.data.frame(lapply(probs, function(X) X)) #X[,-ncol(X)]))
  }

  # now fit the super learner for predicted_probs --> target
  probs[[td$target]] = getTaskTargets(task)
  if (use.feat) {
    # add data with normal features
    feat = getTaskData(task)
    feat = feat[, colnames(feat) %nin% td$target, drop = FALSE]
    probs = cbind(probs, feat)
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = probs,
      target = td$target)
  } else {
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = probs, target = td$target)
  }
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
  base.models = probs = vector("list", length(bls))
  rin = makeResampleInstance(learner$resampling, task = task)
  for (i in seq_along(bls)) {
    bl = bls[[i]]
    r = resample(bl, task, rin, show.info = FALSE)
    probs[[i]] = getResponse(r$pred, full.matrix = FALSE)
    # also fit all base models again on the complete original data set
    base.models[[i]] = train(bl, task)
  }
  names(probs) = names(bls)

  if (type == "regr" | type == "classif") {
    probs = as.data.frame(probs)
  } else {
    probs = as.data.frame(lapply(probs, function(X) X)) #X[,-ncol(X)]))
  }

  # add true target column IN CORRECT ORDER
  tn = getTaskTargetNames(task)
  test.inds = unlist(rin$test.inds)

  pred.train = as.list(probs[order(test.inds), , drop = FALSE])

  probs[[tn]] = getTaskTargets(task)[test.inds]

  # now fit the super learner for predicted_probs --> target
  probs = probs[order(test.inds), , drop = FALSE]
  if (use.feat) {
    # add data with normal features IN CORRECT ORDER
    feat = getTaskData(task)#[test.inds, ]
    feat = feat[, !colnames(feat)%in%tn, drop = FALSE]
    predData = cbind(probs, feat)
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = predData, target = tn)
  } else {
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = probs, target = tn)
  }
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
    if ("response" %in% unique(extractSubList(bls, "predict.type")))
        stop("Hill climbing algorithm only takes probability predict type for classification.")

  }
  # cross-validate all base learners and get a prob vector for the whole dataset for each learner
  base.models = resres = probs = vector("list", length(bls)) #new
  rin = makeResampleInstance(learner$resampling, task = task)
  for (i in seq_along(bls)) {
    bl = bls[[i]]
                message(paste0(i, ">", bl$id)
    resres[[i]] = r = resample(bl, task, rin, show.info = FALSE) #new
                message(paste0(">resample>", round(mem_used()/1024/1024, 2), "-MB"))

    if (type == "regr") {
      probs[[i]] = matrix(getResponse(r$pred, full.matrix = TRUE), ncol = 1)
    } else {
      probs[[i]] = getResponse(r$pred, full.matrix = TRUE)
      colnames(probs[[i]]) = task$task.desc$class.levels
    }
    # also fit all base models again on the complete original data set
    base.models[[i]] = train(bl, task)
                    message(paste0(">train>", round(mem_used()/1024/1024, 2), "-MB"))
        # new
    #print(bl$id)
    #print(gc())
    # new/
  }
  names(probs) = names(bls)
  names(resres) = names(bls) #new

  # add true target column IN CORRECT ORDER
  tn = getTaskTargetNames(task)
  test.inds = unlist(rin$test.inds)

  # now start the hill climbing
  probs = lapply(probs, function(x) x[order(test.inds), , drop = FALSE])
  probs[[tn]] = getTaskTargets(task)[test.inds]
  probs[[tn]] = probs[[tn]][order(test.inds)]
  # probs = probs[order(test.inds), , drop = FALSE]
  m = length(bls)
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
          score[i] = metric(probs[[i]], probs[[tn]])
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
    # current.prob = rep(0, nrow(probs))
    current.prob = matrix(0, nrow(probs[[1]]), ncol(probs[[1]]))
    old.score = Inf
    if (selection.size>0) {
      current.prob = Reduce('+', probs[selection.ind])
      old.score = metric(current.prob/selection.size, probs[[tn]]) #todo-metric
    }
    flag = TRUE

    while (flag) {
      score = rep(Inf, bagsize)
      for (i in bagmodel) {
        if (class(metric) == "function") { #new
          score[i] = metric( (probs[[i]]+current.prob)/(selection.size+1), probs[[tn]] ) #new
        } else { # new
          assertClass(metric, "Measure")
          score[i] = metric$fun(task, model = base.models[[i]], pred = resres[[i]]$pred) #new
        }
      }
      inds = order(score)
      if (!replace) {
        ind = setdiff(inds, selection.ind)[1]
      } else {
        ind = inds[1]
      }

      new.score = score[ind]
      if (old.score-new.score<1e-8) {
        flag = FALSE
      } else {
        current.prob = current.prob+probs[[ind]]
        weights[ind] = weights[ind]+1
        selection.ind = c(selection.ind, ind)
        selection.size = selection.size+1
        old.score = new.score
      }
    }
    weights = weights + bagweight #807
  }
  weights = weights/sum(weights)

  list(method = "hill.climb", base.models = base.models, super.model = NULL,
       pred.train = probs, weights = weights)
}

compressBaseLearners = function(learner, task, parset = list()) {
  lrn = learner
  lrn$method = "hill.climb"
  ensemble.model = train(lrn, task)

  data = getTaskData(task, target.extra = TRUE)
  data = data[[1]]

  pseudo.data = do.call(getPseudoData, c(list(data), parset))
  pseudo.target = predict(ensemble.model, newdata = pseudo.data)
  pseudo.data = data.frame(pseudo.data, target = pseudo.target$data$response)

  td = ensemble.model$task.desc
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))

  if (type == "regr") {
    new.task = makeRegrTask(data = pseudo.data, target = "target")
    if (is.null(learner$super.learner)) {
      m = makeLearner("regr.nnet", predict.type = )
    } else {
      m = learner$super.learner
    }
  } else {
    new.task = makeClassifTask(data = pseudo.data, target = "target")
    if (is.null(learner$super.learner)) {
      m = makeLearner("classif.nnet", predict.type = "")
    } else {
      m = learner$super.learner
    }
  }

  super.model = train(m, new.task)

  list(method = "compress", base.learners = lrn$base.learners, super.model = super.model,
       pred.train = pseudo.data)
}


boostStack = function(learner, task) {
  new.task = task
  td = getTaskDescription(task)
  #FIXME: (Later) Only save the last prediction
  best.lrn = base.models = predictions = vector("list", length = learner$parset$niter)
  # FIXME: arrange classes. tuneParams needs "ModelMultiplexer" 
  class(learner) = c("ModelMultiplexer", "StackedLearner", "BaseEnsemble", "Learner")

  bms.pt = unique(extractSubList(learner$base.learners, "predict.type"))
  for (i in seq_len(learner$parset$niter)) {
  #FIXME: Weiss nicht wieso tuneParams als train/predict class StackedLearner aufruft und nicht class ModelMultiplexer
    res = tuneParams(learner = learner, task = new.task, 
      resampling = learner$resampling, par.set = learner$parset$mm.ps, 
      control = learner$parset$control)
    best.lrn[[i]] = makeLearnerFromTuneResult(res)
    base.models[[i]] = train(best.lrn[[i]], new.task)
    predictions[[i]] = predict(base.models[[i]], new.task)
    ##
    if (bms.pt == "prob") {
        new.feat = getPredictionProbabilities(predictions[[i]], cl = td$class.levels)
        new.task = makeTaskWithNewFeat(task = new.task, 
          new.feat = new.feat[, -NCOL(new.feat), drop = FALSE],
          feat.name = paste0("feat.", i))
      } else {
        new.feat = predictions[[i]]$data[, "response", drop = FALSE]
        new.task = makeTaskWithNewFeat(task = new.task, 
          new.feat = new.feat, feat.name = paste0("feat.", i))
      }
    }
    ##
    #new.task = makeTaskWithNewFeat(new.task, pred = predictions[[i]], 
    #    predict.type = learner$predict.type, feat.name = paste0("feat.", i))
    # FIXME: update par.set (for randomForest.mtry)
    # FIXME: report performance or something
    #message(paste(niter, ":", performace(predictions[[i]], measure = measure)))
  class(learner) = c("StackedLearner", "BaseEnsemble", "Learner")
  list(method = "boost.stack", base.models = base.models, super.model = NULL,
       pred.train = predictions[[learner$parset$niter]])
  
}


### other helpers ###

# Returns response for correct usage in stackNoCV and stackCV and for predictions
# also used in average and hill.climb
# full.matrix only used for predict.type = "prob": only returns positive prob if FALSE, all probs otherwise
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
  data = data[, colnames(unique(as.matrix(data), MARGIN = 2))] # may not be useful for small data sets with predict.type=response
  if (type == "classif") {
    removeConstantFeatures(task = makeClassifTask(data = data, target = target))
  } else {
    removeConstantFeatures(task = makeRegrTask(data = data, target = target))

  }
}

# Count the ratio
rowiseRatio = function(probs, levels, model.weight = NULL) {
  m = length(levels)
  p = ncol(probs)
  if (is.null(model.weight)) {
    model.weight = rep(1/p, p)
  }
  mat = matrix(0,nrow(probs),m)
  for (i in 1:m) {
    ids = matrix(probs==levels[i], nrow(probs), p)
    for (j in 1:p)
      ids[,j] = ids[,j]*model.weight[j]
    mat[,i] = rowSums(ids)
  }
  colnames(mat) = levels
  return(mat)
}

getPseudoData = function(.data, k = 3, prob = 0.1, s = NULL, ...) {
  res = NULL
  n = nrow(.data)
  ori.names = names(.data)
  feat.class = sapply(.data, class)
  ind2 = which(feat.class == "factor")
  ind1 = setdiff(1:ncol(.data), ind2)
  if (length(ind2)>0)
    ori.labels = lapply(.data[[ind2]], levels)
  .data = lapply(.data, as.numeric)
  .data = as.data.frame(.data)
  # Normalization
  mn = rep(0, ncol(.data))
  mx = rep(0, ncol(.data))
  for (i in ind1) {
    mn[i] = min(.data[,i])
    mx[i] = max(.data[,i])
    .data[, i] = (.data[, i]-mn[i])/(mx[i]-mn[i])
  }
  if (is.null(s)) {
    s = rep(0, ncol(.data))
    for (i in ind1) {
      s[i] = sd(.data[,i])
    }
  }
  testNumeric(s, len = ncol(.data), lower = 0)

  # Func to calc dist
  hamming = function(mat) {
    n = nrow(mat)
    m = ncol(mat)
    res = matrix(0,n,n)
    for (j in 1:m) {
      unq = unique(mat[,j])
      p = length(unq)
      for (i in 1:p) {
        ind = which(mat[,j] == unq[i])
        res[ind, -ind] = res[ind, -ind]+1
      }
    }
    return(res)
  }

  one.nn = function(mat, ind1, ind2) {
    n = nrow(mat)
    dist.mat.1 = matrix(0,n,n)
    dist.mat.2 = matrix(0,n,n)
    if (length(ind1)>0) {
      dist.mat.1 = as.matrix(stats::dist(mat[,ind1, drop = FALSE]))
    }
    if (length(ind2)>0) {
      dist.mat.2 = hamming(mat[,ind2, drop = FALSE])
    }
    dist.mat = dist.mat.1+dist.mat.2
    neighbour = max.col( -dist.mat - diag(Inf, n))
    return(neighbour)
  }

  # Get the neighbour
  neighbour = one.nn(.data, ind1, ind2)

  # Start the loop
  p = ncol(.data)
  for (loop in 1:k) {
    data = .data
    prob.mat = matrix(sample(c(0,1), n*p, replace = TRUE, prob = c(prob, 1-prob)), n, p)
    prob.mat = prob.mat == 0
    for (i in 1:n) {
      e = as.numeric(data[i, ])
      ee = as.numeric(data[neighbour[i], ])

      # continuous
      for (j in ind1) {
        if (prob.mat[i,j]) {
          current.sd = abs(e[j]-ee[j])/s[j]
          tmp1 = rnorm(1,ee[j], current.sd)
          tmp2 = rnorm(1,e[j], current.sd)
          e[j] = tmp1
          ee[j] = tmp2
        }
      }
      for (j in ind2) {
        if (prob.mat[i,j]) {
          tmp = e[j]
          e[j] = ee[j]
          ee[j] = tmp
        }
      }

      data[i,] = ee
      data[neighbour[i],] = e
    }
    res = rbind(res, data)
  }
  for (i in ind1)
    res[,i] = res[,i]*(mx[i]-mn[i])+mn[i]
  res = data.frame(res)
  names(res) = ori.names
  for (i in ind2)
    res[[i]] = factor(res[[i]], labels = ori.labels[[i]])
  return(res)
}

makeLearnerFromTuneResult = function(result = res) {
  assertClass(result, "TuneResult")
  if (!is.null(result$x$selected.learner)) { # from ModelMultiplexer
    lrn.char = result$x$selected.learner
    lrn.length = nchar(lrn.char)
    par.list = result$x[-1]
    par.names = substr(names(par.list), lrn.length + 2, nchar(names(par.list)))
    names(par.list) = par.names
    setHyperPars(makeLearner(lrn.char, 
      predict.type = result$learner$predict.type), par.vals = par.list)
  } else { # from "normal" tuning
    setHyperPars(makeLearner(class(result$learner)[1], 
      predict.type = result$learner$predict.type), par.vals = result$x)  
  }
}

makeTaskWithNewFeat = function(task, new.feat, feat.name) {
  assertClass(task, "Task")
  assertClass(new.feat, "data.frame")
  td = getTaskDescription(task)
  raw.data = getTaskData(task)
  n.new.col = NCOL(new.feat)
  if (n.new.col > 1) 
    feat.name = paste(feat.name, seq_len(n.new.col), sep = "_")
  #data = getTaskData(task, target.extra = TRUE)
  data = cbind(raw.data, new.feat)
  colnames(data)[(NCOL(raw.data)+1):NCOL(data)] = feat.name

  if (task$task.desc$type == "classif") {
    makeClassifTask(data = data, target = td$target, positive = td$positive)
  } else {
    makeRegrTask(data = data, target =  td$target)
  }
}


makeDataWithNewFeat = function(data, new.col = NULL, feat.name = "feat") {
  # new.col: vector or data.frame
  n.new.col = NCOL(new.col)
  if (n.new.col > 1) 
    feat.name = paste(feat.name, seq_len(n.new.col), sep = "_")
  #data = getTaskData(task, target.extra = TRUE)
  data1 = cbind(data, new.col)
  colnames(data1)[(NCOL(data)+1):NCOL(data1)] = feat.name
  return(data = data1)
}

# TODOs:
# - document + test + export
# - benchmark stuff on openml
# - allow base.learners to be character of learners (not only list of learners)
# - rename 'probs' in code into 'preds'
# - allow option to remove predictions for one class in multiclass tasks (to avoid collinearity)
# - DONE: return predictions from each single base learner
# - DONE: allow predict.type = "response" for classif using majority vote (for super learner predict type "response")
#   and using average for super learner predict type "prob".
# - DONE: add option to use normal features in super learner
# - DONE: super learner can also return predicted probabilites
# - DONE: allow regression as well

