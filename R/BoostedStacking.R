#' @title Create a BoostedStacking object.
#' 
#' @description
#'   Short descrition of boosted Stacking here!
#' 
#' @param multiplexer [\code{\link{ModelMultiplexer}}]\cr
#'   The muliplexer learner.
#' @param predict.type [\code{character(1)}]\cr 
#'   Classification: \dQuote{response} (= labels) or \dQuote{prob} (= probabilities and labels by selecting the ones with maximal probability).
#'   Regression: \dQuote{response} (= mean response) or \dQuote{se} (= standard errors and mean response).
#' @param resampling [\code{\link{ResampleDesc}} \cr
#'   Resampling strategy.
#'   par.set [\code{\link[ParamHelpers]{ParamSet}}]\cr
#'  @param mm.ps Collection of parameters and their constraints for optimization.
#'   Dependent parameters with a \code{requires} field must use \code{quote} and not
#'   \code{expression} to define it.
#' @param control [\code{\link{TuneControlRandom}}]\cr
#'   Control object for search method. 
#' @param measure [\code{\link{Measure}}]\cr
#'   Performance measure.
#' @param niter [\code{integer}]\cr
#'   Number of boosting iterations.
#' @example 
#'  lrns = list(
#'    #makeLearner("classif.ksvm", kernel = "rbfdot"), # no implm for response and multiclass
#'    makeLearner("classif.gbm"),
#'    makeLearner("classif.randomForest"))
#'  mm = makeModelMultiplexer(lrns)
#'  ctrl = makeTuneControlRandom(maxit = 3L)
#'  ps = makeModelMultiplexerParamSet(mm,
#'    makeIntegerParam("n.trees", lower = 1L, upper = 500L),
#'    makeIntegerParam("interaction.depth", lower = 1L, upper = 10L),
#'    makeIntegerParam("ntree", lower = 1L, upper = 500L),
#'    makeIntegerParam("mtry", lower = 1L, upper = getTaskNFeats(tsk)))
#'  lrns = lapply(lrns, setPredictType, bpt)
#'  stb = makeBoostedStackingLearner(model.multiplexer = mm, 
#'    predict.type = spt, resampling = cv5, mm.ps = ps, control = ctrl, 
#'    measures = mmce, niter = 2L)
#'  r = resample(stb, task = tsk, resampling = cv2)
#'
#' @export  

# FIXME: do we need id?

makeBoostedStackingLearner = function(model.multiplexer = mm, predict.type = "prob", resampling = cv2, mm.ps = ps, control = ctrl, measures = mmce, niter = 2L) {
	# input checks
	# INSERT IT HERE
  assertClass(model.multiplexer, "ModelMultiplexer")
  assertChoice(predict.type, choices = c("response", "prob"))
  assertClass(resampling, "ResampleDesc")
  assertClass(mm.ps, "ParamSet")
  assertClass(control, "TuneControlRandom") # for now
  assertInt(niter, lower = 1L)
  #FIXME check if mm and ps fit together
  # 
  if (model.multiplexer$type == "classif" & model.multiplexer$predict.type == "response" & "factors" %nin% model.multiplexer$properties) {
    stop("base models in model multiplexer does not support classifcation with factor features, which are created by using predict.type='response' within base learners")
  }
  #check: measures and type
  #
	par.set = makeParamSet(makeIntegerParam("niter", lower = 1, tunable = FALSE))
  
  bsl = makeLearnerBaseConstructor(classes = "BoostedStackingLearner", 
  	id = "boostedStacking", 
  	type = model.multiplexer$type,
  	package = model.multiplexer$package,
  	properties = model.multiplexer$properties,
  	par.set = par.set, 
  	par.vals = list(niter = niter), 
  	predict.type = predict.type)

  bsl$fix.factors.prediction = TRUE
  bsl$mm = model.multiplexer
  bsl$mm.ps = mm.ps
  bsl$resampling = resampling
  bsl$measures = measures
  bsl$control = ctrl
  return(bsl)
}

#' @export
trainLearner.BoostedStackingLearner = function(.learner, .task, .subset, ...) {
  # checks
  if (.task$type == "regr") {
    bpt = unique(extractSubList(.learner$mm$base.learners, "predict.type"))
    spt = .learner$predict.type
    if (any(c(bpt, spt) == "prob")) 
      stopf("Base learner predict type are '%s' and final predict type is '%s', but both should be 'response' for regression.", bpt, spt)
  }
  # body
  bms.pt = unique(extractSubList(.learner$mm$base.learner, "predict.type"))
  new.task = subsetTask(.task, subset = .subset)
  niter = .learner$par.vals$niter
  base.models = preds = vector("list", length = .learner$par.vals$niter)

  for (i in seq_len(niter)) {
    # 
    res = tuneParams(learner = .learner$mm, task = new.task, 
      resampling = .learner$resampling, measures = .learner$measures, 
      par.set = .learner$mm.ps, control = .learner$control)
    best.lrn = makeLearnerFromTuneResult(res)
    base.models[[i]] = train(best.lrn, new.task)
    preds[[i]] = predict(base.models[[i]], new.task)
    ##
    
    if (bms.pt == "prob") {
      new.feat = getPredictionProbabilities(preds[[i]])
      # FIXME if new.feat is constant, NA then use the second pred
      new.task = makeTaskWithNewFeat(task = new.task, 
        new.feat = new.feat,
        feat.name = paste0("feat.", i))
    } else {
      new.feat = getPredictionResponse(preds[[i]])
      # FIXME if new.feat is constant, NA then use the second pred
      new.task = makeTaskWithNewFeat(task = new.task, 
          new.feat = new.feat, feat.name = paste0("feat.", i))
      }
  }
  # FIXME pred.train returns acc to bms.pt...is that correct?
  list(base.models = base.models, final.task = new.task, pred.train = preds[[niter]])
}

#predictLearner.BoostedStackingModel = function(.learner, .model, .newdata, ...) {
#' @export
predictLearner.BoostedStackingLearner = function(.learner, .model, .newdata, ...) {
  new.data = .newdata
  sm.pt = .learner$predict.type
  bms.pt = unique(extractSubList(.learner$mm$base.learner, "predict.type"))
  
  td = getTaskDescription(.model)
  niter = length(.model$learner.model$base.models)
  for (i in seq_len(niter)) {
    newest.pred = predict(.model$learner.model$base.models[[i]], newdata = new.data)
    #FIXME for pred with response (or forbid it!?)
    # new.feat = getResponse(newest.pres) # is nicer
    if (bms.pt == "prob") {
      new.feat = getPredictionProbabilities(newest.pred)
      new.data = makeDataWithNewFeat(data = new.data, 
        new.feat = new.feat,
        feat.name = paste0("feat.", i), td)
    } else {
      new.feat = getPredictionResponse(newest.pred)
      new.data = makeDataWithNewFeat(data = new.data, 
        new.feat = new.feat, feat.name = paste0("feat.", i), td)
    }
  }
  if (sm.pt == "prob") {
    if (bms.pt == "prob") {
      return(as.matrix(getPredictionProbabilities(newest.pred, cl = td$class.levels)))
    } else { # if bms.pt="response" and sm.pt="prob" predict must be repeated
      last.model = .model$learner.model$base.models[[niter]]
      last.model$learner$predict.type = "prob"
      newest.pred = predict(last.model, newdata = new.data[, -ncol(new.data)]) #FIXMENOW
      return(as.matrix(getPredictionProbabilities(newest.pred, cl = td$class.levels)))
    }
  } else { #sm.pt = "response" and  bms.pt = "prob"/"response"
    return(getPredictionResponse(newest.pred)) #
  }
  #FIXME multiclass - should work now: check it!
}



setPredictType.BoostedStackingLearner = function(learner, predict.type) {
  
}
