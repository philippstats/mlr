#' @title Create a BoostedStacking object.
#' 
#' @description
#'   Short descrition of boosted Stacking here!
#' 
#' @param model.multiplexer [\code{\link{ModelMultiplexer}}]\cr
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
#' @param tolerance [\code{numeric}]\cr
#'   Tolerance for stopping criterion.
#' @examples 
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
#' 
#' 

#TODO: tol als parameter

makeBoostedStackingLearner = function(model.multiplexer = mm, predict.type = "prob", resampling = cv2, mm.ps = ps, control = ctrl, measures = mmce, niter = 2L, tolerance = 1e-8) {
	# do we need an id?
  # input checks
	# INSERT IT HERE
  assertClass(model.multiplexer, "ModelMultiplexer")
  assertChoice(predict.type, choices = c("response", "prob"))
  assertClass(resampling, "ResampleDesc")
  assertClass(mm.ps, "ParamSet")
  assertClass(control, "TuneControlRandom") # for now
  assertInt(niter, lower = 1L)
  assertNumber(tolerance)
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
  bsl$model.multiplexer = model.multiplexer
  bsl$mm.ps = mm.ps
  bsl$resampling = resampling
  bsl$measures = measures
  bsl$control = control
  bsl$tolerance = tolerance
  return(bsl)
}

#' @export
trainLearner.BoostedStackingLearner = function(.learner, .task, .subset, ...) {
  # checks
  if (.task$type == "regr") {
    bpt = unique(extractSubList(.learner$model.multiplexer$base.learners, "predict.type"))
    spt = .learner$predict.type
    if (any(c(bpt, spt) == "prob")) 
      stopf("Base learner predict type are '%s' and final predict type is '%s', but both should be 'response' for regression.", bpt, spt)
  }
  # body
  bms.pt = unique(extractSubList(.learner$model.multiplexer$base.learner, "predict.type"))
  new.task = subsetTask(.task, subset = .subset)
  niter = .learner$par.vals$niter
  base.models = preds = vector("list", length = niter)
  score = rep(ifelse(.learner$measures$minimize, Inf, -Inf), niter + 1)
  names(score) = c("init.score", paste("not.set", 1:niter, sep = "."))
  tolerance = .learner$tolerance
  
  for (i in seq_len(niter)) {
    messagef("iter %s time: %s", i, Sys.time())
    # Parameter Tuning
    res = tuneParams(learner = .learner$model.multiplexer, task = new.task, 
      resampling = .learner$resampling, measures = .learner$measures, 
      par.set = .learner$mm.ps, control = .learner$control)
    # IDEA: take best from every fold (i.e. anti-correlated/performs best on differnt input spaces/ bagging-like)
    # Stopping criterium
    score[i+1] = res$y[1]
    names(score)[i+1] = paste(res$x$selected.learner, i, sep = ".")
    shift = score[i] - score[i+1]
    #messagef(">shift is %s", shift)
    #messagef(">tol is %s", tolerance)
    tol.reached = ifelse(.learner$measures$minimize, shift < tolerance, shift > tolerance)
    #messagef(">force.stop is %s", tol.reached)
    if (tol.reached) {
      messagef("Boosting iterations stopped after %s niters", i)
      to.rm = i:niter
      #print(score)
      #print(to.rm)
      score = score[-c(to.rm + 1)]
      #print(score)
      base.models[to.rm] = NULL
      preds[to.rm] = NULL
      break()
    }
    # create learner, model, prediction
    best.lrn = makeXBestLearnersFromMMTuneResult(tune.result = res, 
      base.learners = .learner$model.multiplexer$base.learners, 
      x.best = 1, measure = .learner$measures) # FIXME x.best
    base.models[[i]] = train(best.lrn[[1]], new.task)
    preds[[i]] = resample(best.lrn[[1]], new.task, resampling = .learner$resampling, 
      measures = .learner$measures)
    # create new task
    if (bms.pt == "prob") {
      new.feat = getPredictionProbabilities(preds[[i]]$pred)
      # FIXME if new.feat is constant, NA then use the second pred
      new.task = makeTaskWithNewFeat(task = new.task, 
        new.feat = new.feat, feat.name = paste0("feat.", i))
    } else {
      new.feat = getPredictionResponse(preds[[i]]$pred)
      # FIXME if new.feat is constant, NA then use the second pred
      new.task = makeTaskWithNewFeat(task = new.task, 
        new.feat = new.feat, feat.name = paste0("feat.", i))
      }
  }
  # FIXME pred.train returns acc to bms.pt...is that correct?
  list(base.models = base.models, score = score[-1], final.task = new.task, pred.train = preds[[length(preds)]])
}

#predictLearner.BoostedStackingModel = function(.learner, .model, .newdata, ...) {
#' @export
predictLearner.BoostedStackingLearner = function(.learner, .model, .newdata, ...) {
  new.data = .newdata
  sm.pt = .learner$predict.type
  bms.pt = unique(extractSubList(.learner$model.multiplexer$base.learner, "predict.type"))
  
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
