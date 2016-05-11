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
  method = .learner$method
  #
  # average/hill.climb
  #
  if (method %in% c("average", "hill.climb2")) {
    pred.list = getStackedBaseLearnerPredictions(model = .model, newdata = .newdata, type = "pred")
    if (method == "average") {
      final.pred = aggregatePredictions(pred.list, spt = sm.pt)
    } else {
      freq = .model$learner.model$freq
      pred.list1 = expandPredList(pred.list, freq = freq)
      final.pred = aggregatePredictions(pred.list1, spt = sm.pt)
    }
    if (sm.pt == "prob") {
      return(as.matrix(getPredictionProbabilities(final.pred, cl = td$class.levels)))
    } else {
      return(final.pred$data$response) #getPredictionResponse?
    }
  #
  # stack.nocv stack.cv
  #
  } else { 
    pred.data = getStackedBaseLearnerPredictions(model = .model, newdata = .newdata, type = "pred.data")
    #so?:
    pred.data = as.data.frame(pred.data)

    if (use.feat) {
      # feed pred.data into super model and we are done
      feat = .newdata[, colnames(.newdata) %nin% td$target, drop = FALSE]
      #TODO pred.data naming
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