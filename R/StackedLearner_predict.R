#' @export
predictLearner.StackedLearner = function(.learner, .model, .newdata, ...) { # FIXME actually only .learner$method is needed
  # setup
  use.feat = .model$learner$use.feat
  sm.pt = .model$learner$predict.type
  bms.pt = unique(extractSubList(.model$learner$base.learners, "predict.type"))
  td = .model$task.desc
  method = .learner$method

  # obtain predictions
  pred.list = getStackedBaseLearnerPredictions(model = .model, newdata = .newdata)

  # apply average 
  if (method == "average") {
    final.pred = aggregatePredictions(pred.list, spt = sm.pt)
  # apply hill.climb
  } else if (method == "hill.climb") {
    freq = .model$learner.model$freq
    pred.list = expandPredList(pred.list, freq = freq)
    final.pred = aggregatePredictions(pred.list, spt = sm.pt)
  # apply stack.cv
  } else if (method == "stack.cv") {
    pred.data = lapply(pred.list, function(x) getPredictionDataNonMulticoll(x))
    pred.data = as.data.frame(pred.data)
    #names(pred.data) =  extractSubList(.model$learner$base.learners, "id") # FIXME WROGN (multiclass)
    if (use.feat) {
      feat = .newdata[, colnames(.newdata) %nin% td$target, drop = FALSE]
      pred.data = cbind(pred.data, feat)
    } 
    sm = .model$learner.model$super.model
    messagef("[Super Learner] Predict %s with %s features on %s observations", sm$learner$id, ncol(pred.data), nrow(pred.data))
    #print(head(pred.data))
    #messagef("There are %s NA in 'pred.data'", sum(is.na(pred.data)))
    final.pred = predict(sm, newdata = pred.data)
  } 
  # return 
  if (sm.pt == "prob") {
    return(as.matrix(getPredictionProbabilities(final.pred, cl = td$class.levels)))
  } else {
    return(final.pred$data$response) #FIXME getPredictionResponse?
  }
}



