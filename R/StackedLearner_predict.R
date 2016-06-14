#' @export
predictLearner.StackedLearner = function(.learner, .model, .newdata, ...) { # FIXME actually only .learner$method is needed
  # prep
  use.feat = .model$learner$use.feat
  sm.pt = .model$learner$predict.type
  bms.pt = unique(extractSubList(.model$learner$base.learners, "predict.type"))
  td = .model$task.desc
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))
  method = .learner$method
  # average + hill.climb
  if (method %in% c("average", "hill.climb")) {
    pred.list = getStackedBaseLearnerPredictions(model = .model, newdata = .newdata, type = "pred")
    if (method == "average") {
      final.pred = aggregatePredictions(pred.list, spt = sm.pt)
    } else {
      freq = .model$learner.model$freq
      pred.list = expandPredList(pred.list, freq = freq)
      final.pred = aggregatePredictions(pred.list, spt = sm.pt)
    }
  # stack.nocv + stack.cv
  } else { 
    pred.data = getStackedBaseLearnerPredictions(model = .model, newdata = .newdata, type = "pred.data")
    # remoce first level from multiclass prob predictions (Multicollinearity)
    if (type == "multiclassif" && bms.pt == "prob") { #FIXME: only for "stats" methods
      pred.data = lapply(pred.data, function(x) x[, -1])
    }
    pred.data = as.data.frame(pred.data)
    if (use.feat) {
      #feat = .newdata[, colnames(.newdata) %nin% td$target, drop = FALSE] # TODO: there shouldn't be a target present in any case: feat = .newdata
      #pred.data = cbind(pred.data, feat)
      pred.data = cbind(pred.data, .newdata)
    } 
    sm = .model$learner.model$super.model
    #browser()
    messagef("[Super Learner] Predict %s with %s features on %s observations", sm$learner$id, NCOL(pred.data), NROW(pred.data))
    #mgs = paste("[Super Learner] Predict", sm$task.desc$id, "with",  NCOL(pred.data), "features on", NROW(pred.data), "observations")
    #message(mgs)
    #messagef("There are %s NA in 'pred.data'", sum(is.na(pred.data)))
    final.pred = predict(sm, newdata = pred.data)
  }
  # return 
  if (sm.pt == "prob") {
    return(as.matrix(getPredictionProbabilities(final.pred, cl = td$class.levels)))
  } else {
    return(final.pred$data$response) #getPredictionResponse?
  }
}



