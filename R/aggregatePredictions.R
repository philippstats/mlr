#' Aggregate Predictions
#' 
#' Aggregate predicitons results by averaging (for \code{regr}, and  \code{classif} with prob) or mode ( \code{classif} with response). 
#' (works for regr, classif, multiclass)
#' 
#' @param pred.list [list of \code{Predictions}]\cr
#' @export

aggregatePredictions = function(pred.list, spt = NULL) {
  # Check if "equal"
  x = lapply(pred.list, function(x) getTaskDescription(x))
  task.unequal = unlist(lapply(2:length(x), function(i) !all.equal(x[[1]], x[[i]])))
  if (any(task.unequal)) stopf("Task descriptions in prediction '1' and '%s' differ. This is not possible!", which(task.unequal)[1])

  x = lapply(pred.list, function(x) x$predict.type)
  pts.unequal = unlist(lapply(2:length(x), function(i) !all.equal(x[[1]], x[[i]])))
  if (any(pts.unequal)) stopf("Predict type in prediction '1' and '%s' differ. This is not possible!",  which(pts.unequal)[1])
  
  x = unlist(lapply(pred.list, function(x) checkIfNullOrAnyNA(x$data$response)))
  if (any(x)) messagef("Prediction '%s' is broken and will be removed.", which(x))
  pred.list = pred.list[!x]
  
  # Body
  pred1 = pred.list[[1]]
  type = getTaskType(pred1)
  td = getTaskDescription(pred1)
  rn = row.names(pred1$data)
  id = pred1$data$id
  tr = pred1$data$truth
  pt = pred1$predict.type
  if (is.null(spt)) spt = pt
  
  assertChoice(spt, choices = c("prob", "response"))
  ti = NA
  pred.length = length(pred.list) 
  
  # Reduce results
  # type = "classif"
  if (type == "classif") {
    # pt = "prob"
    if (pt == "prob") {
      # same method for spt response and prob
      preds = lapply(pred.list, getPredictionProbabilities, cl = td$class.levels)
      y = Reduce("+", preds) / pred.length
      if (spt == "response") {
        y = factor(max.col(y), labels = td$class.levels)
      }
    # pt = "response"
    } else {
      if (spt == "response") {
        preds = as.data.frame(lapply(pred.list, getPredictionResponse))
        y = factor(apply(preds, 1L, computeMode), td$class.levels)
      } else {
        # rowiseRatio copied from Tong He (he said it's not the best solution). 
        # This method should be rarely used, because pt = "response", 
        # spt = "prob" should perfrom worse than setting pt = "prob" (due to 
        # information loss when convertring probs to factors)
        preds = as.data.frame(lapply(pred.list, function(x) x$data$response))
        y = rowiseRatio(preds, td$class.levels, model.weight = NULL)
      }
    }
  # type = "regr"
  } else {
    preds = lapply(pred.list, getPredictionResponse)
    y = Reduce("+", preds)/pred.length
  }
  return(makePrediction(task.desc = td, rn, id = id, truth = tr, predict.type = spt, predict.threshold = NULL, y, time = ti))
}
