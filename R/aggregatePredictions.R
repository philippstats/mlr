#' Aggregate Predictions
#' 
#' Aggregate predicitons results by averaging (for \code{regr}, and  \code{classif} with prob) or mode ( \code{classif} with response). 
#' (works for regr, classif, multiclass)
#' 
#' @param pred.list [list of \code{Predictions}]\cr
#' @export

aggregatePredictions = function(pred.list, spt = NULL) {
  # return pred if list only contains one pred
  if (length(pred.list) == 1) {
    messagef("'pred.list' only contains one prediction and returns that one unlisted. Argument 'spt' will not be applied.")
    return(pred.list[[1]])
  }
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
  ti = NA_real_
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

# FIXME: clean up naming

#' Expand Predictions according to frequency argument
#' 
#' @param pred.list [\code{list} of \code{Predictions}]\cr
#'  List of Predictions which should be expanded. 
#' @param frequency [\code{named vector}]\cr
#'  Named vector containing the frequency of the chosen predictions. 
#'  Vector names must be set to the model names.
#' @export

expandPredList = function(pred.list, freq) {
  assertClass(pred.list, "list")
  assertClass(freq, "numeric")
  only.preds = unique(unlist(lapply(pred.list, function(x) any(class(x) == "Prediction"))))
  if (!only.preds) stopf("List elements in 'pred.list' are not all of class 'Prediction'")
  
  # remove 0s
  keep = names(which(freq > 0))
  freq1 = freq[keep]
  pred.list1 = pred.list[keep]
  # create grid for loop
  grid = data.frame(model = names(freq1), freq1, row.names = NULL)
  #expand_ = data.frame(model = rep(grid$model, grid$freq1)) %>% as.matrix %>% as.vector()
  expand = as.character(rep(grid$model, grid$freq1)) 
  pred.list2 = vector("list", length(expand))
  names(pred.list2) = paste(expand, 1:length(expand), sep = "_")
  
  for (i in seq_along(expand)) {
    #pred.list[i] %>% print 
    use = expand[i]
    #messagef("This is nr %s, %s", i, use)
    pred.list2[i] = pred.list1[use] 
    #message("---------------------------------------------------")
  }
 pred.list2
}
