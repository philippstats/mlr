### other helpers ###

# Sets the predict.type for the super learner of a stacked learner
#' @export
setPredictType.StackedLearner = function(learner, predict.type) {
  lrn = setPredictType.Learner(learner, predict.type)
  lrn$predict.type = predict.type
  if ("super.learner" %in% names(lrn)) lrn$super.learner$predict.type = predict.type
  return(lrn)
}

#' Returns response from Prediction object in stackNoCV, stackCV, average and hill.climb
#' @param pred Prediction
#' @param full.matrix Wether all n prediction values should be returned or in case of binary classification only one 
#' @export
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
    # for regression case
    pred$data$response
  }
}

# Create a super learner task
makeSuperLearnerTask = function(type, data, target) {
  keep.idx = colSums(is.na(data)) == 0
  data = data[, keep.idx, drop = FALSE]
  if (getMlrOption("show.info") & (length(keep.idx) < ncol(data)))
    warningf("Feature '%s' will be removed\n", names(data)[!keep.idx])
  #message((na_count(data)))
  if (type == "classif") {
    removeConstantFeatures(task = makeClassifTask(id = "level 1 data", 
      data = data, target = target, fixup.data = "no"))
  } else {
    removeConstantFeatures(task = makeRegrTask(id = "level 1 data", 
      data = data, target = target, fixup.data = "no"))

  }
}

#' Count the ratio (used if base.learner predict.type = "response" and 
#' super.learner predict.type is "prob")
#' @param pred.data Prediction data
#' @param levels Target levels of classifiaction task
#' @param model.weight Model weights, default is 1/number of data points
#' export
rowiseRatio = function(pred.data, levels, model.weight = NULL) {
  m = length(levels)
  p = ncol(pred.data)
  if (is.null(model.weight)) {
    model.weight = rep(1/p, p)
  }
  mat = matrix(0,nrow(pred.data), m)
  for (i in 1:m) {
    ids = matrix(pred.data == levels[i], nrow(pred.data), p)
    for (j in 1:p)
      ids[, j] = ids[, j] * model.weight[j]
    mat[, i] = rowSums(ids)
  }
  colnames(mat) = levels
  return(mat)
}


#' Training and prediction in one function (used for parallelMap)
#' @param bls [list of base.learner]
#' @param task [Task]
#' @param show.info show.info
#' @param id Id needed to create unique model name 
#' @param save.on.disc save.on.disc

doTrainPredict = function(bls, task, show.info, id, save.on.disc) {
  setSlaveOptions()
  model = train(bls, task)
  pred = predict(model, task = task)
  if (save.on.disc) {
    model.id = paste("saved.model", id, bls$id, "RData", sep = ".")
    saveRDS(model, file = model.id)
    if (show.info)
      messagef("[Base Learner] %s applied. Model saved as %s", bls$id, model.id)
    X = list(base.models = model.id, pred = pred)
  } else { # save.on.disc = FALSE:
    if (show.info) 
      messagef("[Base Learner] %s is applied. ", bls$id)
    X = list(base.models = model, pred = pred)
  }
  #print(paste(object.size(r)[1]/1000000, "MB"))
 X 
}

#' Resampling and prediction in one function (used for parallelMap)
#' @param bls [list of base.learner]
#' @param task [Task]
#' @param rin Resample Description
#' @param measures Measures for resampling
#' @param show.info show.info
#' @param id Id needed to create unique model name 
#' @param save.on.disc save.on.disc
doTrainResample = function(bls, task, rin, measures, show.info, id, save.on.disc) {
  setSlaveOptions()
  model = train(bls, task)
  r = resample(bls, task, rin, measures, show.info = FALSE)
  if (save.on.disc) {
    model.id = paste("saved.model", id, bls$id, "RData", sep = ".")
    saveRDS(model, file = model.id)
    if (show.info) 
      messagef("[Base Learner] %s applied. Model saved as %s", bls$id, model.id)
    X = list(base.models = model.id, resres = r)
  } else { # save.on.disc = FALSE:
    if (show.info) 
      messagef("[Base Learner] %s applied.", bls$id)
    X = list(base.models = model, resres = r)
  }
  #print(paste(object.size(r)[1]/1000000, "MB"))
  X 
}


# check if NULL or any NA
checkIfNullOrAnyNA = function(x) {
  if (is.null(x)) return(TRUE)
  if (any(is.na(x))) return(TRUE)
  else FALSE
}


# order a score vector and return the init numbers
orderScore = function(scores, minimize, init) {
  # checks
  assertClass(score, "numeric")
  assertChoice(minimize, c(TRUE, FALSE))
  assertInt(init, lower = 1, upper = length(score))
  # body
  if (is.null(init)) init = length(score)
  if (minimize) {
    order(scores)[1:init]
  } else {
    rev(order(scores))[1:init] 
  }
}

#' Convert models names (when model was saved on disc) to base learner name
#' @param base.model.id Unique ID used to save model on disc
#' @param stack.id ID from makeStackedLearner
convertModelNameToBlsName = function(base.model.id, stack.id) {
  id = substr(base.model.id, 1, nchar(base.model.id)-6) # remove .RData
  id = substr(id, 13 + nchar(stack.id) + 1, nchar(id))  
  id
}

removeModelsOnDisc = function(stack.id = NULL, bls.ids = NULL) {
  term = paste0("rm saved.model.", stack.id, "*", bls.ids, ".RData")
  system(term)
}
