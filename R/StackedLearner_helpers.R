### other helpers ###

# Sets the predict.type for the super learner of a stacked learner
#' @export
setPredictType.StackedLearner = function(learner, predict.type) {
  lrn = setPredictType.Learner(learner, predict.type)
  lrn$predict.type = predict.type
  if ("super.learner"%in%names(lrn)) lrn$super.learner$predict.type = predict.type
  return(lrn)
}

# Returns response for correct usage in stackNoCV and stackCV and for predictions
# also used in average and hill.climb
# full.matrix only used for predict.type = "prob": only returns positive prob if FALSE, all pred.data otherwise
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
  keep.idx = colSums(is.na(data)) == 0
  data = data[, keep.idx, drop = FALSE]
  if (getMlrOption("show.info"))
    messagef("Feature '%s' will be removed\n", names(data)[!keep.idx])
  #message((na_count(data)))
  if (type == "classif") {
    removeConstantFeatures(task = makeClassifTask(id = "level 1 data", 
      data = data, target = target, fixup.data = "no"))
  } else {
    removeConstantFeatures(task = makeRegrTask(id = "level 1 data", 
      data = data, target = target, fixup.data = "no"))

  }
}

# Count the ratio (used if base.learner predict.type = "response" and 
# super.learner predict.type is "prob")
rowiseRatio = function(pred.data, levels, model.weight = NULL) {
  m = length(levels)
  p = ncol(pred.data)
  if (is.null(model.weight)) {
    model.weight = rep(1/p, p)
  }
  mat = matrix(0,nrow(pred.data),m)
  for (i in 1:m) {
    ids = matrix(pred.data==levels[i], nrow(pred.data), p)
    for (j in 1:p)
      ids[,j] = ids[,j]*model.weight[j]
    mat[,i] = rowSums(ids)
  }
  colnames(mat) = levels
  return(mat)
}


#' Training and prediction in one function (used for parallelMap)
#'
doTrainPredict = function(bls, task, show.info) {
  #print(options()$error)
  setSlaveOptions()
  print("***************************")
  if (show.info)
    messagef("[Base Learner] %s is used", bls$id)
  model = train(bls, task)
  print(model); print("------------")
  pred = predict(model, task = task)
  print(pred); print("------------")
  print(object.size(list(base.models = model, pred = pred)))
  list(base.models = model, pred = pred)
}

#' Resampling and prediction in one function (used for parallelMap)
#' 
doTrainResample = function(bls, task, rin, show.info) {
  setSlaveOptions()
  if (show.info)
    messagef("[Base Learner] %s is used", bls$id)
  model = train(bls, task)
  r = resample(bls, task, rin, show.info = FALSE)
  print(object.size(list(resres = r, base.models = model)))
  list(resres = r, base.models = model)
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