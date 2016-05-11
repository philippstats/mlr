### other helpers ###

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
  #na_count <-function (x) sapply(x, function(y) sum(is.na(y)))
  #print(na_count(data))
  #data = data[, colnames(unique(as.matrix(data), MARGIN = 2))] # may not be useful for small data sets with predict.type=response
  # FIX it for now:
  keep.idx = colSums(is.na(data)) == 0
  data = data[, keep.idx, drop = FALSE]
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

# Count the ratio
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


# Training and prediction in one function (used for parallelMap)
doTrainPredict = function(bls, task) {
    model = train(bls, task)
    pred = predict(model, task = task)
    list(base.models = model, pred = pred)
}

# Resampling and prediction in one function (used for parallelMap)
doResampleTrain = function(bls, task, rin) {
  r = resample(bls, task, rin, show.info = FALSE)
  model = train(bls, task)
  list(resres = r, base.models = model)
}



checkIfNullOrAnyNA = function(x) {
  if (is.null(x)) return(TRUE)
  if (any(is.na(x))) return(TRUE)
  else FALSE
}