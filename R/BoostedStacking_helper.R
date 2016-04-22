#' Creates learner from TuneResult (from ModelMultiplexer).
#' 
makeLearnerFromTuneResult = function(tune.res = res) {
  # FIXME make me nicer
  assertClass(tune.res, "TuneResult")
  selected.learner = tune.res$x$selected.learner
  lrn.length = nchar(selected.learner)
  par.list = tune.res$x[-1]
  par.names = substr(names(par.list), lrn.length + 2, nchar(names(par.list)))
  names(par.list) = par.names
  setHyperPars(makeLearner(selected.learner, 
    predict.type = tune.res$learner$predict.type), par.vals = par.list)
}

#' Adds new feature(s) to task
makeTaskWithNewFeat = function(task, new.feat, feat.name) {
  # FIXME make me nicer
  assertClass(task, "Task")
  td = getTaskDescription(task)
  raw.data = getTaskData(task)
  if (class(new.feat) == "data.frame") {
    new.feat = new.feat[, -1, drop = FALSE]
    if (ncol(new.feat) > 1) 
      feat.name = paste(feat.name, td$class.levels[-1], sep = "_")
    data = cbind(raw.data, new.feat)
    colnames(data)[(NCOL(raw.data)+1):NCOL(data)] = feat.name
  } else {
    data = cbind(raw.data, data.frame(new.feat))
    colnames(data)[(NCOL(raw.data)+1)] = feat.name
  }
  if (td$type == "classif") {
    makeClassifTask(data = data, target = td$target, positive = td$positive)
  } else {
    makeRegrTask(data = data, target =  td$target)
  }
}

# Adds new feature(s) to data
makeDataWithNewFeat = function(data, new.feat, feat.name, task.desc) {
  assertClass(data, "data.frame")
  raw.data = data
  if (class(new.feat) == "data.frame") {
    new.feat = new.feat[, -1, drop = FALSE]
    if (ncol(new.feat) > 1) 
      feat.name = paste(feat.name, task.desc$class.levels[-1], sep = "_")
    data = cbind(raw.data, new.feat)
    colnames(data)[(NCOL(raw.data)+1):NCOL(data)] = feat.name
  } else {
    data = cbind(raw.data, data.frame(new.feat))
    colnames(data)[(NCOL(raw.data)+1)] = feat.name
  }
  return(data)
}


makeWrappedModel.BoostedStackingLearner = function(learner, learner.model, task.desc, subset, features, factor.levels, time) {
  x = NextMethod(x)
  addClasses(x, "BoostedStackingModel")
}