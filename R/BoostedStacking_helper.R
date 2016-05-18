#' Creates x best learners from ModelMultiplexer TuneResult.
#' 
#' @param tune.result [\code{\link{TuneResult}}]\cr
#'   \code{TuneResult} from \code{ModelMultiplexer}
#' @param base.learners [\code{list of \link{TuneResult}}]\cr
#'   Needed to exract fix parameters.
#' @param x.best [\code{integer(1)}]
#'   Number of best learners to extract.
#' @param measure [\code{\link{Measure}}]\cr
#'   Measure to apply to "extract from best".
#' @return [\code{list of x best learners}]
#' @export


makeXBestLearnersFromMMTuneResult = function(tune.result, model.multiplexer, mm.ps = mm.ps, x.best = 5, measure = mmce) {
  # checks
  assertClass(tune.result, "TuneResult")
  assertClass(model.multiplexer, "ModelMultiplexer")
  assertClass(measure, "Measure")
  # body
  measure.name = paste0(measure$id, ".test.mean")
  opt.grid = as.data.frame(trafoOptPath(tune.result$opt.path), stringsAsFactors = FALSE)
  #FIXME minimize
  if (measure$minimize) {
    ord = order(opt.grid[, measure.name])[1:x.best]
  } else {
    ord = rev(order(opt.grid[, measure.name]))[1:x.best]
  }
  opt.grid = opt.grid[ord, ]
  j = sapply(opt.grid, is.factor)
  opt.grid[j] = lapply(opt.grid[j], as.character)
  # checks2
  if(NROW(opt.grid) < x.best) stopf("'x.best' is %s and cannot be set larger than the number of tuning results in '%s", x.best, quote(tune.result))
  #
  lrns = vector("list", x.best)
  for (i in 1:nrow(opt.grid)) {
    # get tuned parameter
  	cl = as.character(opt.grid[i, 1])
  	pars = opt.grid[i, grepl(pattern = cl, names(opt.grid)), drop = FALSE]
  	pars.names.long = names(pars)
  	pars.names = substr(names(pars), nchar(cl) + 2, nchar(names(pars)))
  	names(pars) = pars.names
  	pars.list = as.list(pars)
  	# get and apply trafo
  	#trafo = lapply(pars.names.long, function(x) mm.ps$pars[[x]]$trafo)
  	#trafo = lapply(trafo, function(x) if(is.null(x)) {function(x) x} else x)
  	#names(trafo) = pars.names
  	#pars.list = Map(do.call, trafo, lapply(par.list[names(trafo)], list)) #mapply(do.call, trafo, lapply(par.list[names(trafo)], list))
  	# get fix parameters, and final parameter set
  	pars.list.fix = model.multiplexer$base.learners[[cl]]$par.vals
  	idx = setdiff(names(pars.list.fix), names(pars.list))
  	pars.fin = c(pars.list, pars.list.fix[idx])
    # apply all parameters
  	lrns[[i]] = makeLearner(cl, id = paste(cl, i, sep = "_"),
  		predict.type = tune.result$learner$predict.type,
  		fix.factors.prediction = tune.result$learner$fix.factors.prediction,
  		par.vals = pars.fin)
  }
  lrns
}


#' Adds new feature(s) to task
makeTaskWithNewFeat = function(task, new.feat, feat.name) {
  # FIXME make me nicer
  assertClass(task, "Task")
  td = getTaskDescription(task)
  # check raw.data
  raw.data = getTaskData(task)
  
  if (class(new.feat) == "data.frame") {
    new.feat = new.feat[, -1, drop = FALSE]
    if (ncol(new.feat) > 1) 
      feat.name = paste(feat.name, td$class.levels[-1], sep = "_")
    data = cbind(raw.data, new.feat)
    data = data[, complete.cases(t(data))] 
    colnames(data)[(NCOL(raw.data)+1):NCOL(data)] = feat.name
  } else {
    data = cbind(raw.data, data.frame(new.feat))
    data = data[, complete.cases(t(data))] 
    colnames(data)[(NCOL(raw.data)+1)] = feat.name
  }
  if (td$type == "classif") {
    removeConstantFeatures(makeClassifTask(data = data, target = td$target, positive = td$positive))
  } else {
    removeConstantFeatures(makeRegrTask(data = data, target =  td$target))
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


#' Creates learner from TuneResult (from ModelMultiplexer) (veraltet)
#' 
makeLearnerFromMMTuneResult = function(tune.res = res, base.learners = base.learners) {
  # FIXME make me nicer
  assertClass(tune.res, "TuneResult")
  selected.learner = tune.res$x$selected.learner
  lrn.length = nchar(selected.learner)
  par.list = tune.res$x[-1]
  par.names = substr(names(par.list), lrn.length + 2, nchar(names(par.list)))
  names(par.list) = par.names
  par.list.fix = base.learners[[selected.learner]]$par.vals
  par.list = c(par.list, par.list.fix)
  # makeLearner
  setHyperPars(makeLearner(selected.learner, 
                           predict.type = tune.res$learner$predict.type, fix.factors.prediction = TRUE), 
               par.vals = par.list)
}
