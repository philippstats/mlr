hillclimbBaseLearners = function(learner, task, replace = TRUE, init = 1, bagprob = 1, bagtime = 1,
  metric = mmce, maxiter = "NULL", tolerance = 1e-8, ...) {
  # checks, inits
  assertFlag(replace)
  assertInt(init, lower = 0, upper = length(learner$base.learners)) #807
  assertNumber(bagprob, lower = 0, upper = 1)
  assertInt(bagtime, lower = 1)
  assertClass(metric, "Measure")

  td = getTaskDescription(task)
  type = ifelse(td$type == "regr", "regr",
                ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))

  bls = learner$base.learners
  id = learner$id
  save.on.disc = learner$save.on.disc
  if (maxiter == "NULL") maxiter = length(bls)
  assertInt(maxiter, lower = 1)
  assertNumber(tolerance)

  if (type != "regr") {
    if (any(extractSubList(bls, "predict.type") == "response"))
      stop("Hill climbing algorithm only takes probability predict type for classification.")
  }
  # body
  rin = makeResampleInstance(learner$resampling, task = task)
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stacking", show.info = FALSE)
  exportMlrOptions(level = "mlr.stacking")
  show.info = getMlrOption("show.info")
      #FIXME: only do resampling
  results = parallelMap(doTrainResample, bls, more.args = list(task, rin, show.info, id, save.on.disc), 
      impute.error = function(x) x, level = "mlr.stacking")
  
  base.models = lapply(results, function(x) x[["base.models"]])
  resres = lapply(results, function(x) x[["resres"]])
  pred.list = lapply(resres, function(x) x[["pred"]])
  bls.performance = sapply(resres, function(x) x$aggr)
  
  names(base.models) = names(bls)
  names(resres) = names(bls) 
  names(pred.list) = names(bls)
  names(bls.performance) = names(bls) # this will not be removed below!
  
  # Remove FailureModels which would occur problems later
  #broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pl = which(unlist(lapply(pred.list, function(x) anyNA(x$data))))# FIXME?!
  broke.idx.rr = which(unlist(lapply(resres, function(x) is.na(x$aggr[1]))))
  #broke.idx = unique(c(broke.idx.bm, broke.idx.rr, broke.idx.pl))
  broke.idx = unique(c(broke.idx.rr, broke.idx.pl))

  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    resres = resres[-broke.idx]
    #pred.data = pred.data[-broke.idx]
    base.models = base.models[-broke.idx]
    pred.list = pred.list[-broke.idx]
  }

  m = length(base.models)
  freq = rep(0, m)
  names(freq) = names(base.models)
  flag = TRUE
  freq.list = vector("list", bagtime)
  
  for (bagind in seq_len(bagtime)) {
    # bagging of models
    bagsize = ceiling(m * bagprob)
    bagmodel = sample(1:m, bagsize)
    #bagfreq = rep(0, m) #807

    # Initial selection of strongest learners
    inds.init = NULL
    inds.selected = NULL
    sel.algo = NULL
    #if (init > 0) {
    single.scores = rep(ifelse(metric$minimize, Inf, -Inf), m)
    for (i in bagmodel) {
      single.scores[i] = bls.performance[i] #resres[[i]]$aggr
    }
    if (metric$minimize) {
      inds.init = order(single.scores)[1:init]
    } else {
      inds.init = rev(order(single.scores))[1:init] 
    }
    #bagfreq[inds.init] = 1 #807
    freq[inds.init] = freq[inds.init] + 1  
    
    current.pred.list = pred.list[inds.init]
    current.pred = aggregatePredictions(current.pred.list)
    bench.score = metric$fun(pred = current.pred)

    #FIXME: maybe i will need them
    inds.selected = inds.init

    #FIXME: nicht 2. gl BLs hintereinander (ist gewährleistet wenn tolerance > 0)
    #flag = TRUE
    #iter.count = 1
    for (i in seq_along(maxiter)) {
    #while (flag) {
      temp.score = rep(ifelse(metric$minimize, Inf, -Inf), m)
      for (i in bagmodel) {
        temp.pred.list = append(current.pred.list, pred.list[i])
        aggr.pred = aggregatePredictions(temp.pred.list)
        temp.score[i] = metric$fun(pred = aggr.pred)
      }
      # order
      if (metric$minimize) {
        inds.ordered = order(temp.score)
      } else {
        inds.ordered  = rev(order(temp.score)) 
      }
      if (!replace) {
        best.ind = setdiff(inds.ordered, inds.selected)[1]
      } else {
        best.ind = inds.ordered[1]
      }
      # take the 1 best pred
      new.score = temp.score[best.ind]
      if (bench.score - new.score < tolerance) {
        break() # break inner loop #flag = FALSE
      } else {
        current.pred.list = append(current.pred.list, pred.list[best.ind])
        current.pred = aggregatePredictions(current.pred.list)
        freq[best.ind] = freq[best.ind] + 1
        inds.selected = c(inds.selected, best.ind)
        bench.score = new.score
        #iter.count = iter.count + 1
      }
      #if (iter.count >= maxiter) break()
    } # end while (now for)
    #freq = freq + bagfreq #807
    sel.algo = names(base.models)[inds.selected]
    freq.list[[bagind]] = sel.algo
  }
  
  weights = freq/sum(freq) #TODO: drop in future?

  # TODO current.pred gleich freq*bls gleich 
  # pred.list is list of Predictions, no data.frame, but I would say that is not needed
  list(method = "hill.climb", base.models = base.models, super.model = NULL,
    pred.train = pred.list, bls.performance = bls.performance, 
    weights = weights, freq = freq, freq.list = freq.list)
}