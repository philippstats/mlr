hillclimbBaseLearners = function(learner, task, replace = TRUE, init = 1, bagprob = 1, bagtime = 1,
  metric = mmce, ...) {
  assertFlag(replace)
  assertInt(init, lower = 0, upper = length(learner$base.learners)) #807
  assertNumber(bagprob, lower = 0, upper = 1)
  assertInt(bagtime, lower = 1)
  assertClass(metric, "Measure")

  td = getTaskDescription(task)
  type = ifelse(td$type == "regr", "regr",
                ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))

  bls = learner$base.learners
  if (type != "regr") {
    if (any(extractSubList(bls, "predict.type") == "response"))
      stop("Hill climbing algorithm only takes probability predict type for classification.")
  }
  # cross-validate all base learners and get a prob vector for the whole dataset for each learner
  rin = makeResampleInstance(learner$resampling, task = task)
  
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stack", show.info = FALSE)
  exportMlrOptions(level = "mlr.stack")
  results = parallelMap(doResampleTrain, bls, more.args = list(task, rin), impute.error = function(x) x)
  
  resres = lapply(results, function(x) x[["resres"]])
  base.models = lapply(results, function(x) x[["base.models"]])
  pred.list = lapply(resres, function(x) x[["pred"]])

  # as before:
  #for (i in seq_along(bls)) {
  #  r = resres[[i]]
  #  if (type == "regr") {
  #    pred.data[[i]] = matrix(getResponse(r$pred, full.matrix = TRUE), ncol = 1)
  #  } else {
  #    pred.data[[i]] = getResponse(r$pred, full.matrix = TRUE)
  #    colnames(pred.data[[i]]) = task$task.desc$class.levels
  #  }
  #}
  
  names(base.models) = names(bls)
  names(resres) = names(bls) 
  names(pred.list) = names(bls)
  # Remove FailureModels which would occur problems later
  broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pl = which(unlist(lapply(pred.list, function(x) anyNA(x$data))))# FIXME?!
  broke.idx.rr = which(unlist(lapply(resres, function(x) anyNA(x$aggr))))
  broke.idx = unique(c(broke.idx.bm, broke.idx.rr, broke.idx.pl))

  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    resres = resres[-broke.idx, drop = FALSE]
    #pred.data = pred.data[-broke.idx]
    base.models = base.models[-broke.idx, drop = FALSE]
    pred.list = pred.list[-broke.idx, drop = FALSE]
  }

  m = length(base.models)
  freq = weights = rep(0, m)
  names(freq) = names(base.models); names(weights) = names(base.models)
  flag = TRUE
  freq.list = vector("list", bagtime)
  
  for (bagind in seq_len(bagtime)) {
    # bagging of models
    bagsize = ceiling(m * bagprob)
    bagmodel = sample(1:m, bagsize)
    bagfreq = rep(0, m) #807

    # Initial selection of strongest learners
    inds = NULL
    #if (init > 0) {
    single.scores = rep(ifelse(metric$minimize, Inf, -Inf), m)
    for (i in bagmodel) {
      single.scores[i] = resres[[i]]$aggr
    }
    if (metric$minimize) {
      inds = order(single.scores)[1:init]
    } else {
      inds = rev(order(single.scores))[1:init] 
    }
    bagfreq[inds] = 1 #807
    
    #current.pred.list = makeEqualPrediction(resres[[1]]$pred) # 1/3
    #bench.score = ifelse(metric$minimize, Inf, -Inf)

    #if (init > 0) {
    #current.prob = Reduce('+', pred.data[selection.ind])
    #bench.score = metric(current.prob/selection.size, pred.data[[tn]]) #todo-metric
    #res = resres[inds]
    current.pred.list = pred.list[inds] #lapply(res, function(x) x$pred)
    current.pred = aggregatePredictions(current.pred.list)
    bench.score = metric$fun(pred = current.pred)

    #FIXME: maybe i will need them
    #selection.size = init
    selection.ind = inds
    
    flag = TRUE
    while (flag) {
      score = rep(ifelse(metric$minimize, Inf, -Inf), m)
      # calulate pred: init.preds plus 1 new
      for (i in bagmodel) {
        #list.names = c(names(current.pred.list), names(resres[i])) #TODO resres to pred.list if it's named
        temp.pred.list = append(current.pred.list, pred.list[i])
        #current.pred.list$resres[[1]] = resres[[i]]$pred #TODO
        #names(temp.pred.list) = list.names
        aggr.pred = aggregatePredictions(temp.pred.list)
        score[i] = metric$fun(pred = aggr.pred)
      }
      # order
      if (metric$minimize) {
        inds = order(score)
      } else {
        inds = rev(order(score)) 
      }
      if (!replace) {
        #TODO what if character(0):
        ind = setdiff(inds, selection.ind)[1]
      } else {
        ind = inds[1]
      }
      # take the 1 best pred
      new.score = score[ind]
      if (bench.score - new.score < 1e-8) {
        flag = FALSE
      } else {
        current.pred.list = append(current.pred.list, pred.list[ind])
        current.pred = aggregatePredictions(current.pred.list)
        freq[ind] = freq[ind] + 1
        bench.score = new.score
        selection.ind = c(selection.ind, ind)
      }
    } # while
    freq = freq + bagfreq #807
    # FIXME add a list for weights
    sel.algo = names(base.models)[selection.ind]
    freq.list[[bagind]] = sel.algo
  }
  #TODO: drop in future
  weights = freq/sum(freq)

  list(method = "hill.climb", base.models = base.models, super.model = NULL,
       pred.train = current.pred, weights = weights, freq = freq, freq.list = freq.list)
}