
# stacking where we predict the training set in-sample, then super-learn on that
stackNoCV = function(learner, task) {
  td = getTaskDescription(task)
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))
  bls = learner$base.learners
  use.feat = learner$use.feat
  #base.models = pred.data = vector("list", length(bls))
  # parallelMap
  parallelLibrary("mlr", master = FALSE, level = "mlr.stack", show.info = FALSE)
  exportMlrOptions(level = "mlr.stack")
  results = parallelMap(doTrainPredict, bls, more.args = list(task), impute.error = function(x) x)
  base.models = lapply(results, function(x) x[["base.models"]])
  pred.data = lapply(results, function(x) try(getResponse(x[["pred"]], full.matrix = FALSE)))
  
  #for (i in seq_along(bls)) {
  #  bl = bls[[i]]
  #  model = train(bl, task)
  #  base.models[[i]] = model
  #  pred = predict(model, task = task)
  #  pred.data[[i]] = getResponse(pred, full.matrix = FALSE)
  #}
  names(base.models) = names(bls)
  names(pred.data) = names(bls)

  # Remove FailureModels which would occur problems later
  broke.idx.bm = which(unlist(lapply(base.models, function(x) any(class(x) == "FailureModel"))))
  broke.idx.pd = which(unlist(lapply(pred.data, function(x) any(is.na(x)))))
  broke.idx = unique(broke.idx.bm, broke.idx.pd)

  if (length(broke.idx) > 0) {
    messagef("Base Learner %s is broken and will be removed\n", names(bls)[broke.idx])
    base.models = base.models[-broke.idx]
    pred.data = pred.data[-broke.idx]
  }
  
  pred.train = pred.data

  if (type == "regr" | type == "classif") {
    pred.data = as.data.frame(pred.data)
  } else {
    pred.data = as.data.frame(lapply(pred.data, function(X) X)) #X[,-ncol(X)]))
  }

  # now fit the super learner for predicted_pred.data --> target
  pred.data[[td$target]] = getTaskTargets(task)
  if (use.feat) {
    # add data with normal features
    feat = getTaskData(task)
    feat = feat[, colnames(feat) %nin% td$target, drop = FALSE]
    pred.data = cbind(pred.data, feat)
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data,
      target = td$target)
  } else {
    super.task = makeSuperLearnerTask(learner$super.learner$type, data = pred.data, target = td$target)
  }
  messagef("Super learner '%s' will be trained with %s features and %s observations", learner$super.learner$id, getTaskNFeats(super.task), getTaskSize(super.task))
  super.model = train(learner$super.learner, super.task)
  list(method = "stack.no.cv", base.models = base.models,
       super.model = super.model, pred.train = pred.train)
}





compressBaseLearners = function(learner, task, parset = list()) {
  lrn = learner
  lrn$method = "hill.climb"
  ensemble.model = train(lrn, task)

  data = getTaskData(task, target.extra = TRUE)
  data = data[[1]]

  pseudo.data = do.call(getPseudoData, c(list(data), parset))
  pseudo.target = predict(ensemble.model, newdata = pseudo.data)
  pseudo.data = data.frame(pseudo.data, target = pseudo.target$data$response)

  td = ensemble.model$task.desc
  type = ifelse(td$type == "regr", "regr",
    ifelse(length(td$class.levels) == 2L, "classif", "multiclassif"))

  if (type == "regr") {
    new.task = makeRegrTask(data = pseudo.data, target = "target")
    if (is.null(learner$super.learner)) {
      m = makeLearner("regr.nnet", predict.type = )
    } else {
      m = learner$super.learner
    }
  } else {
    new.task = makeClassifTask(data = pseudo.data, target = "target")
    if (is.null(learner$super.learner)) {
      m = makeLearner("classif.nnet", predict.type = "")
    } else {
      m = learner$super.learner
    }
  }

  super.model = train(m, new.task)

  list(method = "compress", base.learners = lrn$base.learners, super.model = super.model,
       pred.train = pseudo.data)
}






getPseudoData = function(.data, k = 3, prob = 0.1, s = NULL, ...) {
  res = NULL
  n = nrow(.data)
  ori.names = names(.data)
  feat.class = sapply(.data, class)
  ind2 = which(feat.class == "factor")
  ind1 = setdiff(1:ncol(.data), ind2)
  if (length(ind2)>0)
    ori.labels = lapply(.data[[ind2]], levels)
  .data = lapply(.data, as.numeric)
  .data = as.data.frame(.data)
  # Normalization
  mn = rep(0, ncol(.data))
  mx = rep(0, ncol(.data))
  for (i in ind1) {
    mn[i] = min(.data[,i])
    mx[i] = max(.data[,i])
    .data[, i] = (.data[, i]-mn[i])/(mx[i]-mn[i])
  }
  if (is.null(s)) {
    s = rep(0, ncol(.data))
    for (i in ind1) {
      s[i] = sd(.data[,i])
    }
  }
  testNumeric(s, len = ncol(.data), lower = 0)

  # Func to calc dist
  hamming = function(mat) {
    n = nrow(mat)
    m = ncol(mat)
    res = matrix(0,n,n)
    for (j in 1:m) {
      unq = unique(mat[,j])
      p = length(unq)
      for (i in 1:p) {
        ind = which(mat[,j] == unq[i])
        res[ind, -ind] = res[ind, -ind]+1
      }
    }
    return(res)
  }

  one.nn = function(mat, ind1, ind2) {
    n = nrow(mat)
    dist.mat.1 = matrix(0,n,n)
    dist.mat.2 = matrix(0,n,n)
    if (length(ind1)>0) {
      dist.mat.1 = as.matrix(stats::dist(mat[,ind1, drop = FALSE]))
    }
    if (length(ind2)>0) {
      dist.mat.2 = hamming(mat[,ind2, drop = FALSE])
    }
    dist.mat = dist.mat.1+dist.mat.2
    neighbour = max.col( -dist.mat - diag(Inf, n))
    return(neighbour)
  }

  # Get the neighbour
  neighbour = one.nn(.data, ind1, ind2)

  # Start the loop
  p = ncol(.data)
  for (loop in 1:k) {
    data = .data
    prob.mat = matrix(sample(c(0,1), n*p, replace = TRUE, prob = c(prob, 1-prob)), n, p)
    prob.mat = prob.mat == 0
    for (i in 1:n) {
      e = as.numeric(data[i, ])
      ee = as.numeric(data[neighbour[i], ])

      # continuous
      for (j in ind1) {
        if (prob.mat[i,j]) {
          current.sd = abs(e[j]-ee[j])/s[j]
          tmp1 = rnorm(1,ee[j], current.sd)
          tmp2 = rnorm(1,e[j], current.sd)
          e[j] = tmp1
          ee[j] = tmp2
        }
      }
      for (j in ind2) {
        if (prob.mat[i,j]) {
          tmp = e[j]
          e[j] = ee[j]
          ee[j] = tmp
        }
      }

      data[i,] = ee
      data[neighbour[i],] = e
    }
    res = rbind(res, data)
  }
  for (i in ind1)
    res[,i] = res[,i]*(mx[i]-mn[i])+mn[i]
  res = data.frame(res)
  names(res) = ori.names
  for (i in ind2)
    res[[i]] = factor(res[[i]], labels = ori.labels[[i]])
  return(res)
}