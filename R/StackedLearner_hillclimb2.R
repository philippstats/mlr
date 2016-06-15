# ensemble selection algo:

applyEnsembleSelection = function(pred.list = pred.list, bls.length = bls.length,
  bls.names = bls.names, bls.performance = bls.performance, parset = parset) {
  
  # parset
  assertClass(parset, "list")
  replace = parset$replace
  init = parset$init
  bagprob = parset$bagprob
  bagtime = parset$bagtime
  maxiter = parset$maxiter
  metric = parset$metric
  tolerance = parset$tolerance
  #
  m = bls.length
  freq = rep(0, m)
  names(freq) = bls.names
  freq.list = vector("list", bagtime)
  
  for (bagind in seq_len(bagtime)) {
    # bagging of models
    bagsize = ceiling(m * bagprob)
    bagmodel = sample(1:m, bagsize)
    
    # Initial selection of strongest learners
    inds.init = NULL
    inds.selected = NULL
    sel.algo = NULL
    single.scores = rep(ifelse(metric$minimize, Inf, -Inf), m)
    
    # inner loop
    for (i in bagmodel) { #FIX ME use apply
      single.scores[i] = bls.performance[i] #resres[[i]]$aggr
    }
    if (metric$minimize) { # FIXME use own func
      inds.init = order(single.scores)[1:init]
    } else {
      inds.init = rev(order(single.scores))[1:init] 
    }
    freq[inds.init] = freq[inds.init] + 1  
    
    current.pred.list = pred.list[inds.init]
    current.pred = aggregatePredictions(current.pred.list)
    bench.score = metric$fun(pred = current.pred)
    
    #FIXME: maybe i will need them
    inds.selected = inds.init
    
    #FIXME: nicht 2. gl BLs hintereinander (ist gewährleistet wenn tolerance > 0)
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
    sel.algo = bls.names[inds.selected]
    freq.list[[bagind]] = sel.algo
  }
  weights = freq/sum(freq) #TODO: drop in future?
  list(freq = freq, freq.list = freq.list, weights = weights)
}