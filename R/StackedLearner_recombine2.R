createTestPreds = function(i, bls, test.idx, task, save.on.disc) {
    bls.len = length(bls)
    if (save.on.disc) {
    # This only works if outer resampling is Holdout (save model does do not 
    # get infos about the fold figure, therefore only one fold is allowed): 
    if (i != 1) {
      stopf("Using 'save.on.disc = TRUE' and outer resampling strategies others 
        than Holdout is not supported. Switch save.on.disc to FALSE or use Holdout")
    }
    all.preds = vector("list", length = bls.len)
    for (b in seq_len(bls.len)) { # do it seqentially
      bm = readRDS(bls[[b]]) # i is always 1
      all.preds[[b]] = predict(bm, subsetTask(task, test.idx))
      names(all.preds)[b] = bm$learner$id
      #rm(bm)
    }
  } else { # save.on.disc = FALSE
    all.preds = lapply(seq_len(bls.len), function(b) predict(bls[[b]], subsetTask(task, test.idx)))
    all.preds.names = unlist(lapply(seq_len(bls.len), function(b) bls[[b]]$learner$id))
    names(all.preds) = all.preds.names
  }
  all.preds
}



#'
#'
 
createNewParset = function(org.parset, new.parset) {
  used.org = setdiff(names(org.parset), names(new.parset))
  org.parset = org.parset[used.org]
  final.parset = c(org.parset, new.parset)
  allowed =  c("replace", "init", "bagprob", "bagtime", "metric", "tolerance")
  unallowed = setdiff(names(new.parset), allowed) 
  if (length(unallowed) > 0) 
    stopf("'%s' is no allowed argument for parset.", unallowed)
  final.parset
}



