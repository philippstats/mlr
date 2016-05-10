makeRLearner.classif.nnlogreg = function() {
  makeRLearnerClassif(
    cl = "classif.nnlogreg",
    package = "stats",
    par.set = makeParamSet(),
    properties = c("twoclass", "numerics", "factors", "prob"),
    name = "",
    short.name = "nnlogreg",
    note = "Maximum likelihood estimates of a logistic regression model where slopes are constrained to non-negative values."
  )
}

trainLearner.classif.nnlogreg = function(.learner, .task, .subset, .weights = NULL, ...) {
  f = getTaskFormula(.task, explicit.features = TRUE)
  data = getTaskData(.task, .subset)
  nnlogreg(formula = f, data = data)
}


predictLearner.classif.nnlogreg = function(.learner, .model, .newdata, ...) {
  # add intercept vector to data frame if necessary
  betas = .model$learner.model$beta
  data = createDummyFeatures(.newdata, method = "reference")
  if (names(betas)[1L] == "(Intercept)") {
    data1 = cbind("(Intercept)" = 1L, data)
  } else {
    data1 = .newdata
  }
  data2 = as.matrix(data1)
  x = as.numeric(exp(betas%*%t(data2)) / (1 + exp(betas%*%t(data2))))
  levs = .model$task.desc$class.levels
  if (.learner$predict.type == "prob") {
    propVectorToMatrix(x, levs)
  } else {
    p = as.factor(ifelse(x > 0.5, levs[2L], levs[1L]))
    unname(p)
  }
}


# Author: Thomas Debray (with small adjustments) Version: 22 dec 2011
# http://www.r-bloggers.com/logistic-regression/ 
# accessed 2016-05-09

nnlogreg = function(formula, data) {
  # Define the negative log likelihood function
  logl = function(theta, x, y){
    x = as.matrix(x)
    beta = theta[1:ncol(x)]

    # Use the log-likelihood of the Bernouilli distribution, where p is
    # defined as the logistic transformation of a linear combination
    # of predictors, according to logit(p)=(x%*%beta)
    loglik = sum(-y*log(1 + exp(-(x%*%beta))) - (1-y)*log(1 + exp(x%*%beta)))
    return(-loglik)
  }

  # Prepare the data
  #target.char = as.character(formula[2])
  #X = as.matrix(data[, -target.char])
  
  outcome = rownames(attr(terms(formula), "factors"))[1]
  design = model.frame(data)
  x = as.matrix(model.matrix(formula, data = design))
  #y = as.numeric(data[, match(outcome, colnames(data))]) - 1
  y = as.numeric(data[, outcome]) - 1
  #y %>% head
  #- 1

  # Define initial values for the parameters
  theta.start = rep(0, (dim(x)[2]))
  names(theta.start) = colnames(x)
  
  # Non-negative slopes constraint
  lower = c(-Inf, rep(0, (length(theta.start) - 1)))

  # Calculate the maximum likelihood
  mle = optim(par = theta.start, fn = logl, x = x, y = y, hessian = FALSE, 
    lower = lower, method = "L-BFGS-B")

  # Obtain regression coefficients
  beta = mle$par
 
  # Calculate the Information matrix
  # The variance of a Bernouilli distribution is given by p * (1-p)
  p = 1 / (1 + exp(-x %*% beta))
  V = array(0, dim = c(dim(x)[1], dim(x)[1]))
  diag(V) = p * (1 - p)
  IB = t(x) %*% V %*% x
  vcov = try(solve(IB))
 
  list(beta = beta, vcov = vcov, dev = 2 * mle$value)
}
