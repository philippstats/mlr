# impl. classif prob
# impl. regr

context("Check makeTaskWithNewFeat")

tasks = list(binaryclass.task, multiclass.task)
pts = c("prob", "response")

for (tsk in tasks) {
  for (pt in pts) {
    lrn = makeLearner("classif.rpart", predict.type = pt)
    r = resample(lrn, tsk, cv2)
    if (pt == "prob") {
      new.feat = getPredictionProbabilities(r$pred)
    } else {
      new.feat = getPredictionResponse(r$pred)
    }
    #new.feat %>% class %>% print
    new.task = makeTaskWithNewFeat(task = tsk, new.feat = new.feat, feat.name = "new.feat")  
    #message("org")
    #tsk %>% getTaskData %>% head %>% print 
    #message("new")
    #new.task %>% getTaskData %>% head %>% print 
    #message("###########################################################")
    if (pt == "prob") {
      feat.names = grep("new.feat", names(getTaskData(new.task)))
    } else {
      new = getTaskData(new.task)$new.feat
      expect_that(new, is_a("factor"))
    }
  }
}  


tasks = list(regr.task)
