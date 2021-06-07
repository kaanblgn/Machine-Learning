library(tidyverse)
library(lubridate)
library(data.table)
library(SmartEDA)
library(caret)
library(ranger)
library(glmnet)
full_data = read.csv("/home/kaan/Desktop/Lectures/IE582/Project/IE582_Fall20_ProjectTrain.csv")
full_test = read.csv("/home/kaan/Desktop/Lectures/IE582/Project/IE582_Fall20_ProjectTest.csv")

source("/home/kaan/Desktop/Lectures/IE582/Project/example_submission.r")
setDT(full_data)
setDT(full_test)

# ExpData(data=full_data,type=1)
# SmartEDA::ExpReport(full_data,Target="y",op_file = "deneme.html",op_dir = "/home/kaan/Desktop/Lectures/IE582/Project/",label = "y",Rc = "a")
# ExpNumViz(full_data,Page=c(2,2),target = "y",type = 2)


# Remove near zero variance and unnecessary columns
near_zero = paste0("x",nearZeroVar(full_data))
temp_fit = ranger::ranger(y~.,data=full_data,importance = "impurity")
a_imp=ranger::importance(temp_fit)
a_imp=data.table(feat=attr(a_imp,'names'),imp=a_imp)
a_imp=a_imp[order(-imp)]
final_remove = intersect(tail(a_imp,30)$feat,near_zero)
full_data_mod = full_data[,-final_remove,with=F]
full_test_mod = full_test[,-final_remove,with=F]

#### OOB features
oob_fit = ranger::ranger(as.numeric(y)-1~.,data = full_data_mod,num.trees = 500,splitrule = "extratrees",min.node.size = 5)
oob_fit$predictions
oob_fit1 = ranger::ranger(as.numeric(y)-1~.,data = full_data_mod,num.trees = 500,splitrule = "extratrees",min.node.size = 3,classification = T)
oob_fit1$predictions
full_data_mod$xoob1 <- oob_fit$predictions
full_data_mod$xoob2 <- oob_fit1$predictions
full_test_mod$xoob1 <- predict(oob_fit,full_test_mod)$prediction
full_test_mod$xoob2 <- predict(oob_fit1,full_test_mod)$prediction
full_test_res = copy(full_test_mod)


# Good
full_scale_mod = rbind(full_data_mod,full_test_mod)
full_data_mod_scale = as.data.frame(scale(full_scale_mod%>%select(-y)))
full_data_mod = full_data_mod_scale%>%mutate(y=full_scale_mod$y)%>%head(nrow(full_data_mod))%>%as.data.table()
full_test_mod = full_data_mod_scale%>%mutate(y=full_scale_mod$y)%>%as.data.table()%>%tail(nrow(full_test_mod))
full_data_mod$y <- factor(full_data_mod$y)
levels(full_data_mod$y)
# Perform stratified train/test split to keep target balance. 1/4 for test data.
set.seed(1)
test_data = splitstackshape::stratified(full_data_mod,group = "y",size=1/4,replace = F)
train_data = full_data_mod%>%anti_join(test_data)
table(train_data$y)
# Good
set.seed(1)
train_data_extra <- DMwR::SMOTE(y ~ ., train_data, perc.over = 100,perc.under=0)
train_data_xgb = copy(train_data)
train_data=rbind(train_data,train_data_extra)
table(train_data$y)

test_data_res = copy(test_data)

# Start Cross validation for methods

# LASSO, Ridge, Middle
# Penalized Regression Approaches Using Glmnet
lasso_reg = cv.glmnet(x = as.matrix(train_data%>%select(-y)),
                           y=as.numeric(train_data$y)-1,
                           family="binomial",
                           type.measure = c("auc"),
                           nfolds = 10,
                           alpha=1)

test_data_res$lasso_lambda_min = 
  predict(lasso_reg,newx=as.matrix(test_data%>%select(-y)),s=c("lambda.min"),type="response")[,1]


middle_reg = cv.glmnet(x = as.matrix(train_data%>%select(-y)),
                      y=as.numeric(train_data$y)-1,
                      family="binomial",
                      type.measure = c("auc"),
                      nfolds = 10,
                      alpha=0.5)

test_data_res$middle_lambda_min = 
  predict(middle_reg,newx=as.matrix(test_data%>%select(-y)),s=c("lambda.min"),type="response")[,1]

### Compare test performance
test_data_res%>%select(-starts_with("x"))%>%gather(model,prediction,-y)%>%
  transmute(y=as.numeric(y)-1,model,prediction)%>%
  group_by(model)%>%summarise(AUC=MLmetrics::AUC(prediction,y))

### lasso and middle lambda.min 



# RandomForest
control <- trainControl(method='cv', 
                        number=5, 
                        search='grid',
                        summaryFunction = twoClassSummary,
                        classProbs = T)
tune_grid <- expand.grid(.mtry = seq(2,10,2)) 
rf_cv2 <- train(make.names(as.factor(y)) ~ ., 
                data =train_data,
                method = 'rf',
                ntree=1000,
                nodesize=5,
                metric = 'ROC',
                trControl=control,
                tuneGrid = tune_grid)

test_data_res$random_forest = predict(rf_cv2,test_data,type = "prob")[,2]

### Compare test performance
test_data_res%>%select(-starts_with("x"))%>%gather(model,prediction,-y)%>%
  transmute(y=as.numeric(y)-1,model,prediction)%>%
  group_by(model)%>%summarise(AUC=MLmetrics::AUC(prediction,y))


# Boosting

tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = 1000, by = 100),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- trainControl(method='cv', 
                  number=5, 
                  search='grid',
                  summaryFunction = twoClassSummary,
                  classProbs = T,
                  allowParallel = T)


xgb_tune <- caret::train(
  make.names(as.factor(y)) ~ ., 
  data =train_data,
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  metric = 'ROC',
  verbose = TRUE
)

# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)

test_data_res$gxgboost = predict(xgb_tune,test_data,type="prob")[,2]

test_data_res%>%select(-starts_with("x"))%>%gather(model,prediction,-y)%>%
  transmute(y=as.numeric(y)-1,model,prediction)%>%
  group_by(model)%>%summarise(AUC=MLmetrics::AUC(prediction,y))

#### Ensemble
small_ens = test_data_res%>%select(random_forest,lasso_lambda_min)
test_data_res$small_ensemble = rowMeans(small_ens)
full_ens = test_data_res%>%select(-y,-middle_lambda_min,-small_ensemble,-starts_with("x"))
test_data_res$full_ensemble = rowMeans(full_ens)

test_data_res%>%select(-starts_with("x"))%>%gather(model,prediction,-y)%>%
  transmute(y=as.numeric(y)-1,model,prediction)%>%
  group_by(model)%>%summarise(AUC=MLmetrics::AUC(prediction,y))

### small ensemble
temp_res= test_data_res%>%select(-starts_with("x"))%>%gather(model,prediction,-y)%>%
  transmute(y=as.numeric(y)-1,model,prediction)%>%mutate(pred_y=ifelse(prediction>=0.5,1,0))%>%
  group_by(model)%>%summarise(AUC=MLmetrics::AUC(prediction,y),Accuracy=MLmetrics::Accuracy(pred_y,y),
                              BAC =1 -(sum(pred_y==0&y==1)/sum(y==1) + sum(pred_y==1&y==0)/sum(y==0))/2)

test_data_res%>%select(-starts_with("x"))%>%gather(model,prediction,-y)%>%
  transmute(y=as.numeric(y)-1,model,prediction)%>%mutate(pred_y=ifelse(prediction>=0.3,1,0))%>%
  group_by(model)%>%summarise(AUC=MLmetrics::AUC(prediction,y),Accuracy=MLmetrics::Accuracy(pred_y,y),
                              BAC =1 -(sum(pred_y==0&y==1)/sum(y==1) + sum(pred_y==1&y==0)/sum(y==0))/2)



table(full_data_mod$y)
set.seed(1)
full_data_mod_extra <- DMwR::SMOTE(y ~ ., full_data_mod, perc.over = 100,perc.under=0)
full_data_mod=rbind(full_data_mod,full_data_mod_extra)
table(full_data_mod$y)


lasso_final = cv.glmnet(x = as.matrix(full_data_mod%>%select(-y)),
                        y=as.numeric(full_data_mod$y)-1,
                        family="binomial",
                        type.measure = c("auc"),
                        nfolds = 10,
                        alpha=1)

full_test_res$lasso_lambda_min = 
  predict(lasso_final,newx=as.matrix(full_test_mod%>%select(-y)),s=c("lambda.min"),type="response")[,1]

fitControl <- trainControl(method = "none", classProbs = TRUE)
rf_final <- train(make.names(as.factor(y)) ~ ., 
                  data =full_data_mod,
                  method = 'rf',
                  ntree=1000,
                  nodesize=5,
                  tuneGrid = data.frame(mtry=rf_cv2$bestTune$mtry),
                  trControl=fitControl
)

full_test_res$random_forest = 
  predict(rf_final,full_test_mod,type="prob")[,2]

fitControl <- trainControl(method = "none", classProbs = TRUE)
xgb_final <- caret::train(
  make.names(as.factor(y)) ~ ., 
  data =full_data_mod,
  trControl = fitControl,
  tuneGrid = xgb_tune$bestTune,
  method = "xgbTree",
  metric = 'ROC',
  verbose = TRUE
)

full_test_res$gxgboost = 
  predict(xgb_final,full_test_mod,type="prob")[,2]

small_ens = full_test_res%>%select(random_forest,lasso_lambda_min)
full_test_mod$small_ensemble = rowMeans(small_ens)

plot(small_ens)

# submission

# # this part is main code
subm_url = 'http://46.101.121.83'

u_name = "The Miners"
p_word = "VW5GwVC0Eqy7exvm"
submit_now = TRUE

username = u_name
password = p_word

token = get_token(username=u_name, password=p_word, url=subm_url)
# this part is where you need to provide your prediction method/function or set of R codes
predictions=full_test_mod$small_ensemble

send_submission(predictions, token, url=subm_url, submit_now= submit_now)
#
#


#### small ensemble 87.3
#### lambdamin 88.1
#### random 87.7
#### random ,lambda min 89.07