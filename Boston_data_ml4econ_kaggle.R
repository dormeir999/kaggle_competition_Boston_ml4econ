## ----setup, include=FALSE-------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ---- echo=TRUE, results='hide', message=FALSE, warning=FALSE-------------------------------------------------------------
install.packages("readr", repos = "http://cran.us.r-project.org")
library(readr)

train <- read_csv("data/train.csv")
attach(train)
test <- read_csv("data/test.csv")
submissionExample <- read_csv("data/submissionExample.csv")


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
#install.packages("MASS", repos = "http://cran.us.r-project.org")
library(MASS)
#Boston?
str(Boston)
summary(Boston)


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
which(is.na(train))
which(is.na(test))


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
plot(train[-1])


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
train_scaled <- scale(train[-1])
boxplot(train_scaled)


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
heatmap(cor(train_scaled))


## ---- echo=TRUE, results='hide', message=FALSE, warning=FALSE-------------------------------------------------------------
install.packages("glmnetUtils", repos = "http://cran.us.r-project.org") 
library(glmnetUtils)
elastic_net <- cva.glmnet(medv ~ . - ID, data = train) 


## ---- echo=TRUE, results='hide', message=FALSE, warning=FALSE-------------------------------------------------------------
install.packages("data.table", repos = "http://cran.us.r-project.org") 
library(data.table)

num_alphas <- length(elastic_net$alpha)
table <- data.table()

for (i in 1:num_alphas){
  alpha <- elastic_net$alpha[i] # A given alpha
  min_lambda <- elastic_net$modlist[[i]]$lambda.min # Lambda that minimizes CV-MSE for the given alpha
  min_mse <-  min(elastic_net$modlist[[i]]$cvm) # The minimum value of CV-MSE over lambdas for the given alpha
  
  new_row <- data.table(alpha, min_lambda, min_mse)
  table <- rbind(table, new_row)
}

best_alpha_lambda <- table[which.min(table$min_mse)]
colnames(best_alpha_lambda) <- c("Optimal alpha", "Optimal lambda", "CV-MSE")
best_alpha  <- c(as.matrix(best_alpha_lambda[1,"Optimal alpha"]))
best_lambda <- c(as.matrix(best_alpha_lambda[1,"Optimal lambda"]))


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
best_alpha_lambda


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
predicted_test_elastic_net  <- predict(elastic_net, s = best_lambda, alpha = best_alpha, newdata = test[-1])
predicted_train_elastic_net <- predict(elastic_net, s = best_lambda, alpha = best_alpha, newdata = train[-c(1,15)])


## ---- echo=TRUE, results='hide', message=FALSE, warning=FALSE-------------------------------------------------------------
install.packages("pls", repos = "http://cran.us.r-project.org")
library(pls)


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
pls <- plsr(medv ~ . - ID, data = train, scale = TRUE, validation = "CV")

summary(pls)
validationplot(pls)


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
predicted_train_pls <- predict(pls, newdata = train, ncomp = 5) 
predicted_test_pls  <- predict(pls, newdata = test, ncomp = 5)


## ---- echo=TRUE, results='hide', message=FALSE, warning=FALSE-------------------------------------------------------------
install.packages("randomForest", repos = "http://cran.us.r-project.org")
library(randomForest)


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
random_forest <- randomForest(medv ~ . - ID, data = train) 

random_forest


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
predicted_test_random_forest  <- predict(random_forest, newdata = test)
predicted_train_random_forest <- predict(random_forest, newdata = train)


## ---- echo=TRUE, results='hide', message=FALSE, warning=FALSE-------------------------------------------------------------
install.packages("gbm", repos = "https://CRAN.R-project.org")
library(gbm)

boosting <- gbm(train$medv ~ predicted_train_elastic_net + predicted_train_pls + predicted_train_random_forest - 1, distribution="gaussian")
weights <- as.matrix(summary(boosting, plotit = FALSE)[2])
weights <- weights / sum(weights)


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
weights


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
X <- cbind(predicted_test_elastic_net, predicted_test_pls, predicted_test_random_forest)
ensemble <- X[,1] * weights[2] + X[,2] * weights[3] + X[,3] * weights[1]

submission <- data.frame(test$ID, ensemble)
colnames(submission) <- c("ID", "medv")
write.csv(submission, file = "submission.csv",row.names=FALSE)

