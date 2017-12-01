debut <- Sys.time()

library(plyr)
library(dplyr)
library(caret)
library(corrplot)
library(mice)
library(lattice)
library(VIM)
library(car)
library(leaps)
library(DAAG)
library(randomForest)
library(MASS)
library(nlme)
library(lme4)

setwd("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT")

#----------------------------------------

##################################
#                                #
#   DATA WEATHER : TEAM SAKHIR   #
#                                #
##################################

# Cf. http://rug.mnhn.fr/semin-r/PDF/semin-R_lme_SBallesteros_181208.pdf
# Cf. https://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-scenar-reg-penal-prostate.pdf
# Cf. https://www.stat4decision.com/fr/consulting-data-science/outils/methodes-statistiques/


#----------
# Loading and rearrange data :
#----------

df <- read.csv("./Sakhir/data/final_train.csv", sep=";", dec = ".")
df$ddH10_rose4 = as.factor(as.numeric(as.character(df$ddH10_rose4)))


df <- df %>% mutate(
  date = as.Date(date, "%Y-%m-%d"),
  insee = as.factor(as.character(insee)),
  ddH10_rose4 = as.factor(as.character(ddH10_rose4)),
  ech = as.numeric(ech)
) %>% select(
  -capeinsSOL0
)

reality = df$tH2_obs

df <- df[,!(colnames(df) %in% c("tH2_obs", "capeinsSOL0", "mois"))] # Empty

str(df)



stepwise <- regsubsets(ecart ~ ., data = df, method = "seqrep", nbest=1) # method = "forward" / "backward"
par(mfrow=c(1,2))
plot(stepwise, scale = "adjr2", main = "Stepwise Selection\nAIC")
plot(stepwise, scale = "bic", main = "Stepwise Selection\nBIC")
par(mfrow=c(1,1))


# select model
# ecart ~ insee*ech + flsen1sol0 + flvis1sol0 + hcoulimSOL0 + huH2 + tH2 + tH2_YGrad
# select useful var
# impute only needed var

# ...


dfmod <- df[,colnames(df) %in% c("mois","ecart","insee","ech","ddH10_rose4","flsen1SOL0","flvis1SOL0","hcoulimSOL0","huH2","tH2","tH2_YGrad")]
dfmod$insee <- as.factor(as.character(dfmod$insee))

test <- sample(nrow(dfmod), size = round(nrow(dfmod)*0.8), replace = F)
learn <- dfmod[-test,]
test <- dfmod[test,]

# Change NA to mean
for(i in 1:ncol(learn)){
  if(class(learn[,i]) == "numeric"){
    learn[is.na(learn[,i]), i] <- mean(learn[,i], na.rm = TRUE)
  }
}
for(i in 1:ncol(test)){
  if(class(test[,i]) == "numeric"){
    test[is.na(test[,i]), i] <- mean(test[,i], na.rm = TRUE)
  }
}





#----------
# Regression linéaire

reg <- lm(ecart ~
            ddH10_rose4-1 + flsen1SOL0 +
            flvis1SOL0 + hcoulimSOL0 +
            huH2 + tH2 + tH2_YGrad +
            ech*insee, data = learn)
summary(reg)

pred = predict(reg, test)
table(is.na(pred))
pred[is.na(pred)] = 0
test$ecart[is.na(test$ecart)] = 0
RMSE = mean((test$ecart - pred) ^2)
print(RMSE)
plot(pred, test$ecart, main=paste("Regression Linéaire\nRMSE =", round(RMSE,5)))
abline(a=0, b=1, col = "red")




#----------
# Regression à effets aléatoires mixtes

# ddH10_rose4-1 + flsen1SOL0 + flvis1SOL0 + hcoulimSOL0 + huH2 + tH2 + tH2_YGrad
model <- lme(fixed = ecart ~ ech*insee*ddH10_rose4*tH2 + hcoulimSOL0 + huH2 + tH2_YGrad,
             data=learn,
             random = ~1|ech/insee,
             na.action=na.exclude)
anova(model)

# model <- lme(fixed = ecart ~ insee + ech + ddH10_rose4:tH2 + ech:ddH10_rose4 +
#                ddH10_rose4 + tH2 + hcoulimSOL0 + huH2 + tH2_YGrad,
#              data=learn,
#              random = ~1|ech/insee,
#              na.action=na.exclude)
# anova(model)

# model <- lmer(ecart ~ ech + (1|insee) + ddH10_rose4*tH2 + hcoulimSOL0 + huH2 + tH2_YGrad,
#              data=learn,
#              na.action=na.exclude)
# anova(model, test='Chisq')

pred = predict(model, test, na.action = na.pass)
table(is.na(pred))
pred[is.na(pred)] = 0
pred[abs(pred) > 8] = 0
RMSE = mean((test$ecart - pred) ^2)
print(RMSE)
plot(pred, test$ecart, main=paste("Regression Mixte\nRMSE =", round(RMSE,5)))
abline(a=0, b=1, col = "red")





#----------
# Lasso regression

library(lars)

dfmod2 <- dfmod[,colnames(dfmod) %in% c("ecart","ech","flsen1SOL0","flvis1SOL0","hcoulimSOL0","huH2","tH2","tH2_YGrad")]
test_lasso <- sample(nrow(dfmod2), size = round(nrow(dfmod2)*0.8), replace = F)
learn_lasso <- dfmod2[-test_lasso,]
test_lasso <- dfmod2[test_lasso,]
# Change NA to mean
for(i in 1:ncol(learn_lasso)){
  if(class(learn_lasso[,i]) == "numeric"){
    learn_lasso[is.na(learn_lasso[,i]), i] <- mean(learn_lasso[,i], na.rm = TRUE)
  }
}
for(i in 1:ncol(test_lasso)){
  if(class(test_lasso[,i]) == "numeric"){
    test_lasso[is.na(test_lasso[,i]), i] <- mean(test_lasso[,i], na.rm = TRUE)
  }
}

Independent_variable <- as.matrix(learn_lasso[,!(colnames(learn_lasso) %in% "ecart")])
Dependent_Variable <- as.matrix(learn_lasso$ecart)
laa <- lars(Independent_variable, Dependent_Variable, type = 'lasso')
# plot(laa)
coef(laa, s = which.min(summary(laa)$Cp), mode="step")

best_step <- laa$df[which.min(laa$RSS)]
x = as.matrix(test_lasso[,!(colnames(test_lasso) %in% "ecart")])
predict <- predict(laa, x, s=best_step, type="fit")$fit
table(is.na(predict))

RMSE = mean((test_lasso$ecart - predict) ^2)
print(RMSE)
plot(predict, test_lasso$ecart, main=paste("Regression Lasso\nRMSE =", round(RMSE,5)))
abline(a=0, b=1, col = "red")





#----------
# XGboost

library(xgboost)

# learn_xgb <- as.matrix(learn_lasso, rownames.force=NA)
# test_xgb <- as.matrix(test_lasso, rownames.force=NA)
# learn_xgb <- as(learn_xgb, "sparseMatrix")
# test_xgb <- as(test_xgb, "sparseMatrix")

# Never forget to exclude objective variable in 'data option'
# learn_Data <- xgb.DMatrix(data = as.matrix(learn[,seq(1,9)]), label = as.matrix(learn[,"ecart"]))

# Tuning the parameters #
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1, number = 3)

xgb.grid <- expand.grid(nrounds = 500,
                        max_depth = seq(6,10),
                        eta = c(0.01,0.3, 1),
                        gamma = c(0.0, 0.2, 1),
                        colsample_bytree = c(0.5,0.8, 1),
                        min_child_weight = seq(1,9, by=2),
                        subsample = c(0.5,1,5,8,10)
)
print(xgb.grid)

xgb_tune <- train(ecart ~.,
                  data=learn,
                  method="xgbTree",
                  metric = "RMSE",
                  trControl=cv.ctrl,
                  tuneGrid=xgb.grid
)
print(xgb.grid)

predict <- predict(xgb_tune, test)
table(is.na(predict))

RMSE = mean((test$ecart - predict) ^2)
print(RMSE)
plot(predict, test$ecart, main=paste("XGboost\nRMSE =", round(RMSE,5)))
abline(a=0, b=1, col = "red")










#----------
# Application on TEST set

test <- read.csv("./Sakhir/data/test/test.csv", sep=";", dec=",")

test <- test %>% mutate(
  date = as.Date(date, "%Y-%m-%d"),
  insee = as.factor(as.character(insee)),
  ddH10_rose4 = as.factor(as.numeric(as.character(ddH10_rose4))),
  ech = as.numeric(ech),
  flvis1SOL0 = as.numeric(flvis1SOL0)
) %>% select(
  -capeinsSOL0
)

# test$ecart = rep(0, nrow(test))

testmod <- test[,colnames(test) %in% c("insee","ech","ddH10_rose4","flsen1SOL0","flvis1SOL0","hcoulimSOL0","huH2","tH2","tH2_YGrad")]
testmod$insee <- as.factor(as.character(testmod$insee))

pred = predict(model, testmod)
prop.table(table(is.na(pred)))*100

hist(pred, breaks = 50)














#----------------------------------------
fin <- Sys.time()
print(fin-debut)




