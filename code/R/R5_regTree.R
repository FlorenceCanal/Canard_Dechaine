debut <- Sys.time()

library(dplyr)
library(caret)
library(corrplot)
library(mice)
library(lattice)
library(VIM)
library(car)
library(leaps)
library(DAAG)
library(MASS)
library(nlme)
library(lme4)
library(randomForest)


setwd("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT")

#----------------------------------------

##################################
#                                #
#   DATA WEATHER : TEAM SAKHIR   #
#                                #
##################################

# https://www.statmethods.net/advstats/cart.html
# https://www.r-bloggers.com/variable-importance-plot-and-variable-selection/
# https://www.r-bloggers.com/random-forests-in-r/


#----------
# Loading and rearrange data :
#----------

df <- read.csv("./Sakhir/data/final_train.csv", sep=";", dec = ".")
df$ddH10_rose4 = as.factor(as.numeric(as.character(df$ddH10_rose4)))


df <- df %>% mutate(
  date = as.Date(date, "%Y-%m-%d"),
  insee = as.factor(as.character(insee)),
  ddH10_rose4 = as.factor(as.integer(as.character(ddH10_rose4))),
  ech = as.numeric(ech)
) %>% select(
  -capeinsSOL0
)

reality = df$tH2_obs

df <- df[,!(colnames(df) %in% c("tH2_obs", "capeinsSOL0"))] # Empty

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

dfmod = subset(df, select=c("ecart", "insee", "ech", "ddH10_rose4",
                            "flsen1SOL0", "flvis1SOL0", "hcoulimSOL0", "huH2",
                            "tH2", "tH2_YGrad", "fllat1SOL0", "tH2_VGrad_2.100"))
dfmod$insee <- as.factor(as.character(dfmod$insee))

# # Change NA to mean
# for(i in 1:ncol(df)){
#   if(class(df[,i]) == "numeric"){
#     df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
#   }
# }
# sort(apply(df,2,pMiss))

set.seed(29)
test <- sample(nrow(dfmod), size = round(nrow(dfmod)*0.7), replace = F)
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


pMiss <- function(x){sum(is.na(x))/length(x)*100}

sort(apply(learn,2,pMiss))
sort(apply(test,2,pMiss))




fit <- randomForest(ecart ~ insee + ech + ddH10_rose4 + flsen1SOL0 + flvis1SOL0 +
                      hcoulimSOL0 + huH2 + tH2 + tH2_YGrad + fllat1SOL0 + tH2_VGrad_2.100,
                    method = "anova", data = learn,
                    xtest = NULL, ytest = NULL,
                    ntree = 750,
                    mtry = 3,
                    replace = FALSE,
                    nodesize = 1 #floor(nrow(learn)*0.0001)
                    )
print(fit) # view results
# 46.3 % of var explained


# importance(fit) # importance of each predictor
# 
# summary(fit) # detailed summary of splits
# 
# par(mfrow=c(1,2))
# plot(fit)
# varImpPlot(fit)
# par(mfrow=c(1,1))
# 
# 
# VI_F=importance(fit)
# # VI_F
# barplot(t(VI_F/sum(VI_F)))



# PREDICTIONS

pred <- predict(fit, test) #Predictions on Test Set for each Tree
table(is.na(pred))
# pred[is.na(pred)] = 0

plot(pred, test$ecart)
abline(a=0, b=1)

RMSE = mean((test$ecart - pred) ^2)
print(RMSE)

# 1.168082










#----------

VALID <- read.csv("./Sakhir/data/test/test.csv", sep=";", dec = ",")
df <- VALID
ecart = NA
df <- as.data.frame(cbind(ecart, df))

df$flvis1SOL0 = as.character(df$flvis1SOL0)
df$flvis1SOL0 = as.numeric(gsub(x = df$flvis1SOL0, pattern = ",", replacement = "."))

date = df$date
insee = df$insee
ech = df$ech

df <- df %>% mutate(
  date = as.Date(date, "%Y-%m-%d"),
  insee = as.factor(as.character(insee)),
  ddH10_rose4 = as.factor(as.integer(as.character(ddH10_rose4))),
  ech = as.numeric(ech)
) %>% select(
  -capeinsSOL0
)

real_test = subset(df, select = c("ecart", "insee", "ech", "ddH10_rose4",
                                  "flsen1SOL0", "flvis1SOL0", "hcoulimSOL0", "huH2",
                                  "tH2", "tH2_YGrad", "fllat1SOL0", "tH2_VGrad_2.100"))
real_test$ddH10_rose4 <- as.factor(as.integer(as.character(real_test$ddH10_rose4)))
real_test$insee <- as.factor(as.character(real_test$insee))

str(real_test)

# Change NA to mean
for(i in 1:ncol(real_test)){
  if(class(real_test[,i]) == "numeric"){
    real_test[is.na(real_test[,i]), i] <- mean(real_test[,i], na.rm = TRUE)
  }
}
sort(apply(real_test,2,pMiss))


resu <- predict(fit, real_test)

table(is.na(resu))
# resu[is.na(resu)] = 0

tH2_obs = real_test$tH2 + resu
# real_test$tH2_obs = tH2_obs

submiss <- data.frame(date, insee, ech, tH2_obs)

head(submiss)

# write.table(submiss, file = "./new_sub.csv", row.names = F, sep=";", dec = ".", quote = F)







#----------------------------------------
fin <- Sys.time()
print(fin-debut)









