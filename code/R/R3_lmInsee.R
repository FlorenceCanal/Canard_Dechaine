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
library(randomForest)

setwd("C:/Users/mbriens/Documents/M2/Apprentissage/Projet/GIT")

#----------------------------------------

##################################
#                                #
#   DATA WEATHER : TEAM SAKHIR   #
#                                #
##################################



df <- read.csv("./Sakhir/data/train_imputNA.csv", sep=";", dec = ".")

df <- df %>% mutate(
  insee = as.factor(as.character(insee)),
  ddH10_rose4 = as.factor(as.character(ddH10_rose4)),
  ech = as.numeric(ech)
) %>% select(
  -date,
  -tH2_obs
)

str(df)



df$insee1 = FALSE
df$insee1[df$insee == 31069001] = TRUE
df$insee2 = FALSE
df$insee2[df$insee == 33281001] = TRUE
df$insee3 = FALSE
df$insee3[df$insee == 35281001] = TRUE
df$insee4 = FALSE
df$insee4[df$insee == 59343001] = TRUE
df$insee5 = FALSE
df$insee5[df$insee == 6088001] = TRUE
df$insee6 = FALSE
df$insee6[df$insee == 67124001] = TRUE
df$insee7 = FALSE
df$insee7[df$insee == 75114001] = TRUE

df<- df[,!(colnames(df) %in% c("insee"))]



# NA values ?
pMiss <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(df,2,pMiss))

# Variance ?
for(i in seq(ncol(df))) {
  print(colnames(df)[i])
  print(var(df[,i], na.rm = T))
}


test <- sample(nrow(df), size = round(nrow(df)*0.3), replace = F)
learn <- df[-test,]
test <- df[test,]
reality = test$ecart
test <- test[,!(colnames(test) %in% "ecart")]


stepwise <- regsubsets(ecart ~ . , data = learn, method = "seqrep", nbest=1) # method = "forward" / "backward"
par(mfrow=c(1,2))
plot(stepwise, scale = "adjr2", main = "Stepwise Selection\nAIC")
plot(stepwise, scale = "bic", main = "Stepwise Selection\nBIC")
par(mfrow=c(1,1))

reg <- lm(ecart ~ ., data = learn)
summary(reg)


reg <- lm(ecart ~ factor(ddH10_rose4) + fllat1SOL0 + hcoulimSOL0 + huH2 +
            tH2 + tH2_VGrad_2.100 + tH2_YGrad + ech +
            insee1 + insee2 + insee3 + insee4 + insee5 + insee6 + insee7, data = learn)
summary(reg)

# NA PROBLEM -> imputation
pred = predict.lm(reg, test)
# pred[is.na(pred)] = 0
RMSE = mean((reality - pred) ^2)
print(RMSE)





