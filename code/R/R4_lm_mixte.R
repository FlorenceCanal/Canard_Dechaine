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

# # Change NA to mean
# for(i in 1:ncol(df)){
#   if(class(df[,i]) == "numeric"){
#     df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
#   }
# }
# sort(apply(df,2,pMiss))

test <- sample(nrow(dfmod), size = round(nrow(dfmod)*0.7), replace = F)
learn <- dfmod[-test,]
test <- dfmod[test,]


reg <- lm(ecart ~
            ddH10_rose4-1 + flsen1SOL0 +
            flvis1SOL0 + hcoulimSOL0 +
            huH2 + tH2 + tH2_YGrad +
            ech*insee, data = learn)
summary(reg)

pred = predict(reg, test)
table(is.na(pred))
pred[is.na(pred)] = 0
plot(pred, test$ecart)
abline(a=0, b=1, col = "red")
test$ecart[is.na(test$ecart)] = 0
RMSE = mean((test$ecart - pred) ^2)
print(RMSE)



# Change NA to mean
for(i in 1:ncol(learn)){
  if(class(learn[,i]) == "numeric"){
    learn[is.na(learn[,i]), i] <- mean(learn[,i], na.rm = TRUE)
  }
}


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
plot(pred, test$ecart)
abline(a=0, b=1, col = "red")
RMSE = mean((test$ecart - pred) ^2)
print(RMSE)





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


pred = predict(model, testmod)
prop.table(table(is.na(pred)))*100

par(mfrow=c(2,1))
hist(pred, breaks = 50)
hist(dfmod$ecart, breaks = 50)
par(mfrow=c(2,2))




