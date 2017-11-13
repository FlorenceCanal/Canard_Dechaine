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

