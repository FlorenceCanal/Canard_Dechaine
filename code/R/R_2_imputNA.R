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


#----------
# Loading and rearrange data :
#----------

df <- read.csv("./Sakhir/data/final_train.csv", sep=";", dec = ".")
df$ddH10_rose4 = as.factor(as.numeric(as.character(df$ddH10_rose4)))
tH2_obs = df$tH2_obs
df <- df[,!(colnames(df) %in% c("tH2_obs"))]
df$tH2_obs = tH2_obs
df$test = FALSE

test <- read.csv("./Sakhir/data/test/test.csv", sep=";", dec=",")
test$ddH10_rose4 = as.factor(as.numeric(as.character(test$ddH10_rose4)))
test$flvis1SOL0 = as.character(test$flvis1SOL0)
test$flvis1SOL0 = gsub(",", ".", test$flvis1SOL0)
test$flvis1SOL0 = as.numeric(test$flvis1SOL0)
test$ecart = 0
test$tH2_obs = 0
test$test = TRUE

df <- rbind(df, test)
# str(df)

df <- df %>% mutate(
  date = as.Date(date, "%Y-%m-%d"),
  insee = as.factor(as.character(insee)),
  ddH10_rose4 = as.factor(as.character(ddH10_rose4)),
  ech = as.numeric(ech),
  mois = as.character(mois)
) %>% select(
  -capeinsSOL0
)

# df$mois[df$mois == "jan"] = "1_jan"
# df$mois[df$mois == "février"] = "2_fev"
# df$mois[df$mois == "mars"] = "3_mar"
# df$mois[df$mois == "avril"] = "4_avr"
# df$mois[df$mois == "mai"] = "5_mai"
# df$mois[df$mois == "juin"] = "6_juin"
# df$mois[df$mois == "juillet"] = "7_juil"
# df$mois[df$mois == "août"] = "8_aout"
# df$mois[df$mois == "septembre"] = "9_sept"
# df$mois[df$mois == "octobre"] = "10_oct"
# df$mois[df$mois == "novembre"] = "11_nov"
# df$mois[df$mois == "décembre"] = "12_dec"
# 
# df <- df %>% mutate(
#   mois = as.factor(mois)
# )

str(df)

# prop.table(table(df$ddH10_rose4, useNA = "ifany"))*100



#----------
# Variance filter :
#----------

for(i in seq(ncol(df))) {
  print(colnames(df)[i])
  print(var(df[,i], na.rm = T))
}





#----------
# NA imputation :
#----------

pMiss <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(df,2,pMiss))

df <- df[!(is.na(df$tH2_obs)),] # Not learn on missing Y
df <- df[,!(colnames(df) %in% c("nH20"))] # TOO MANY NAs
df <- df[,!(colnames(df) %in% c("ciwcH20", "clwcH20"))] # VAR ~= 0
df <- df[,!(colnames(df) %in% c("mois"))] # Useless

X <- df[,!(colnames(df) %in% c("date", "insee", "ddH10_rose4", "ecart", "test"))]

# md.pattern(df)
# dev.off()
aggr_plot <- aggr(X, col=c('navyblue','red'), numbers=T, sortVars=T, labels=names(X), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
summary(aggr_plot)
aggr_plot$percent

# Correlation
X <- scale(X, center = F, scale = T)
corr <- cor(X, method = "pearson", use = "complete.obs")
corrplot(corr, order = "hclust", hclust.method = "ward.D2", diag = F, type="upper")

X <- as.data.frame(X)
# marginplot(cbind(X$fllat1SOL0, X$flsen1SOL0))

# Imputing the missing values
#methods(mice)


X <- cbind(df$insee, df$ddH10_rose4, X, df$ecart)
str(X)

tempdf <- mice(X, m=5, maxit=50, meth='pmm', seed=500) # cart
# summary(tempdf)

# tempdf$imp$tH2
# tempdf$meth

#xyplot(tempdf,tH2_obs ~ capeinsSOL0 + tH2 + tpwHPA850, pch=1, cex=1)
densityplot(tempdf)
# stripplot(tempdf, pch = 20, cex = 1.2)

completedf <- complete(tempdf,2)



newdf <- cbind(df$date, completedf, df$test)
colnames(newdf) <- c('date', 'insee', 'ddH10_rose4', 'tH2_obs', 'ffH10', 'flir1SOL0',
                     'fllat1SOL0', 'flsen1SOL0', 'flvis1SOL0', 'hcoulimSOL0', 'huH2',
                     'iwcSOL0', 'nbSOL0_HMoy', 'ntSOL0_HMoy', 'pMER0', 'rr1SOL0', 'rrH20',
                     'tH2', 'tH2_VGrad_2.100', 'tH2_XGrad', 'tH2_YGrad', 'tpwHPA850',
                     'ux1H10', 'vapcSOL0', 'vx1H10', 'ech', 'ecart', 'test')

# write.table(newdf, "./Sakhir/data/train_imputNA.csv", sep=";", dec = ".", row.names = F, quote = F)

df <- newdf




#----------
# Check NA on TEST set :
#----------

# test <- read.csv("./Sakhir/data/test/test.csv", sep=";", dec=",")
# head(test)

# NA imputation :
pMiss <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(df,2,pMiss))

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

valid <- df[df$test == T,]
valid <- valid[,!(colnames(valid) %in% c("test", "date", "tH2_obs"))]
valid <- valid[,!(colnames(valid) %in% "ecart")]

df <- df[df$test == F,]
df <- df[,!(colnames(df) %in% c("test", "date", "tH2_obs"))]



#----------
# Check better imputation with base linear regression model :
#----------

# df <- read.csv("./Sakhir/data/train_imputNA.csv", sep=";", dec = ".")
# 
# df <- df %>% mutate(
#   insee = as.factor(as.character(insee)),
#   ddH10_rose4 = as.factor(as.character(ddH10_rose4)),
#   ech = as.numeric(ech)
# ) %>% select(
#   -date,
#   -tH2_obs
# )
# 
# str(df)
# 
# # NA values ?
# pMiss <- function(x){sum(is.na(x))/length(x)*100}
# sort(apply(df,2,pMiss))
# 
# # Variance ?
# for(i in seq(ncol(df))) {
#   print(colnames(df)[i])
#   print(var(df[,i], na.rm = T))
# }


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


# reg <- lm(ecart ~ ., data = learn)
# summary(reg)
# cv.lm(reg, data=learn, m=5)
# 
# reg <- lm(ecart ~ insee+ddH10_rose4+flsen1SOL0+flvis1SOL0+huH2+tH2+tH2_VGrad_2.100+tH2_YGrad, data = learn)
# summary(reg)
# cv.lm(reg, data=learn, m=5)


#hcoulimSOL0 huH2 iwcSOL0 nbSOL0_HMoy ntSOL0_HMoy pMER0
#rr1SOL0 rrH20 tH2 tH2_VGrad_2.100 tH2_XGrad tH2_YGrad
#tpwHPA850 ux1H10 vapcSOL0 vx1H10 ech ecart


reg <- lm(ecart ~ factor(ddH10_rose4) + fllat1SOL0 + hcoulimSOL0 +
            tH2 + tH2_VGrad_2.100 + tH2_YGrad + ech +
            insee1 + insee2 + insee3 + insee4 + insee5 + insee6 + insee7, data = learn)
summary(reg)


pred = predict.lm(reg, test)
hist(pred)

plot(x=pred, y=reality, type='p', col="red")
abline(a = 0, b = 1)

RMSE = mean((reality - pred) ^2)
print(RMSE)

# RMSE (origin data) = 1.58
# RMSE (NOW) = 1.54






pred = predict.lm(reg, valid)
hist(pred)












VALID <- read.csv("./Sakhir/data/test/test.csv", sep=";", dec = ",")

df <- VALID
str(df)

X <- df[,!(colnames(df) %in% c("date", "insee", "ddH10_rose4"))]
X <- scale(X, center = T, scale = T)
X <- as.data.frame(X)
X <- cbind(df$insee, df$ddH10_rose4, X, df$ecart)




ecart = NA
df <- as.data.frame(cbind(ecart, df))

df <- df %>% mutate(
  #date = as.Date(date, "%Y-%m-%d"),
  insee = as.factor(as.character(insee)),
  # ddH10_rose4 = direction du vent (126 == "" -> NA ou O ?),
  ddH10_rose4 = as.numeric(as.character(ddH10_rose4)),
  ech = as.numeric(ech),
  mois = as.character(mois)
) %>% select(
  -flvis1SOL0
  # flvis1SOL0 = 1 level ? : var = 0,
)
df$mois[df$mois == "jan"] = "1_jan"
df$mois[df$mois == "février"] = "2_fev"
df$mois[df$mois == "mars"] = "3_mar"
df$mois[df$mois == "avril"] = "4_avr"
df$mois[df$mois == "mai"] = "5_mai"
df$mois[df$mois == "juin"] = "6_juin"
df$mois[df$mois == "juillet"] = "7_juil"
df$mois[df$mois == "août"] = "8_aout"
df$mois[df$mois == "septembre"] = "9_sept"
df$mois[df$mois == "octobre"] = "10_oct"
df$mois[df$mois == "novembre"] = "11_nov"
df$mois[df$mois == "décembre"] = "12_dec"
df <- df %>% mutate(
  mois = as.factor(mois)
)

resu <- predict(reg, df)

tH2_obs = df$tH2 + resu
df$tH2_obs = tH2_obs

submiss <- df %>% select(
  date, insee, ech, tH2_obs
)

head(submiss)

# write.table(submiss, file = "./new_sub.csv", row.names = F, sep=";", dec = ".", quote = F)












#----------------------------------------
fin <- Sys.time()
print(fin-debut)

