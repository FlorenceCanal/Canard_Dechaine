
debut <- Sys.time()

library(dplyr)
library(caret)
library(corrplot)
library(mice)
library(lattice)
library(VIM)
library(car)
library(leaps)

#----------------------------------------

##################################
#                                #
#   DATA WEATHER : TEAM SAKHIR   #
#                                #
##################################


#----------
# Loading and rearrange data :
#----------

# setwd("~/M2/Apprentissage/Projet/GIT")
df <- read.csv("./Sakhir/code/final_train.csv", sep=";", dec = ".")

str(df)

df <- df %>% mutate(
  date = as.Date(date, "%Y-%m-%d"),
  insee = as.factor(as.character(insee)),
  # ddH10_rose4 = direction du vent (126 == "" -> NA ou O ?),
  ddH10_rose4 = as.numeric(as.character(ddH10_rose4)),
  ech = as.factor(ech),
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

table(df$ddH10_rose4, useNA = "ifany")





#----------
# NA values :
#----------

# test de pr�sence de valeur manquante
pMiss <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(df,2,pMiss))

X <- df[,!(colnames(df) %in% c("date", "insee", "ech", "mois"))]

# md.pattern(df)
aggr_plot <- aggr(X, col=c('navyblue','red'), numbers=T, sortVars=T, labels=names(X), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
summary(aggr_plot)
aggr_plot$percent
aggr_plot$x














#----------
# NA imputation :
#----------

marginplot(cbind(X$capeinsSOL0,X$pMER0))

# Imputing the missing values
#methods(mice)
tempX <- mice(X,m=5,maxit=50,meth='pmm',seed=500)
summary(tempX)
# tempX$imp$tH2
# tempX$meth

xyplot(tempX,tH2_obs ~ capeinsSOL0 + tH2 + tpwHPA850, pch=1, cex=1)
densityplot(tempX)
# stripplot(tempX, pch = 20, cex = 1.2)

completedX <- complete(tempX,2)

newdf <- cbind(df$date, df$insee, completedX, df$ech, df$mois)





#----------
# Correlation :
#----------

X <- as.matrix(df[,!(colnames(df) %in% c("date", "insee", "tH2_obs", "ech", "mois", "flvis1SOL0"))])
X <- scale(X, center = T, scale = T)
Y <- as.matrix(df$tH2_obs)
corr <- cor(cbind(Y,X), method = "pearson", use = "complete.obs")
corrplot(corr, order = "hclust", hclust.method = "ward.D2", diag = F, type="upper")

# Applying a correlation filter :
# highCorr <- findCorrelation(descrCorr, 0.85)
# X <- X[, -highCorr]
# X <- scale(X, center = T, scale = T)
# corr <- cor(X, method = "pearson", use = "complete.obs")
# corrplot(corr, order = "hclust")







#----------
# 1st submission -> predictions without any correction :
#----------
df <- read.csv("./Sakhir/data/test/test_answer_template.csv", sep=";")
test <- read.csv("./Sakhir/data/test/test.csv", sep=";")

head(df)
df$tH2_obs <- test$tH2

# write.csv2(df, file = "./Sakhir/submission/sub_1.csv", quote = F, row.names=F)

head(test)
















#----------
# 2nd submission -> predictions from linear regression model :
#----------

test <- sample(nrow(df), size = round(nrow(df)*0.3), replace = F)
learn <- df[-test,]
test <- df[test,]

df <- learn

df$tH2[is.na(df$tH2)] <- mean(df$tH2, na.rm = T)
delta <- df$tH2_obs-df$tH2
X <- as.matrix(cbind(df[,!(colnames(df) %in% c("date", "insee", "tH2", "tH2_obs", "ech", "mois", "flvis1SOL0"))], delta))
X <- scale(X, center = T, scale = T)
corr <- cor(cbind(X), method = "pearson", use = "complete.obs")
corrplot(corr, order = "hclust", hclust.method = "ward.D2", diag = F, type="upper")


dfmod <- as.data.frame(cbind(delta, df$insee, df$ciwcH20, df$clwcH20, df$ddH10_rose4, df$ffH10, df$flir1SOL0, df$fllat1SOL0, df$flsen1SOL0, df$hcoulimSOL0, df$huH2, df$iwcSOL0, df$nbSOL0_HMoy, df$nH20, df$ntSOL0_HMoy, df$pMER0, df$rr1SOL0, df$rrH20, df$tH2, df$tH2_VGrad_2.100, df$tH2_XGrad, df$tH2_YGrad, df$tpwHPA850, df$ux1H10, df$vapcSOL0, df$vx1H10, df$ech))
colnames(dfmod) <- c("delta", "insee", "ciwcH20", "clwcH20", "ddH10_rose4", "ffH10", "flir1SOL0", "fllat1SOL0", "flsen1SOL0", "hcoulimSOL0", "huH2", "iwcSOL0", "nbSOL0_HMoy", "nH20", "ntSOL0_HMoy", "pMER0", "rr1SOL0", "rrH20", "tH2", "tH2_VGrad_2.100", "tH2_XGrad", "tH2_YGrad", "tpwHPA850", "ux1H10", "vapcSOL0", "vx1H10", "ech")

stepwise <- regsubsets(delta ~ . , data = dfmod, method = "seqrep", nbest=1) # method = "forward" / "backward"
par(mfrow=c(1,2))
plot(stepwise, scale = "adjr2", main = "Stepwise Selection\nAIC")
plot(stepwise, scale = "bic", main = "Stepwise Selection\nBIC")
par(mfrow=c(1,1))

reg <- lm(delta ~ ., data = dfmod)
summary(reg)

reg <- lm(delta ~ clwcH20 + ddH10_rose4 + ffH10 + fllat1SOL0 + flsen1SOL0 + hcoulimSOL0 + huH2 +
            nH20 + pMER0 + tH2 + tH2_VGrad_2.100 + tH2_YGrad + tpwHPA850 + vapcSOL0 + vx1H10 + ech, data = dfmod)
summary(reg)

dfmod2 <- dfmod[,c("delta", "clwcH20", "ddH10_rose4", "ffH10", "fllat1SOL0", "flsen1SOL0", "hcoulimSOL0", "huH2", "nH20", "pMER0", "tH2", "tH2_VGrad_2.100", "tH2_YGrad", "tpwHPA850", "vapcSOL0", "vx1H10")]#, "ech")]

X <- as.matrix(dfmod2)
X <- scale(X, center = T, scale = T)
corr <- cor(X, method = "pearson", use = "complete.obs")
corrplot(corr, order = "hclust", hclust.method = "ward.D2", diag = F, type="upper")



reg <- lm(delta ~ ddH10_rose4 + fllat1SOL0 + flsen1SOL0 + hcoulimSOL0 + huH2 +
            nH20 + pMER0 + tH2 + tH2_VGrad_2.100 + tH2_YGrad + vapcSOL0 + vx1H10 + as.factor(ech) + as.factor(insee), data = dfmod)
summary(reg)




VALID <- read.csv("./Sakhir/data/test/test.csv", sep=";")















#----------------------------------------
fin <- Sys.time()
print(fin-debut)

