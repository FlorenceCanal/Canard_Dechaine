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
#df <- read.csv("./Sakhir/data/train_imputNA.csv", sep=";", dec = ".")

# str(df)

df <- df %>% mutate(
  date = as.Date(date, "%Y-%m-%d"),
  insee = as.factor(as.character(insee)),
  ddH10_rose4 = as.numeric(as.character(ddH10_rose4)),
  ech = as.numeric(ech),
  mois = as.character(mois)
) %>% select(
  -capeinsSOL0
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

str(df)

# prop.table(table(df$ddH10_rose4, useNA = "ifany"))*100










#----------
# NA values (dealing with blocs) :
#----------

# test de pr?sence de valeur manquante
pMiss <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(df,2,pMiss))


X <- df[,!(colnames(df) %in% c("date", "insee", "ech", "mois"))]

# md.pattern(df)
aggr_plot <- aggr(X, col=c('navyblue','red'), numbers=T, sortVars=T, labels=names(X), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
summary(aggr_plot)
aggr_plot$percent
# aggr_plot$x

nb_na <- 1 * data.frame(is.na(df))
table(rowSums(nb_na))
nb_rows = nrow(df)

#df = df[rowSums(na) < 12,]
#print(paste("Nombre de lignes supprim?es (trop de manquants) =", nb_rows - nrow(df)))
#print(paste("Nombre de lignes restantes =", nrow(df)))

#nb_na <- 1 * data.frame(is.na(df))
#table(rowSums(nb_na))

#aggr_plot <- aggr(df, col=c('navyblue','red'), numbers=F, sortVars=T, labels=names(df),
#                  cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

#write.table(df, file = "./new_df.csv", row.names = F, sep=";", dec = ".", quote = F)





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
test <- read.csv("./Sakhir/data/test/test.csv", sep=";")

head(test)
test$tH2_obs <- test$tH2

# write.csv2(df, file = "./Sakhir/submission/sub_1.csv", quote = F, row.names=F)
















#----------
# 2nd submission -> predictions from linear regression model :
#----------

test <- sample(nrow(df), size = round(nrow(df)*0.3), replace = F)
learn <- df[-test,]
test <- df[test,]

df <- learn

df$tH2[is.na(df$tH2)] <- mean(df$tH2, na.rm = T)
# ecart <- df$tH2_obs-df$tH2

dfmod <- df[,c("ecart", "insee", "ciwcH20", "clwcH20", "ddH10_rose4", "ffH10", "flir1SOL0", "fllat1SOL0", "flsen1SOL0", "hcoulimSOL0", "huH2", "iwcSOL0", "nbSOL0_HMoy", "nH20", "ntSOL0_HMoy", "pMER0", "rr1SOL0", "rrH20", "tH2", "tH2_VGrad_2.100", "tH2_XGrad", "tH2_YGrad", "tpwHPA850", "ux1H10", "vapcSOL0", "vx1H10", "ech")]
#colnames(dfmod)[1] <- "ecart"

stepwise <- regsubsets(ecart ~ . , data = dfmod, method = "seqrep", nbest=1) # method = "forward" / "backward"
par(mfrow=c(1,2))
plot(stepwise, scale = "adjr2", main = "Stepwise Selection\nAIC")
plot(stepwise, scale = "bic", main = "Stepwise Selection\nBIC")
par(mfrow=c(1,1))

reg <- lm(ecart ~ ., data = dfmod)
summary(reg)

reg <- lm(ecart ~ clwcH20 + ddH10_rose4 + ffH10 + fllat1SOL0 + flsen1SOL0 + hcoulimSOL0 + huH2 +
            nH20 + pMER0 + tH2 + tH2_VGrad_2.100 + tH2_YGrad + tpwHPA850 + vapcSOL0 + vx1H10 + ech + insee, data = dfmod)
summary(reg)

dfmod2 <- dfmod[,c("ecart", "clwcH20", "ddH10_rose4", "ffH10", "fllat1SOL0", "flsen1SOL0", "hcoulimSOL0", "huH2", "nH20", "pMER0", "tH2", "tH2_VGrad_2.100", "tH2_YGrad", "tpwHPA850", "vapcSOL0", "vx1H10", "ech", "insee")]

reg <- lm(ecart ~ clwcH20 + factor(ddH10_rose4) + ffH10 + fllat1SOL0 + flsen1SOL0 +
            hcoulimSOL0 + huH2 + nH20 + tH2 + tH2_VGrad_2.100 + tH2_YGrad +
            tpwHPA850 + vapcSOL0 + vx1H10 + ech + insee, data = dfmod2)
summary(reg)

# X <- as.matrix(dfmod2[!(colnames(dfmod2) %in% "insee")])
# X <- scale(X, center = T, scale = T)
# corr <- cor(X, method = "pearson", use = "complete.obs")
# corrplot(corr, order = "hclust", hclust.method = "ward.D2", diag = F, type="upper")
# 
reg <- lm(ecart ~ factor(ddH10_rose4) + fllat1SOL0 + hcoulimSOL0 + huH2 +
            tH2 + tH2_VGrad_2.100 + tH2_YGrad +
            ech + insee, data = dfmod2)
summary(reg)



# reg <- lm.ridge(ecart ~ ddH10_rose4 + fllat1SOL0 + flsen1SOL0 + hcoulimSOL0 + huH2 +
#             nH20 + pMER0 + tH2 + tH2_VGrad_2.100 + tH2_YGrad + vapcSOL0 + vx1H10 +
#             ech + insee, data = dfmod)
# summary(reg)


table(is.na(test$ecart))
test$ecart[is.na(test$ecart)] <- 0

# NA PROBLEM -> imputation
pred = predict.lm(reg, test)
pred[is.na(pred)] = 0

plot(pred, test$ecart)
abline(a=0, b=1)

RMSE = mean((test$ecart - pred) ^2)
print(RMSE)

# RMSE1 = 1.838155

# RMSE2 = 1.57


# BEST (if no NA)

model <- lme(ecart ~ ddH10_rose4 + fllat1SOL0 + hcoulimSOL0 + tH2_VGrad_2.100 + tH2_YGrad +
               ech*insee + tH2 + huH2, data=df, random=~1|date/insee, na.action=na.exclude)
anova(model)

pred = predict(model, test, na.action = na.pass)
pred[is.na(pred)] = 0
plot(pred, test$ecart)
abline(a=0, b=1)
RMSE = mean((test$ecart - pred) ^2)
print(RMSE)





#----------

VALID <- read.csv("./Sakhir/data/test/test.csv", sep=";", dec = ",")

df <- VALID
str(df)

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

resu[is.na(resu)] = 0

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




