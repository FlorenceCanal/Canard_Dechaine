
debut <- Sys.time()

library(dplyr)
library(caret)
library(corrplot)
library(mice)
library(lattice)
library(VIM)

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
df <- read.csv("./Sakhir/data/train_1.csv", sep=";", dec = ",")

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
# NA values -> imputation :
#----------

# test de pr�sence de valeur manquante
pMiss <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(df,2,pMiss))

X <- df[,!(colnames(df) %in% c("date", "insee", "ech", "mois"))]

# md.pattern(df)
aggr_plot <- aggr(X, col=c('navyblue','red'), numbers=T, sortVars=T, labels=names(X), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
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










#----------------------------------------
fin <- Sys.time()
print(fin-debut)

