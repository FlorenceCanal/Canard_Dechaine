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


X <- df[,!(colnames(df) %in% c("date", "insee", "mois"))]

# md.pattern(df)
aggr_plot <- aggr(X, col=c('navyblue','red'), numbers=T, sortVars=T, labels=names(X), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
summary(aggr_plot)
aggr_plot$percent

# Correlation
X <- scale(X, center = T, scale = T)
corr <- cor(X, method = "pearson", use = "complete.obs")
corrplot(corr, order = "hclust", hclust.method = "ward.D2", diag = F, type="upper")

marginplot(cbind(X$fllat1SOL0, X$flsen1SOL0))

# Imputing the missing values
#methods(mice)
tempdf <- mice(X, m=5, maxit=50, meth='pmm', seed=500) # cart
# summary(tempdf)

# tempdf$imp$tH2
# tempdf$meth

#xyplot(tempdf,tH2_obs ~ capeinsSOL0 + tH2 + tpwHPA850, pch=1, cex=1)
densityplot(tempdf)
# stripplot(tempdf, pch = 20, cex = 1.2)

completedX <- complete(tempX,2)

newdf <- cbind(df$date, df$insee, completedX, df$ech, df$mois)













#----------------------------------------
fin <- Sys.time()
print(fin-debut)

