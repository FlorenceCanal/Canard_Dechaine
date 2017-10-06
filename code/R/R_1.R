
debut <- Sys.time()

#----------------------------------------

##################################
#                                #
#   DATA WEATHER : TEAM SAKHIR   #
#                                #
##################################

# setwd("~/M2/Apprentissage/Projet/GIT")

df <- read.csv("./Sakhir/data/test/test_answer_template.csv", sep=";")
test <- read.csv("./Sakhir/data/test/test.csv", sep=";")

head(df)
df$tH2_obs <- test$tH2

# write.csv2(df, file = "./Sakhir/submission/sub_1.csv", quote = F, row.names=F)

head(test)





