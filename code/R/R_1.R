
df <- read.csv("~/M2/Apprentissage/Projet/data-meteo/test_answer_template.csv", sep=";")
test <- read.csv("~/M2/Apprentissage/Projet/data-meteo/test.csv", sep=";")

test$tH2
head(df)
df$tH2_obs <- test$tH2

write.csv2(df, file = "~/M2/Apprentissage/Projet/data-meteo/sub_1.csv", quote = F, row.names=F)


head(test)





