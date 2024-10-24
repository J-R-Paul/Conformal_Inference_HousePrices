install.packages("pacman")
pacman::p_load("AmesHousing", "tidyverse", "skimr")

df <- ames_raw

df %>% skim()


# Drop variable if missing more than 70% of observations
missing_lots <- names(which(colMeans(is.na(df)) > 0.7))

df <- df %>% select(-missing_lots)

# Drop observations with missing values
df <- df %>% drop_na()

df %>% skim()
