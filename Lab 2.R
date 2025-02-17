## Lab 2 - Movie Reviews

## Install Packages
install.packages("dplyr")
install.packages("tm")
install.packages("SnowballC")
install.packages("gmodels")
install.packages("caret")
install.packages("naivebayes")
library(dplyr)
library(tm)
library(SnowballC)
library(gmodels)
library(caret)
library(naivebayes)


## Part 1 - Data Exploration
imdb <- read.csv("imdb.csv")
imdb %>% count(label)
  ## 25,000 negative
  ## 25,000 positive


## Part 2 - Data Preparation
random_sample <- sample(50000, 2000)
imdb <- imdb[random_sample,]

prop.table(table(imdb$label))
  ## negative - 0.5
  ## positive - 0.5

imdb$label <- factor(imdb$label, levels = c("0","1"), labels = c("negative", "positive"))

  ## Creating Corpus
imdb_corpus <- VCorpus(VectorSource(imdb$text))
print(imdb_corpus)
inspect(imdb_corpus)

lapply(imdb_corpus[1:3], as.character)

  ## Standardizing Text
imdb_clean <- tm_map(imdb_corpus, content_transformer(tolower))
as.character(imdb_clean[[1]])

    ## Removing Numbers
imdb_clean <- tm_map(imdb_clean, removeNumbers)
as.character(imdb_clean[[1]])

    ## Remove Filler Words
imdb_clean <- tm_map(imdb_clean, removeWords, stopwords())
as.character(imdb_clean[[1]])

   ## Remove Punctuation
imdb_clean <-  tm_map(imdb_clean, removePunctuation)
as.character(imdb_clean[[1]])

   ## Simplify Stemming Words
imdb_clean <- tm_map(imdb_clean, stemDocument)
as.character(imdb_clean[[1]])

   ## Remove white space
imdb_clean <- tm_map(imdb_clean, stripWhitespace)
as.character(imdb_clean[[1]])

    ## Compare original and cleaned dataset
lapply(imdb_corpus[1:3], as.character)
lapply(imdb_clean[1:3], as.character)


  ## Creating Matrix
imdb_dtm <- DocumentTermMatrix(imdb_clean)
imdb_dtm

## Part 3 - Training and Test Data
set.seed(4836)
train_size <- floor(0.75* 2000)
train_size
train_sample <- sample(2000,1500)

imdb_dtm_train <- imdb_dtm[train_sample,]
imdb_dtm_test <- imdb_dtm[-train_sample,]

imdb_train_labels <- imdb[train_sample,]$label
imdb_test_labels <- imdb[-train_sample,]$label

prop.table(table(imdb_train_labels))
  ## negative - 0.4946
  ## positive - 0.5053
prop.table(table(imdb_test_labels))
  ## negative - 0.516
  ## positive - 0.484

## Part 4 - Build the Naive Bayes Algorithm 
imdb_freq_words <- findFreqTerms(imdb_dtm_train, 5)
str(imdb_freq_words)

imdb_dtm_freq_train <- imdb_dtm_train[, imdb_freq_words]
imdb_dtm_freq_test <- imdb_dtm_test[, imdb_freq_words]

imdb_dtm_freq_train
imdb_dtm_freq_test

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes","No")
}

imdb_train <- apply(imdb_dtm_freq_train, MARGIN = 2, convert_counts)
imdb_test <- apply(imdb_dtm_freq_test, MARGIN = 2, convert_counts)

imdb_classifier <- naive_bayes(imdb_train, imdb_train_labels)

imdb_test_pred <- predict(imdb_classifier, imdb_test)

CrossTable(imdb_test_pred, imdb_test_labels,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted','actual'))

  ## Predicting probability 
imdb_test_prob <- predict(imdb_classifier, imdb_test, type = "prob")

hist(imdb_test_prob)

## Part 5 - Improve the Model
  ## Laplace Estimator
imdb_classifier2 <- naive_bayes(imdb_train, imdb_train_labels,laplace = 1)

imdb_test_pred2 <- predict(imdb_classifier2, imdb_test)

CrossTable(imdb_test_pred2, imdb_test_labels,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted','actual'))

  ## Frequent Words
    ## Frequency = 3
imdb_freq_words2 <- findFreqTerms(imdb_dtm_train, 3)
str(imdb_freq_words2)

imdb_dtm_freq_train2 <- imdb_dtm_train[, imdb_freq_words2]
imdb_dtm_freq_test2 <- imdb_dtm_test[, imdb_freq_words2]

imdb_train2 <- apply(imdb_dtm_freq_train2, MARGIN = 2, convert_counts)
imdb_test2 <- apply(imdb_dtm_freq_test2, MARGIN = 2, convert_counts)

imdb_classifier2 <- naive_bayes(imdb_train2, imdb_train_labels)

imdb_test_pred2 <- predict(imdb_classifier2, imdb_test2)

CrossTable(imdb_test_pred2, imdb_test_labels,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted','actual'))

    ## Frequency = 10
imdb_freq_words3 <- findFreqTerms(imdb_dtm_train, 10)
str(imdb_freq_words3)

imdb_dtm_freq_train3 <- imdb_dtm_train[, imdb_freq_words3]
imdb_dtm_freq_test3 <- imdb_dtm_test[, imdb_freq_words3]

imdb_train3 <- apply(imdb_dtm_freq_train3, MARGIN = 2, convert_counts)
imdb_test3 <- apply(imdb_dtm_freq_test3, MARGIN = 2, convert_counts)

imdb_classifier3 <- naive_bayes(imdb_train3, imdb_train_labels)

imdb_test_pred3 <- predict(imdb_classifier3, imdb_test3)

CrossTable(imdb_test_pred3, imdb_test_labels,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted','actual'))
