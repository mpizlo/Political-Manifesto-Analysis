
# Required packages
library(tidyverse)   
library(tm)          
library(SnowballC)    
library(tidytext)     
library(textdata)     
library(topicmodels) 
library(wordcloud)   
library(RColorBrewer) 
library(scales)
library(rstudioapi)

# Setting the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

folder_path <- "./manifestos_eng"

# reading .txt files from the folder
docs <- DirSource(folder_path, pattern = "\\.txt$")

# creating a corpus of files
corpus <- VCorpus(docs)

corpus <- tm_map(corpus, content_transformer(function(x) iconv(x, to = "UTF-8", sub = "byte")))

# function replacing unnecessary characters
toSpace <- content_transformer(function (x, pattern) gsub(pattern, " ", x))


#Removing characters, symbols, patterns
corpus <- tm_map(corpus, toSpace, "[()\\[\\]]") # square and round brackets
corpus <- tm_map(corpus, toSpace, "[@#$&*%+\\/=|]")
corpus <- tm_map(corpus, toSpace, "[\"'“”‘’„]") # quotation marks, apostrophe
corpus <- tm_map(corpus, toSpace, "[©]")
corpus <- tm_map(corpus, toSpace, "[✔✗]")
corpus <- tm_map(corpus, toSpace, "[ \t]{2,}") # tabs
corpus <- tm_map(corpus, toSpace, "\\S+\\.(pl|com)\\b") # sources with .com .pl domains
corpus <- tm_map(corpus, toSpace, "http\\w*") # http and https




# data cleaning 
corpus <- tm_map(corpus, content_transformer(tolower)) # changing to lower case
corpus <- tm_map(corpus, removeNumbers) # removing numbers
corpus <- tm_map(corpus, removeWords, stopwords("english")) # removing stop words
corpus <- tm_map(corpus, removePunctuation) # removing remaining punctuation
corpus <- tm_map(corpus, stripWhitespace) # removing additional spaces


# checking
corpus[[1]]

# stemming 

corpus_copy <- corpus 

corpus_stemmed <- tm_map(corpus, stemDocument)

corpus_stemmed[[1]]

#stem completion

#function dividing the text into words, completes stems into full words
complete_stems <- content_transformer(function(x, dict) {
  x <- unlist(strsplit(x, " "))                  
  x <- stemCompletion(x, dictionary = corpus_copy, type="longest") 
  paste(x, collapse = " ")                 
})


corpus_completed <- tm_map(corpus_stemmed, complete_stems, dict = corpus_copy)


corpus_completed <- tm_map(corpus_completed, toSpace, "NA")
corpus_completed <- tm_map(corpus_completed, stripWhitespace)


# tokenisation DTM
dtm <- DocumentTermMatrix(corpus_completed)
dtm 
inspect(dtm)
dtm_m <- as.matrix(dtm)

dtm_m[1:5, 1:5]


# counting word frequency

v <- sort(colSums(dtm_m), decreasing = TRUE)
dtm_df <- data.frame(word = names(v), freq = v)
head(dtm_df, 10) 


#word clouds for each document  
for (i in seq_along(corpus)) {
  
  # tokenisation and counting
  dtm_i    <- DocumentTermMatrix(VCorpus(VectorSource(corpus[[i]]$content)))
  dtm_m_i  <- as.matrix(dtm_i)
  word_freq<- sort(colSums(dtm_m_i), decreasing = TRUE)
  word_df  <- data.frame(word = names(word_freq), freq = word_freq)
  
  wordcloud(words    = word_df$word,
            freq     = word_df$freq,
            min.freq = 4,
            colors   = brewer.pal(9, "Spectral"))
  
  title(main = paste0("Word cloud: ", basename(docs$filelist[i])))
}

# sentiment analysis

tidy_dtm<-tidy(dtm)

#bing (positive/negative)

bing_sentiment <- tidy_dtm %>%
  inner_join(get_sentiments("bing"), by = c(term = "word"))

sentiment_review_bing <- bing_sentiment %>%
  count(document, sentiment) %>% #counting positive/negatives
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>% 
  mutate(sentiment_score = positive - negative) #overall


print(head(sentiment_review_bing))


#nrc - percentage share of emotions in texts
nrc_sentiment <- tidy_dtm %>%
  inner_join(get_sentiments("nrc"), by = c(term = "word"),relationship = "many-to-many")

nrc_sentiment_percent <- nrc_sentiment %>%
  count(document, sentiment) %>%
  group_by(document) %>%
  mutate(percent = n / sum(n)) %>% #share
  ungroup()


#afinn (sentiment scoring, negative and positive values)
afinn_sentiment <- tidy_dtm %>%
  inner_join(get_sentiments("afinn"), by = c(term = "word"))

sentiment_review_afinn <- afinn_sentiment %>%
  group_by(document) %>%
  summarise(sentiment_score = sum(value))

print(head(sentiment_review_afinn))

#graphs

# bing graph (total sentiment score, each document separately)
ggplot(sentiment_review_bing, aes(x = document, y = sentiment_score, fill = document)) +
  geom_col(width = 0.5) +
  theme_minimal() +
  labs(title = "Total sentiment in the document (BING)",
       x = "Document", y = "Total sentiment")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")


# nrc graphs
ggplot(nrc_sentiment_percent, aes(x = document, y = percent, fill = document)) +
  geom_col(width = 0.5) +
  facet_wrap(~ sentiment, scales = "free_y") +  # separate graph for each sentiment value
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  theme_minimal() +
  labs(
    title = "Percentage share of specific emotions in documents (NRC)",
    x = "Document",
    y = "Share of emotions (%)"
  ) +
  theme(
    strip.text = element_text(face = "bold", size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

# combined graph
ggplot(nrc_sentiment_percent, aes(x = document, y = percent, fill = sentiment)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Percentage share of emotions in politicians' statements (NRC)",
    x = "Politician",
    y = "Share of emotions (%)",
    fill = "Emotion"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# afinn graph
ggplot(sentiment_review_afinn, aes(x = document, y = sentiment_score, fill = document)) +
  geom_col(width = 0.5) +
  theme_minimal() +
  labs(title = "Total sentiment in the document (AFINN)", x = "Document", y = "Total sentiment")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")

# topic modelling

top_terms_by_topic_LDA <- function(dtm_input, k = 6, plot = TRUE) {
  # Removing empty rows
  unique_indexes <- unique(dtm_input$i)
  DTM <- dtm_input[unique_indexes, ]
  
  # LDA execution using the passed 'k' parameter
  lda <- LDA(DTM, k = k, control = list(seed = 1234))
  topics <- tidy(lda, matrix = "beta")
  
  # Top 10 terms per topic
  top_terms <- topics %>%
    group_by(topic) %>%
    slice_max(beta, n = 10) %>% 
    ungroup() %>%
    arrange(topic, -beta)
  
  if (plot) {
    top_terms %>%
      mutate(term = reorder_within(term, beta, topic)) %>%
      ggplot(aes(term, beta, fill = factor(topic))) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~ topic, scales = "free") +
      scale_x_reordered() +
      labs(x = "Terms", y = "β (word importance in topic)") +
      coord_flip() +
      theme_minimal() +
      scale_fill_brewer(palette = "Spectral")
  } else {
    return(top_terms)
  }
}

# Run LDA topic model for 6 topics using the DocumentTermMatrix (dtm)
top_terms_by_topic_LDA(dtm, k = 6)

