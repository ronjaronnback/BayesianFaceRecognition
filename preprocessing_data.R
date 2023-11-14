# -----------------------------------------------------------------------------------------
# LIBRARIES
# -----------------------------------------------------------------------------------------
library(tidyr)
library(dplyr)
library(plyr)
library(reshape2)
library(stringr)
library(rjson)
library(jsonlite)

# -----------------------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------------------

# my R is doing weird things so set wd to whereever your files are here:
setwd("/Users/ronjaronnback/Documents/GitHub/BayesianFaceRecognition")

famous <- read.csv("data/MindLabsFamousIndividuals.csv")

# Remove some columns that aren't of interest
famous <- subset(famous, 
                 select=-c(StartDate,EndDate,Status,IPAddress,Progress, 
                           RecipientLastName,RecipientFirstName,RecipientEmail, 
                           ExternalReference,LocationLatitude,LocationLongitude, 
                           DistributionChannel,UserLanguage,Q96))

# Remove entries that aren't finalized
famous <-famous[!(famous$Finished == "False"),]
famous$Finished <- NULL

# Deal with column names that include difficult symbols
colnames(famous) <- as.character(unlist(famous[1,]))
#names(famous)<-str_replace_all(names(famous), c(" " = "." , "," = "", "-" = ""))

famous <- famous[-c(1,2), ]
colnames(famous)[5] <- "Identifier"

# Change names of columns with identical names
cols <- which(names(famous) == 'Celebrity ID:')
names(famous)[cols] <- paste0('CelebrityID', seq_along(cols))

cols <- which(names(famous) == 'Timing - Page Submit')
names(famous)[cols] <- paste0('TimePageSubmit', seq_along(cols))

cols <- which(names(famous) == 'Timing - Last Click')
names(famous)[cols] <- paste0('TimeLastClick', seq_along(cols))

cols <- which(names(famous) == 'Timing - First Click')
names(famous)[cols] <- paste0('TimeFirstClick', seq_along(cols))

cols <- which(names(famous) == 'Timing - Click Count')
names(famous)[cols] <- paste0('TimeClickCount', seq_along(cols))

cols <- which(names(famous) == 'Did you correctly recognize this person (ignoring any spelling mistakes)? The person in the picture is')
names(famous)[cols] <- paste0('Did.You.Recognize', seq_along(cols))

cols <- which(names(famous) == 'Who is this person?')
names(famous)[cols] <- paste0('Who.Is.This', seq_along(cols))

names(famous)<-str_replace_all(names(famous), c(" " = "." , "," = "", "-" = ""))

# SELECT ONLY IMPORTANT FOR BAYES
df <- famous %>% dplyr:: select(grep("CelebrityID", names(famous)),
                                #grep("TimePageSubmit", names(famous)), 
                                grep("Did.You.Recognize", names(famous)))
# add participant age (in 2018)
#df$age <- famous$`How.old.are.you?..Please.write.your.answer.below:..Text`
# df remove entries without age
#df <- df[!df$age =="",]


# recode answers to:
# NR  - "I did not recognise the person"
# C   - "I got it right"                            
# TT  - "I got it wrong, but the correct name was “on the tip of my tongue”"
# RNN - "I recognised the person, but I could not remember their name"   

df[df=="I did not recognise the person"] <- "NR"
df[df=="I got it right"] <- "C"
df[df=="I got it wrong, but the correct name was “on the tip of my tongue”"] <- "TT"
df[df=="I recognised the person, but I could not remember their name"] <- "RNN"

#write.csv(df, "data/experiment_data.csv", row.names=FALSE)

#-------------------------------------------------------------------------------
# CELEBRITY JSON
#-------------------------------------------------------------------------------

# LINKING CELEBRITY INFO TO CELEBRITY RECOGNITION
celebrities <- rjson::fromJSON(file = "data/Celebrities.json")
celebrities <- data.table::rbindlist(celebrities, fill = TRUE)
# make id column
celebrities$en_curid <- as.factor(celebrities$en_curid)
# clean up column names
names(celebrities)<-str_replace_all(names(celebrities), c(" " = "." , "," = "", "-" = ""))

# get top X famous people identities
#X <- 50
#top_X <- celebrities %>%
  #group_by(domain) %>%
#  top_n(n = 50, wt = TransformedTotalPageViews)
# TAKE SUBSET OF INTEREST
#top_X <- data.frame(top_X$name, top_X$domain)

# save dfs
#write.csv(top_X, "data/topX_celebrities", row.names=FALSE)

#-------------------------------------------------------------------------------
# COMBINE CELEBRITY JSON WITH DATA
#-------------------------------------------------------------------------------

small_celebrities <- celebrities[,c("en_curid","name","TotalPageViews")]

outcome_df <- data.frame(en_curid=integer(),
                         outcome=character(),
                         name=character(),
                         TotalPageViews=integer(),
                         participant=integer())

for(i in 1:nrow(df)) {       # for-loop over rows
  # get participant id
  participant <- as.numeric(rownames(df[i,])[1])
  # get list of celebrity column names
  celebrity_col_idx <- grep("CelebrityID", names(df))
  # get list of answer column names
  answer_col_idx <- grep("Did.You.Recognize", names(df))
  # get celebrities
  sample_celebrities <- cbind(t(df[i,celebrity_col_idx]),
                              t(df[i,answer_col_idx]))
  colnames(sample_celebrities) <- c("en_curid","outcome")
  # merge into batch
  batch <- merge(sample_celebrities,small_celebrities, by = "en_curid")
  batch$participant <- rep(participant,nrow(batch))
  # add to output dataframe
  outcome_df <- rbind(outcome_df,batch)
}

# clean up the participant column and add a trial column
outcome_df <- outcome_df %>%
  dplyr::arrange(participant) %>%
  dplyr::group_by(participant) %>%
  dplyr::mutate(participant = cur_group_id(),
         trial_number = row_number())

require(dplyr)
outcome_df %>% group_by(participant) %>% summarise(trial_number = max(trial_number))
t <- tapply(outcome_df$trial_number, outcome_df$participant, max)
idx <- which(t < 20)
# remove those participants with less than 20 trials
outcome_df <- outcome_df[!(outcome_df$participant %in% idx),]
# clean up the participant column again
outcome_df <- outcome_df %>%
  dplyr::arrange(participant) %>%
  dplyr::group_by(participant) %>%
  dplyr::mutate(participant = cur_group_id())

# save outcome df
write.csv(outcome_df, "data/outcome_data.csv", row.names=FALSE)
