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

famous <- read.csv("/Users/ronjaronnback/Downloads/data/MindLabsFamousIndividuals.csv")

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

# Make column for number of recognized per participant
famous$RightAnwers <- as.numeric(rowSums(famous == "I got it right"))
famous$NotRecognizedAnwers <- as.numeric(rowSums(famous == "I did not recognise the person"))
famous$ForgotName <- as.numeric(rowSums(famous == "I recognised the person, but I could not remember their name"))

famous$TipOfTongue <- famous$RightAnwers + famous$NotRecognizedAnwers + famous$ForgotName
famous$TipOfTongue <- 20 - famous$TipOfTongue

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

# recode answers to:
# NR  - "I did not recognise the person"
# C   - "I got it right"                            
# TT  - "I got it wrong, but the correct name was “on the tip of my tongue”"
# RNN - "I recognised the person, but I could not remember their name"   

df[df=="I did not recognise the person"] <- "NR"
df[df=="I got it right"] <- "C"
df[df=="I got it wrong, but the correct name was “on the tip of my tongue”"] <- "TT"
df[df=="I recognised the person, but I could not remember their name"] <- "RNN"

#write.csv(df, "/Users/ronjaronnback/Documents/GitHub/BayesianFaceRecognition/fr_data.csv", row.names=FALSE)

#-------------------------------------------------------------------------------
# CELEBRITY JSON
#-------------------------------------------------------------------------------

# LINKING CELEBRITY INFO TO CELEBRITY RECOGNITION
celebrities <- rjson::fromJSON(file = "/Users/ronjaronnback/Downloads/data/Celebrities.json")
celebrities <- data.table::rbindlist(celebrities, fill = TRUE)
# make id column
celebrities$en_curid <- as.factor(celebrities$en_curid)
# clean up column names
names(celebrities)<-str_replace_all(names(celebrities), c(" " = "." , "," = "", "-" = ""))

# get top X famous people identities
X <- 50
top_X <- celebrities %>%
  #group_by(domain) %>%
  top_n(n = 50, wt = TransformedTotalPageViews)
# TAKE SUBSET OF INTEREST
#top_X <- data.frame(top_X$name, top_X$domain)

# save dfs
write.csv(top_X, "/Users/ronjaronnback/Documents/GitHub/BayesianFaceRecognition/topX_celebrities", row.names=FALSE)

#-------------------------------------------------------------------------------
# COMBINE CELEBRITY JSON WITH DATA
#-------------------------------------------------------------------------------







