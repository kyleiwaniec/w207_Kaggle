library(LiblineaR)
library(readr)

setwd("/Users/koza/Documents/UCBerkeley/207/project/w207_Kaggle")
#Prepare training set
data <- read.csv("data/train.csv")

## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]

test_target <- test$Category

target <- train$Category
#One=hot encoded hour of the crime
train$Hour <- ordered(format(strptime(train$Dates, format = "%Y-%m-%d %H:%M:%S"), "%H"), labels = paste0("H", 0:23))
h <- as.data.frame(model.matrix(~train$Hour - 1))
names(h) <- levels(train$Hour)
#One-hot encoded day of week
dow <- as.data.frame(model.matrix(~train$DayOfWeek - 1))
names(dow) <- levels(train$DayOfWeek)
#One-hot district
district <- as.data.frame(model.matrix(~train$PdDistrict - 1))
names(district) <- levels(train$PdDistrict)

#Our simple training set
train <- data.frame(dow, h, district)
head(train)
#Cleanup, like a good kaggler should
rm(h)
rm(dow)
rm(district)
gc()

#Build a linear model - only use part of train set for now, otherwise we run out of time :(
model <- LiblineaR(train[1:500000, ], target[1:500000], type = 7, verbose = FALSE)
rm(train)
gc()

# acc=LiblineaR(data=s,labels=yTrain,type=ty,cost=co,bias=TRUE,cross=10,verbose=FALSE)

#Prepare test set. Same code as for building training set. Lazy.
#test <- read.csv("data/test.csv")
Id <- test$Id
#One=hot encoded hour of the crime
test$Hour <- ordered(format(strptime(test$Dates, format = "%Y-%m-%d %H:%M:%S"), "%H"), labels = paste0("H", 0:23))
h <- as.data.frame(model.matrix(~test$Hour - 1))
names(h) <- levels(test$Hour)
#One-hot encoded day of week
dow <- as.data.frame(model.matrix(~test$DayOfWeek - 1))
names(dow) <- levels(test$DayOfWeek)
#One-hot district
district <- as.data.frame(model.matrix(~test$PdDistrict - 1))
names(district) <- levels(test$PdDistrict)
test <- data.frame(dow, h, district)


#Cleanup, like a good kaggler should
rm(h)
rm(dow)
rm(district)
gc()

#Build submission. Note : sorting output of predict function by levels of the original
#target. For some reason LiblineaR orders them differently(todo)
submit <- data.frame(predict(model, test, proba = TRUE)$probabilities[, levels(target)])
names(submit) <- levels(target)
head(submit)
predictions = colnames(submit)[max.col(submit,ties.method="first")]
bools = predictions == test_target
sum(bools)/length(bools)





# shrink the size of the submission
submit <- format(submit, digits=4, scientific = FALSE)
#Add the Id, in front
submit <- cbind(Id = Id, submit)



#Thanks peterCooman(https://www.kaggle.com/petercooman)
#At time of writing the data loaded by this script still contains
#commas in the names of the crimes(ie not updated yet). 
#comment out, or not, makes no difference of course
# work-around for removing commas from the column names of submission
names(submit) <- gsub(",", "", names(submit), fixed = TRUE)

gz_out <- gzfile("submit.csv.gz", "w")
writeChar(format_csv(submit, ""), gz_out, eos=NULL)
close(gz_out)
