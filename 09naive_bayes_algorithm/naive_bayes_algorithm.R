require(e1071)#Holds the Naive Bayes Classifier
Train <- read.csv(file.choose())
Test <- read.csv(file.choose())
#Make sure the target variable is of a two-class classification problem only
levels(Train$Item_Fat_Content)
model <- naiveBayes(Item_Fat_Content~., data = Train)
class(model) 
pred <- predict(model,Test)
table(pred)