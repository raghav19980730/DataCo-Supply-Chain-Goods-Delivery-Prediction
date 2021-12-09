setwd("C:/Users/ragha/Desktop/Raghav/Great Learning/Capstone Projects/DataCo. Supply Chain")
library(class)
library(lubridate)
library(dplyr)
library(tidyverse)
library(corrplot)
library(gridExtra)
library(ggmap)
library(geosphere)
library(caTools)
library(psych)
library(CatEncoders)
library(REdaS)
library(factoextra)
library(ClustOfVar)
library(FactoMineR)
library(xgboost)
library(caret)
library(ROCR)

data = read.csv('DataCoSupplyChainDataset.csv')
data = data[!(data$Customer.State == 91732 | data$Customer.State == 95758),]
data = data[data$Benefit.per.order > -2500,]
data$order.date..DateOrders. = as.POSIXct(data$order.date..DateOrders.,format = c("%d/%m/%Y"))
data$shipping.date..DateOrders. = as.POSIXct(data$shipping.date..DateOrders.,format = c("%d/%m/%Y"))
order_long=data$Longitude
order_lat=data$Latitude
Target = data$Late_delivery_risk
data$Late_delivery_risk = NULL

dim(data)

head(data)
tail(data)
str(data)
summary(data)

mydf = Filter(is.numeric,data)
facdf=Filter(is.factor,data)
num_df = data.frame(unclass(summary(mydf)), check.names = FALSE, stringsAsFactors = FALSE)
fac_df = data.frame(unclass(summary(facdf)), check.names = FALSE, stringsAsFactors = FALSE)
write.csv(num_df,"Summary.csv")
write.csv(fac_df,"Summary1.csv")

#Proportion in target variables
prop.table(table(Target))*100


unwanted_col = c("Days.for.shipping..real.","Delivery.Status","Category.Id",
                 "Customer.Email","Customer.Fname","Customer.Id","Customer.Lname",
                 "Customer.Password","Customer.Street","Customer.Zipcode","Department.Id",
                 "Order.Customer.Id","Order.Id","Order.Item.Cardprod.Id","Order.Item.Id",
                 "Order.Status","Product.Card.Id","Product.Category.Id","Product.Image")


#Removing unwanted variables
for(i in unwanted_col){
  data[i] = NULL
}

#Removing column with low variance
data$Product.Status = NULL

#Missing values treatment
#Removing columns having more than 80% null values
any(is.na(data))

null_col = c("Product.Description","Order.Zipcode")
for(i in null_col){
  data[i] = NULL
}



#Histogram
options(scipen = 5)
par(mfrow = c(2,2))
for(i in names(Filter(is.numeric,data))[c(2,3,6,7)]){
  hist(data[,i], xlab = names(data[i]), col = "red", border = "black", ylab = "Frequency",
       main =paste("Histogram of", names(data[i])),col.main = "darkGreen")
}
par(mfrow = c(1,1))

par(mfrow = c(2,2))
for(i in names(Filter(is.numeric,data))[c(1,4,5,8)]){
  hist(data[,i], xlab = names(data[i]), col = "red", border = "black", ylab = "Frequency",
       main =paste("Histogram of", names(data[i])),col.main = "darkGreen")
}
par(mfrow = c(1,1))


par(mfrow = c(2,2))
for(i in names(Filter(is.numeric,data))[9:12]){
  hist(data[,i], xlab = names(data[i]), col = "red", border = "black", ylab = "Frequency",
       main =paste("Histogram of", names(data[i])),col.main = "darkGreen")
}
par(mfrow = c(1,1))



#Bivariate Analysisw
#Boxplot
A <- ggplot(data = data, aes(as.factor(Target),Days.for.shipment..scheduled.)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Days of Shipment vs Late Delivery Risk")
B <- ggplot(data = data, aes(as.factor(Target),Benefit.per.order)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Benefit per order vs Late Delivery Risk")
C <- ggplot(data = data, aes(as.factor(Target),Sales.per.customer)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Sales per customer vs Late Delivery Risk")
D <- ggplot(data = data, aes(as.factor(Target),Order.Item.Discount)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Order Item Discount vs Late Delivery Risk")
E <- ggplot(data = data, aes(as.factor(Target),Order.Item.Discount.Rate)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Order Item Discount Rate vs Late Delivery Risk")
G <- ggplot(data = data, aes(as.factor(Target),Order.Item.Product.Price)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Order Item Product Price vs Late Delivery Risk")
H <- ggplot(data = data, aes(as.factor(Target),Order.Item.Profit.Ratio)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Order Item Profit Ratio vs Late Delivery Risk")
I <- ggplot(data = data, aes(as.factor(Target),Sales)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Sales vs Late Delivery Risk")
J <- ggplot(data = data, aes(as.factor(Target),Order.Item.Quantity)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Order Item Quantity vs Late Delivery Risk")
K <- ggplot(data = data, aes(as.factor(Target),Order.Item.Total)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("order Item Total vs Late Delivery Risk")
L <- ggplot(data = data, aes(as.factor(Target),Order.Profit.Per.Order)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Order Profit per Order vs Late Delivery Risk")
M <- ggplot(data = data, aes(as.factor(Target),Product.Price)) + geom_boxplot(color = c("Red","Blue")) + xlab("Late Delivery Risk") + ggtitle("Product Price vs Late Delivery Risk")


grid.arrange(A,B,C,D)
grid.arrange(E,G,H,I)
grid.arrange(J,K,L,M)
par(mfrow =c(1,1))


#Correlation 
corr = cor(Filter(is.numeric,data))
corrplot.mixed(corr, tl.pos = "lt", diag = "l",lower = "number", upper = "pie")



#Outliers treatment
mydata <- Filter(is.numeric,data)

Outliers<- mydata[1:60,]

for(i in c(1:4)) {
  Outliers[,i] <- NA
  Box <-boxplot(mydata[,i],plot =F)$out
  if (length(Box)>0){
    Outliers[1:length(Box),i] <- Box
  }
}

###Treatment of Outliers
#Removing record where benefit per order is less than -2500
data = data[data$Benefit.per.order > -2500,]


#Transformation of continuous variables

par(mfrow = c(2,2))
for(i in names(Filter(is.numeric,data))[c(2,3)]){
  hist(data[,i],xlab = i, col = "red", border = "black", ylab = "Frequency",
       main =paste("Histogram of", i),col.main = "darkGreen")
  hist(log(data[,i]),xlab = i, col = "red", border = "black", ylab = "Frequency",
       main =paste("Log transformed histogram of ", i),col.main = "darkGreen")
}
par(mfrow= c(1,1))

par(mfrow = c(2,2))
for(i in names(Filter(is.numeric,data))[c(4,8)]){
  hist(data[,i],xlab = i, col = "red", border = "black", ylab = "Frequency",
       main =paste("Histogram of", i),col.main = "darkGreen")
  hist(log(data[,i]),xlab = i, col = "red", border = "black", ylab = "Frequency",
       main =paste("Log transformed histogram of ", i),col.main = "darkGreen")
}
par(mfrow= c(1,1))




#New variables formation

#Extracting days, week days and hours variables
data$shipping_day = day(data$shipping.date..DateOrders.)
data$ordering_day = day(data$order.date..DateOrders.)
data$shipping_weekday = weekdays(data$shipping.date..DateOrders.)
data$ordering_weekday = weekdays(data$order.date..DateOrders.)


#Chi-square test
names= c()
chi_value = c()
for(i in names(Filter(is.factor,data))){
  options(scipen = 9)
  names = append(names,i)
  a = chisq.test(data[,i],Target)
  chi_value = append(chi_value,round(a$p.value,15))
}
write.csv(data.frame(names,chi_value),"chi_values.csv")

insign_var = c('shipping.date..DateOrders.','order.date..DateOrders.','Latitude','Longitude',
               'Category.Name','Customer.Country','Customer.Segment','Department.Name','Product.Name')

for(i in insign_var){
  data[i] = NULL
}


data = read.csv('Before_pca_data.csv')

#Encoding the data
for(i in names(Filter(is.factor,data))){
a = LabelEncoder.fit(data[,i])
data[,i] = transform(a,data[,i])
}

data = scale(data)
#Kaiser rule application/ Scree Plot
options(scipen = 5)
ev <- eigen(cor(data))
ev
Scree <- data.frame(1:25, ev$values)
par(mfrow =c(1,1))
plot(Scree, main = "Scree Plot", ylab = "eigen value", xlab = "Factors")
lines(Scree, col ="blue")
abline(h = 1, lty = "dotted", col = "red", lwd = 1.5)


# Test to check the applicability of the principal componenet analysis
#Bartlett Sphercity Test
bart_spher(data)


#Sampling adequacy to perform factor analysis
#Kaiser - Meyes -Olkin Test
KMO(cor(data))


#Principal component analysis
pca<- principal(data,nfactors = 7,rotate = "varimax")
print(pca$loadings, cutoff = 0.8)

final_data = data.frame(Target,pca$scores[,-7])
colnames(final_data) = c('Target','Product.Price','Profit','Shipment.info','Discount','Quantity','Location.info')

final = read.csv('Final_data.csv')

#Train and Test data
sample = sample.split(final$Target,SplitRatio = 0.75)
train = final[sample == T,]
test = final[sample==F,]

model = glm(Target~., data = train, family = 'binomial')
summary(model)
model = glm(Target~.-Product.Price,data = train, family = 'binomial')
summary(model)
final_model = glm(Target~.,data = train[-c(2,5)], family = 'binomial')
summary(final_model)

#Predicition
pred = predict(final_model,test[,-c(1,2,5)],type = 'response')
pred = ifelse(pred>0.5,1,0)
confusionMatrix(data = as.factor(pred), reference =as.factor(test$Target))

#ROCR
y_pred <- prediction(test$Target,pred)
auc <- performance(y_pred,"auc")
auc <- auc@y.values[[1]]
auc <- auc[[1]]
auc


