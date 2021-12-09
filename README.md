# DataCo-Supply-Chain-Good-Delivery-Prediction

## Introduction
In today’s world, ensuring faster delivery mechanism and faster grievances redressal system are the key challenges for the product-based companies. Thus, the main objective of this project is to build a model to identify whether a delivery of an order will be late or on-time.<br/>
Identification of the late delivery patterns can provide a tremendous boost to the overall revenue of the company and can provide an edge against the competitors. The key benefits of addressing the issues/risks related on-time delivery mechanism are as follows:<br/>
1.	Higher Customer Retention Rate (CRR) – CRR is the rate which measures the percentage of the people stayed or remained loyal to the company’s product for a specific period of time. On-time or early deliveries can strengthen the company’s relationship with the customers and thus lead to higher CRR. <br/>
2.	Reduces Customer Acquisition Cost – CAC is the cost incurred by the company for acquiring one new customer. Reducing the late delivery will build loyalty among the customers and we know cost of acquiring a new customer is way more than retaining an existing customer. Thus, on-time deliveries can help in reducing the CAC.<br/>
3.	Rise in Customer Lifetime Value - CLV is a metric which measures value/revenue generated from a single customer during the loyalty period. Higher customer retention rate due to smooth delivery mechanism will eventually lead to higher CLV.

## Environment set up and Data Import
The data used for analysis in this project belongs to “DataCo Global” company. The data is from 2015 to 2019 and contains 180519 records and 53 variables. Dependent variable for this dataset is Late Delivery risk. The target variable reflects that around 45.17% are on-time deliveries while 54.83% are late deliveries. <br/>
Setting up of the working directory help in accessing the dataset easily. Different packages like “tidyverse”, “car”, “InformationValue”, “bloom”, “caTools”, “caret”, “dplyr”, “gini”, “ROCR”, “gridExtra” and “corrplot” are installed to make the analysis process slightly easier. The data file is in “.csv” format. To import the dataset, read.csv function is used.

## Variable Identification
“head” function is used to show top six rows of the dataset.<br/>
“tail” function is used to show last six rows of the dataset.<br/>
“str” function is used to identify type of the variables. This data set contains mixed type variables like numerical, categorical, date, geo-spatial, etc.<br/>
“summary” function is used to show the descriptive statistics of all the variables<br/>

![image](https://user-images.githubusercontent.com/61781289/145459311-76546eba-3892-4c46-8e36-ec5703e74a48.png)

## Exploratory Data Analysis
### Univariate Analysis
#### A)	Histograms
The histogram is used to signify the normality and skewness of the data. It also signifies central location, shape and size of the data.

![image](https://user-images.githubusercontent.com/61781289/145459377-136b0667-8e76-41e9-abcc-1d4fd102da83.png)

From above figure we can infer that:
1.	Variables like Sales per customer and Order Item Product Price shows sign of right skewness.
2.	Variables like Order Item Profit Ratio shows sign of left skewness.
3.	Benefit per order is almost normally distributed with having higher outliers on left side of histogram.

![image](https://user-images.githubusercontent.com/61781289/145459416-4986776f-8ebd-4e21-95ab-45232db151cf.png)


From above figure we can infer that:
1.	Variable like Order Item Discount shows sign of right skewness.
2.	All other variables have discreet values. Thus, they doesn’t represent any distribution


![image](https://user-images.githubusercontent.com/61781289/145459448-c30e1b8a-b935-4bb0-80e3-b93aa7e19843.png)


From above figure we can infer that:
1.	Variables like Sales, Order Item Total and Product Price shows sign of right skewness.
2.	Order Profit per order is almost normally distributed with having higher outliers on left side of histogram.

#### B)	Barplots

![image](https://user-images.githubusercontent.com/61781289/145459516-6ab87423-f3fc-45d1-ae07-bc72d73cd4e3.png)

From above figure we can infer that:
1)	The first graph shows that the highest orders are from the United States but percentage of late delivery is around equal in both countries.
2)	The city with highest sales is Caguas.
3)	Out of top 10 states, Puerto Rico, California and New York contributes 74.8% to the sales of the company.


![image](https://user-images.githubusercontent.com/61781289/145459561-b9385a75-e5a5-4d84-a865-3eb00418f78c.png)


From above figure we can infer that:
1)	Departments with the highest sales are Fan shop, Apparel, Golf and so on.
2)	Top 3 product sold are Field & Stream Sportsman 16 Gun Fire Safe, Perfect Fitness Perfect Rip Deck and Diamondback Women's Serene Classic Comfort Bi.
3)	The categories with the highest sales are fishing, cleats, camping & hiking and so on.
4)	The most preferable mode of payment is Debit.


![image](https://user-images.githubusercontent.com/61781289/145459583-b101e079-eff3-4aa2-b52e-9ce3e4adb85e.png)

From above figure we can infer that:
1)	The most preferable shipment mode is standard class and first class has the highest rate of delivery risk.
2)	Company has highest sales in the European countries and lowest in the African countries.


### Bivariate Analysis
#### A)	Boxplot

![image](https://user-images.githubusercontent.com/61781289/145459657-a3c2fb38-50b6-441c-ab99-63297ac0c86f.png)

![image](https://user-images.githubusercontent.com/61781289/145459673-a07dd4be-c1ea-4c7d-aa77-7174a23b7485.png)

It can be seen from the figures that only variables like Order Item Discount Rate and Days of shipment does not show signs of outliers.

### Multicollinearity
The problem of multicollinearity exists when the independent variables are highly correlated with each other.  Variance Inflation Factor (VIF) and correlation matrix are the key factors to analyse the level of the multicollinearity among independent variables.  In VIF, any value closer to 1 signifies low level of correlation and any value above 10 signifies high level of correlation.

![image](https://user-images.githubusercontent.com/61781289/145459747-8a77ddef-0bec-4f1b-93d2-326379cb623e.png)

It can be seen in the above figure that: 
1.	There exists high correlation between Sales, Sales per customer and Order Item Total.
2.	Order Profit per Order and Benefit per order also shows signs of high correlation (0.99).
3.	Product Price and Order Item Product Price shows signs of high correlation (1).
4.	There exists high correlation (0.82) between Order Item Profit Ratio, Order Profit per Order and Benefits per order.


### Removal of unwanted variables
Variable like days for shipping real, Delivery Status, Category Id, Customer Email, Customer Fname, Customer Id, Customer Lname, Customer Password, Customer Street, Customer Zip code, Department Id, Order Customer Id, Latitiude and Longitude, Order Id, Order Item Cardprod Id, Order Item Id, Order Status, Product Card Id, Product Category Id, Product Image have no significance in the analysis and thus removed. <br/>
Product Status variables is also removed from the dataset as it only contains “0” value and will not have any significant effect on target variable. Records, where Customer States are 91732 and 95758, are also removed from the data set.

### Missing values and Outlier Detection      
Variables like Product Description and Order Zip code contains more than 75% null values and thus, removed from the dataset. Apart from this, there is no missing values in the dataset.

### Outliers Treatment
Presence of outliers can significantly influence the result of various machine learning algorithms. Thus, treatment of such outliers is of utmost importance. There are various methods available to treat the outliers like capping method, transformations, etc.</br>
There were some non-recurring outlier values in the variable Benefit per order. All the records with values less than -2500 are removed from the dataset.
To handle other outliers, feature transformation techniques are used. Log transformation is used in case of variables like Benefit per order, Sales per customer, Order Item Discount and Order Item Quantity.<br/>

![image](https://user-images.githubusercontent.com/61781289/145459878-77783bcb-54f6-43b4-9725-c23c16dc33dd.png)

![image](https://user-images.githubusercontent.com/61781289/145459888-1106e573-8ee4-47b8-89fd-a455aa7bc587.png)

From the above figure it is quite clear that log transformation helped in handling the outlier effect by providing almost normal distribution for each of the variables.

## New variables formed 
1.	Distance: - This variable shows the distance between the location of warehouse where the product is stored and location where it needs to be delivered. The “ggmap” libraray is used to generate the latitude and longitude corresponding to location where order needs to be delivered. 
2.	Weekday of order and shipment: - Weekday variable is extracted from the order dates and shipment dates to study its impact on the dependent variable.
3.	Time of order and shipment: - These variables are extracted for optimal analysis of the on-time delivery mechanism.


### Insights from Exploratory Data Analysis
Since the data is almost equally distributed (45.17% are on-time deliveries while 54.83% are late deliveries), there is no need for balancing the data. In case, the target variable was in the ratio 95:5, then we have to proceed with the SMOTE analysis first. </br>
Clustering algorithms like K-means or hierarchical clustering is useful in case of unsupervised dataset. In this project, clustering is not required.

### Analysis of significance of categorical variables using Chi-Square Test
A chi-square test for independence compares two variables in a contingency table to see if they are related or not. The two hypotheses for chi-square test are as follows:<br/>
H0: there is no association between the two variables. <br/>
H1: there is an association between the two variables <br/>
The significance level for the test is 0.05. The result of all the variables are mentioned below:<br/>

![image](https://user-images.githubusercontent.com/61781289/145460050-64aa75c8-8885-447d-b5d9-61d79cd673f3.png)


## PRINCIPAL COMPONENT ANALYSIS
Principal component analysis is a dimension reduction technique in which factors having high correlation between them are dropped into single component. This technique is generally used when there exists the problem of multicollinearity. Before proceeding with principal component analysis, we need to perform two tests - Bartlett's sphericity test and KMO (Kaiser – Meyer – Olkin) test.

### Bartlett's sphericity test
Bartlett’s sphericity test is performed to check whether principal component analysis and factor analysis can be performed or not. The two hypotheses are as follows: <br/>
H0: The correlation matrix is an identity matrix. i.e. PCA can’t be performed.<br/>
H1: The correlation matrix is not an identity matrix. i.e. PCA can be performed. <br/>
After applying the Bartlett's sphericity test, it was found that p-value (< 2.22e-16) is much less than the alpha or significance level (5%). <br/>
So, we can reject the null hypothesis and conclude that PCA Analysis can be performed. <br/>

### KMO (Kaiser – Meyer – Olkin) test
The KMO test is performed to check the sampling adequacy of the dataset. Any value greater than or equal to 0.5 is an acceptable criterion to proceed with the factor analysis or principal component analysis. <br/>
After applying KMO test on the given dataset, the overall MSA value = 0.50 which satisfy the acceptable criterion. <br/>

### Factor Selection
After passing both the test with flying colours, next step is to determine the number of factors to be selected for principal component analysis. There are mainly two ways:
1)Kaiser-Guttman normalization rule
2)Elbow Method

### Kaiser-Guttman normalization rule for factor selection
The rule states that factors or principal components with eigen value more than 1 should be selected for the principal component analysis and factor analysis. For this particular problem statement, there exist 10 principal components whose eigen value is more than 1.

![image](https://user-images.githubusercontent.com/61781289/145460283-45a4f727-65bb-45a9-bfe8-59c0af2ffe16.png)

We will be using top 6 components in terms of eigen value. The information is given as under:
|Principal components	|Product |Price	|Profit	|Shipment Info|	Discount|	Quantity	|Location| 
|----|------|------|------|------|------|------|------|
|Eigen Values|	4.738|	2.778|	1.919|	1.605|	1.491|	1.491|

### Identifying variables under each principal component

After determining the number of factors, principal component analysis is performed under orthogonal rotation(varimax). Varimax rotation means that loadings below 0.5 will be pushed towards to 0 and loadings value above 0.5 will pushed towards 1. This will help to improve the interpretability of the data. 

So, the loading after principal component analysis are as follows:

![image](https://user-images.githubusercontent.com/61781289/145460594-456d2226-19f7-413f-bdc4-7fc612fd836a.png)

It can be seen that six  principal components are able to explain 61.4% cumulative variation in the data. The first component (RC1) explains around 19% variation in data, followed by RC2 with 11.1%, RC3 with 7.7%, RC4 with 6.4%, RC5 with 5.7% and RC6 with 5.6%.
Type of variables under each principal component is as under:

|Variables	|Principal Components|
|------|------|
|Sales per customer, Order Item Product Price, Sales, Order Item Total, Product Price	|Product Price|
|Benefit per order, Order Item Price Ratio, Order Profit per Order	|Profit|
|Days of shipment schedule, Shipping Mode	|Shipment Info|
|Order Item Discount, Order Item Discount Rate	|Discount|
|Order Item Quantity|	Quantity|
|Customer City, Customer State|	Location|

## Logistic Regression 
Logistic Regression is a type of generalized linear model which is used to solve binary classification problem. Logistic regression uses maximum likelihood method to obtain a best fit line. It uses logit function to estimate the coefficients. The function is given by <br/>
log(p/1-p) = beta0 + beta1*X1 +beta2*X2 + ……… + beta(n)* Xn

#### Assumptions:
1.	Linear Relationship: There should exist a linear relationship between log(odds) and regressors. It can be seen in the figure that most of the variables depicts this linear relationship.
2.	Outliers Treatment: The dataset should be free from outliers. Using cook’s distance, it was found that there is presence of outliers in the model and mean imputation method is used to remove these outliers.
3.	No Multicollinearity: The independent variables should not be highly correlated with each other. 

### Model Building 
The data has been split randomly into training set and test set with a split ratio of 2/3. K-fold validation technique was used to build the model on training data set and the model which best fits the training set (i.e. model with highest accuracy) was selected. Then the best fit model is used to predict the test dataset. 


|Variables	|Estimate	|z-value	|p-value	|Status|
|-----|-----|----|----|----|
|Intercept	|0.2862222	|46.452	|< 2e-16	|Significant|
|Product.Price	|0.0009721	|0.670	|0.5026	|Not-Significant|
|Profit        |	-0.0043577	-|2.013	|0.0441|	Significant|
|Shipment.info|	-0.4918679	|-133.595|  	< 2e-16|	Significant|
|Discount	|0.0010410	|0.271	|0.7865	|Not -Significant|
|Quantity |     	-0.0080111|  	-1.962|	0.0498|	Significant|
|Location|	-0.0204016|	-4.845|	0.00000126	|Significant|



After removing insignificant variables, we get, <br/>
Late Delivery = 0.286 - 0.004*Profit -0.49*Shipment – 0.009*Quantity – 0.021* Location <br/>
From the analysis, we can infer that variables like Product Price and Discount doesn’t has any significant effect on the Late delivery. Shipment info has the highest effect on the late delivery variable followed by Location, Profit and Quantity.

## Random Forest

Random forest is an ensemble machine learning technique used to solve regression and classification problems. Random forest is an improvement over the cart algorithm analysis because instead of building a single tree from the dataset, the dataset set is divided into smaller subsets and then multiple trees are formed. The average value of all the trees in taken as the final value. In case of regression, we use mean of regression trees and in case of classification we use mode of regression trees. <br/>

##### Steps involved in a Random Forest:
1)	**Bagging/Bootstrap aggregating**: It is a method of taking out n number of samples from the dataset by sampling with replacement. 
2)	Selecting the n independent variables out of the total variables(N) so as to avoid problem of multicollinearity among the variables.
3)	Building decision tree model on each n samples and using cross- validation to obtain the required results.
4)	Take average of the all n results obtained from n decision trees.

#### Hyperparameter Tuning

The data has been split randomly into training set and test set with a split ratio of 0.75. K-fold validation technique was used to build the model on training data set and the model which best fits the training set (i.e. model with highest accuracy) was selected. Then the best fit model is used to predict the test dataset.  <br/>

The randomized search method is used to tune the hyperparameter which are given under:
1)	n_estimators: –This parameter signifies the total number of trees to be drawn in the algorithm. The final optimal value selected is 100.
2)	max_depth: - This parameter signifies max depth that a tree is allowed to grow. The final value selected is 10.
3)	max_features: - Number of variables randomly sampled as candidates at each split. The final value will be sqrt(n).
4)	min_samples_split: - It refers to the number of minimum observations required at terminal node to make the next split. The final value selected is 10.
5)	Min_samples_leaf - It refers to the number of minimum observations required at leaf node to be considered as leaf. The final value selected is 2.


## Boosting
Boosting is type of ensemble technique which tries to reduce both bias and variance by building large number of weak learners/models where each successive weak learner learns from the mistakes of predecessor learners. It can be used for both classification and regression problems. <br/>
Extreme Gradient boosting method is used to train the model. Extreme Gradient boosting uses the concept of Gini Impurity(to make a split) and Pseudo Residuals(= (original – predicted)^2) to come up with the final answer for a tree. In case of regression task, average of all models’ measure and in case of classification task, mode of all models’ measure is chosen as the final answer. <br/>


### Hyperparameter Tuning
The data has been split randomly into training set and test set with a split ratio of 0.70. K-fold validation technique was used to build the model on training data set and the model which best fits the training set (i.e. model with highest accuracy) was selected. Then the best fit model is used to predict the test dataset. <br/>

The grid search method is used to tune the hyperparameter which are given under: <br/>
1)	n_estimators: –This parameter signifies the total number of trees to be drawn in the extreme GBM algorithm. The final optimal value selected is 140.
2)	max_depth: - This parameter signifies max depth that a tree is allowed to grow. The final value selected is 5			.
3)	learning_rate: - It is known as learning rate which is applied to each tree in expansion. The final value selected is 0.1.
4)	colsample_bynode: - The final value selected is 1.
5)	Colsample_bytree: - The final value selected is 0.8.


### Model Performance
Once the model is prepared on the training dataset, next step to is to measure the performance of the model on test dataset. Since the models predicts the test values in the form of probability, a threshold is selected to convert it to either 0 or 1. In these model, 0.5 is selected as threshold. Any probability less than 0.5 will be shifted to 0 and any probability above 0.5 will be shifted to 1. <br/>

Different key performance measures are used to check the efficiency and effectiveness of the model.
1)	Accuracy: - It is the ratio of total number of correct predictions to total number of samples. 
Accuracy = (True Positive + False Negative)/ (True Positive + False Negative + True Negative + False Positive)
2)	Classification Error: - It is the ratio of total number of incorrect predictions to total number of samples.
Classification Error = (False Positive + True Negative)/ (True Positive + False Negative + True Negative + False Positive)
3)	Sensitivity: - It is the proportion of customers who didn’t cancel the post-paid services got predicted correctly to total number of customers who didn’t cancel the services.
Sensitivity = True Positive/ (True Positive + False Negative)
4)	Specificity: - It is the proportion of customers who cancelled the post-paid services got correctly predicted to total number of customers who cancelled the post-paid services.
Sensitivity = True Negative/ (True Negative + False Positive)
5)	Area Under Curve/ Receiver Operating Characteristics (AUC/ROC): - It signifies the degree of correct predictions made by the model. Higher the AUC, better the model. 
<br/>

The results drawn from performance measure are as follows

|Performance Measures|	Logistic|	Random Forest|	Xgboost|
|----|----|----|-----|
|Accuracy|   	0.69	|0.69	|0.70|
|Classification error|	0.3095	|0.31	|0.30|
|Sensitivity|	0.8152	|0.801	0.83|
|Specificity|	0.5878|	0.619	|0.617|
|AUC|	0.7069	|0.7034	|0.711|

We can draw following conclusion from the above table: 
1)	All the models perform equally good in term of accuracy. 
2)	In terms of AUC score, xgboost perform better than other models.
3)	Xgboost outperform all other models in terms of Sensitivity. Thus, we can say that Xgboost model is the best model out of three models with accuracy of 90%.


## Recommendations
1.	Mode of shipment affects the delivery risk by 30 – 40%. Highest late deliveries are made when the mode of payment is first class. Proper inspection of items sent using first class mode should be done. Company can increase the number of standard delivery.
2.	Location of the customer has an effect of 3% on the delivery risk. New fulfilment company can be enlisted to look after long-distance location like Pacific Asia. New warehouse can be opened at the delivery location so that exported products can be stored in abundance and risk can be reduced.
3.	Integration of ERP with supply chain can also be done. Enterprise resource planning is the integrated management of main business processes, often in real time and mediated by software and technology. It will help in monitoring of the supply chain and continuous enhance the model.
4.	Inventory for items under high profit categories like fishing, cleats and men’s apparels should be managed regularly.



