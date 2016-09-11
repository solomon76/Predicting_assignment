#  Predicting Activities


### Introduction
This is a R Markdown document for the assignment titled "Predicting Activities". The document would include exploratory analysis of the data and using K-fold method for cross-validation approach.The developed model would be used to predict 20 cases from the actual test data. The seed was set as system default

### Exploratory
This portion of the code that reads on the data from csv inputs. 

```r
training<-read.csv(file="pml-training.csv")
testing<-read.csv(file="pml-testing.csv")
```
### Training

This portion of the code does the training part and some processing before the model is trained to predict different classes. There are different techniques employed to fully understand the advantage of using the Cross-validation approach to obtained the best averaged model to predicting the outcomes.

Firstly, the index for the entries are removed from both the training and testing data set. There are a number of NA entries in training data set and they were removed. 

The first model was trained with cross-validation with fold of 30. This means that the data was spitted into 30 different groups and out of each group one of the sample was used for testing. This was implemented on the training data with the random forest to compare the improvement in accuracy. It is to note that the training was implemented with parallel cluster on a multi-core machine due to the large number of predictors involved.

The predictors with NA values from the test data set were removed and the same was done for the training data-set. 



```r
train_control <- trainControl(method="cv", number=30)
training2<-na.omit(training[,c(-1)])
testing2<-testing[,c(-1)]  
#Filtering away data that is not useful,Cols with NA)
testing2_fil<-data.frame(t(na.omit(t(testing2))))
da<-c(names(testing2_fil))
trainingfi<-training2[,c(da[1:58],"classe")]
testing_final<-testing2_fil[,c(da[1:58])]
```



```r
trainmod_cv<-train(classe~.,method="rf",trControl=train_control,na.action = na.exclude,data=trainingfi)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```


```r
trainmod_rf<-train(classe~.,method="rf",na.action = na.exclude,data=trainingfi)
```

*Accuracy from using the cross-validation approach yielded accuracy of 0.8975885 vs 0.8169673 for the non cross-validation approach.*

### Analysing model
This portion of the code looks at the important predictors from the CV and RF models.

```r
trainmod_rf
```

```
## Random Forest 
## 
## 406 samples
##  58 predictor
##   5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 406, 406, 406, 406, 406, 406, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.7800548  0.7212902
##   41    0.8169673  0.7687519
##   80    0.7985573  0.7457397
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 41.
```

```r
trainmod_cv
```

```
## Random Forest 
## 
## 406 samples
##  58 predictor
##   5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (30 fold) 
## Summary of sample sizes: 394, 393, 392, 391, 391, 394, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.8486813  0.8080475
##   41    0.8975885  0.8708282
##   80    0.8749451  0.8422901
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 41.
```

```r
trainpre_rf<-predict(trainmod_rf,training2)
trainpre_cv<-predict(trainmod_cv,training2)
confusionMatrix(trainpre_cv,training2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 109   0   0   0   0
##          B   0  79   0   0   0
##          C   0   0  70   0   0
##          D   0   0   0  69   0
##          E   0   0   0   0  79
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.991, 1)
##     No Information Rate : 0.2685    
##     P-Value [Acc > NIR] : < 2.2e-16 
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000     1.00   1.0000
## Specificity            1.0000   1.0000   1.0000     1.00   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000     1.00   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000     1.00   1.0000
## Prevalence             0.2685   0.1946   0.1724     0.17   0.1946
## Detection Rate         0.2685   0.1946   0.1724     0.17   0.1946
## Detection Prevalence   0.2685   0.1946   0.1724     0.17   0.1946
## Balanced Accuracy      1.0000   1.0000   1.0000     1.00   1.0000
```

```r
confusionMatrix(trainpre_rf,training2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 109   0   0   0   0
##          B   0  79   0   0   0
##          C   0   0  70   0   0
##          D   0   0   0  69   0
##          E   0   0   0   0  79
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.991, 1)
##     No Information Rate : 0.2685    
##     P-Value [Acc > NIR] : < 2.2e-16 
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000     1.00   1.0000
## Specificity            1.0000   1.0000   1.0000     1.00   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000     1.00   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000     1.00   1.0000
## Prevalence             0.2685   0.1946   0.1724     0.17   0.1946
## Detection Rate         0.2685   0.1946   0.1724     0.17   0.1946
## Detection Prevalence   0.2685   0.1946   0.1724     0.17   0.1946
## Balanced Accuracy      1.0000   1.0000   1.0000     1.00   1.0000
```
The Confusion matrices show that the models from both the RF and 30 fold using cross validation. The results looks very good with perfect agreement for every class for the training set. 

### Predicting the classes

```r
cv_real<-predict(trainmod_rf,testing2)
cv_real
```

```
##  [1] B A B A A E D B A A B C B A E E A B A B
## Levels: A B C D E
```
