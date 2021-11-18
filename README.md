# Comparing-ML-models-using-scatter-plot

## Describtion
The prediction power of differnt ML models can be compared using equality line (1:1) of the scatter plot. 1:1 line is used as a refrence in 2-D scatter plot to compare the predicted value vs the real value. This line has a slope of 1. The prediction power of ML models can be compared using this line as the best performance of a model has a corresponding scatters fall exactly on the identity line or close to the line. 

## ML models (Artifical Neural Network (ANN), and random forest regresion) 


## Dataset

Please refer to the repository (https://github.com/AmaniAbri/Neural-Network-for-structured-data) for data loading. 


## Artifical Neural Network (ANN)
Please refer to the repository (https://github.com/AmaniAbri/Neural-Network-for-structured-data) for ANN model

## Random forest regressor

### Define Random forest model:
```ruby
model_RF = RandomForestRegressor(n_estimators=1000,
              max_depth=20,
              random_state=42,
              verbose=0)
              
 
### Train model              
model_RF.fit(X_train_scaled, Y_train)


### Make predictions on test set
ypred_rf = pd.DataFrame(model_RF.predict(X_test_scaled))
ypred_rf.index = Y_test.index

### Testing the accuracy

meanSquaredError=mean_squared_error(Y_test, ypred_rf )
print("MSE:", meanSquaredError)
length = 2018
xmarks = [i for i in range(1988,length+1,3)]
```



## Scatter plot 

# Plot real vs. predicted values for both ANN and RF model
### y_pred is the predicted value from the ANN model

```ruby

plt.figure()
plt.figure(figsize=(6,8))

plt.scatter(y_pred,Y_test, c="r", alpha=0.5, marker='o', label='ANN_One_Layer')
plt.scatter(ypred_rf,Y_test, c="g", alpha=0.5, marker='o', label='RF')

### 1:1 line 
plt.plot([Y_test.min(),Y_test.max()], [Y_test.min(),Y_test.max()], 'k--', lw=4, label='Real = Predicted')

plt.ylabel('Real value')
plt.xlabel('Predicted value')
plt.title('Predicted vs. Real values')
plt.ylim(0,0.25)
plt.xlim(0,0.3)
plt.legend()


```








