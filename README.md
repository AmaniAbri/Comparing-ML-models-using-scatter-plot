# Comparing-ML-models-using-scatter-plot

## Describtion
The prediction power of differnt ML models can be compared using equality line (1:1) of the scatter plot. 1:1 line is used as a refrence in 2-D scatter plot to compare sets of data. This line has a slope of 1. The prediction power of ML models can be compared using this line as the best performance of a model has a corresponding scatters fall exactly on the identity line or close to the line. 

## ML models (logistic regression, neural network, and random forest regresion) 

## Logistic regression











## Random forest regressor




# Define Random forest model:


params_RF = {'n_estimators':1000,
             'max_depth':20,
             'random_state':42,
             'verbose': 0}

# Train model 

model_RF = RandomForestRegressor(**params_RF)
model_RF.fit(X_train_scaled, Y_train)


# Make predictions on test set
ypred_rf = pd.DataFrame(model_RF.predict(X_test_scaled))
ypred_rf.index = Y_test.index

meanSquaredError=mean_squared_error(Y_test, ypred_rf )
print("MSE:", meanSquaredError)
length = 2018
xmarks = [i for i in range(1988,length+1,3)]
# Plot real vs. predicted values for both ANN and RF model
plt.figure()
plt.plot(Y_test, label='real')
plt.plot(Y_train)
plt.plot(y_pred, label='predicted ANN')
plt.plot(ypred_rf, label='predicted Random forest')
plt.plot(y_pred_Train, label='predicted') #new
plt.plot([y_pred.head().index[0],y_pred.head().index[0]], [0,80], 'k--', lw=2, label='Test set')
plt.ylim([0,1])
#plt.xticks(xmarks)
plt.legend()
plt.savefig('Real_vs_Predicted_TwoModels.png')
plt.show()

#predicted vs real

f, (ax1,ax2) = plt.subplots(1,2, figsize=(8,6) )

ax1.scatter(ypred_rf,Y_test, c="g", alpha=0.5, marker='o', label='Predictions')
ax1.plot([Y_test.min(),Y_test.max()], [Y_test.min(),Y_test.max()], 'k--', lw=4, label='Real = Predicted')
ax1.set_ylabel('Real value')
ax1.set_xlabel('Random Forest Predicted value')
ax1.set_title('RFPredicted vs. Real values')
ax1.set_ylim(0,0.25)
ax1.set_xlim(0,0.3) 
ax1.legend()

sns.distplot(perc_err, ax=ax2, color='r', fit=norm, kde=False, bins=15)
ax2.set_title(' Histogram over error')
ax2.set_xlabel('% deviation: Real - Predicted')
#ax2.set_xlim(-0.8,0.8)
plt.savefig('Real_vs_Pred_distribution.png')


# Let's calculate and visualize the feature importance for our random forest model: 
nrows = 1
ncols = 1
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(8,6));

names_regressors = [("Random Forest", model_RF)]

nregressors = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_regressors[nregressors][0]
        regressor = names_regressors[nregressors][1]
        indices = np.argsort(regressor.feature_importances_)[::-1][:40]
        fig = sns.barplot(y=df_train.columns[indices][:40],
                        x=regressor.feature_importances_[indices][:40] , 
                        orient='h',ax=axes);
        fig.set_xlabel("Relative importance",fontsize=12);
        fig.set_ylabel("Features",fontsize=12);
        fig.tick_params(labelsize=12);
        fig.set_title(name + " feature importance");
        nregressors += 1

plt.savefig('Feature_Importance.png')




## Scatter plot 

