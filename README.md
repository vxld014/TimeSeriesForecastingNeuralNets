# TimeSeriesForecastingNeuralNets
Contains R functions to implement neural network models to forecast time series. 

Suppose that y is the vector containing your time-series.
After installing the required packages and loading the functions, the following code should return a k-step ahead forecast.

> k = 7 # Desired number of forecasts
>
> # Apply Cross Validation to find the "best" network structure (constrained to a "sequence" of p lagged variables)
> cv_results = ts.cv(x=y, folds = 10) 
>
> # MLP k-step forecast
> ts.MLP(x=y, structure=cv_results, steps=k, epochs=50)
>
> # Elman NN k-step forecast
> ts.Elman(x=y, structure=cv_results, steps=k, epochs=50)
