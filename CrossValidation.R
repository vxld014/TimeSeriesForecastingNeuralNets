#-------------------------------------------------------------------------------------------------#
#     Auxiliary function for MLP.R and ElmanNN.R     #
#     Applies k-fold cross validation for time series data to find the "best" MLP struture     #
#     Returns the MSE associated with eah network structure: (# inputs, # hidden, mse)     #
#     Considers up to lag-7 variables as inputs (can be changed)     #
#-------------------------------------------------------------------------------------------------#



# x: time series for which the "best" network structure is desired
# folds: number of CV folds to use
#-------------------------------------------------------------------------------------------------#
ts.cv <- function(x, folds) {
  library(neuralnet)
  
  x <- scale(x) # Normalize the data (used to speed up convergence)
  
  MSE <- NULL # Holds the final results
  
  # Consider up to lag-7 values as inputs for the neural network
  for (inputs in 1:7) {
    
    # Build a time series data frame for Neural Network training
    lagged <- NULL 
    for (j in inputs:(length(x) - 1)) {
      lagged <- rbind(lagged, x[(j - inputs + 1): (j + 1)])
    }
    lagged <- data.frame(lagged) # (X1, ... Xi, Xi+1), where Xi+1 serves as the output
    
    # Number of hidden units
    hidden.MSE <- NULL # Holds MSE associated with each # of hidden nodes (given the # input nodes)
    for (hidden in 1:inputs) {
      n <- dim(lagged)[1]
      index <- split(1:n, ceiling(1:n / (n / (folds + 1)))) # list of "equal sized" indices
      
      # Apply CV to find the generalized forecast accuracy of the current structure
      actuals <- NULL
      forecasts <- NULL
      for (k in 1:folds) {
        train <- lagged[unlist(index[1:k]), ]
        test <- lagged[index[[k + 1]], ]
        
        # Train 10 neural networks with the specified structure
        names <- names(train)
        formula <- as.formula(paste(paste(tail(names, 1), "~ "), paste(names[-length(names)], collapse = " + "), sep = ""))
        MLP <- neuralnet(formula, data = train, hidden = hidden, stepmax = 1e5, rep = 10,
                           algorithm = "rprop+", act.fct = "logistic", 
                           linear.output = T)
        
        # Forecast the test set
        input.values <- tail(train, 1)[-1] # Use the last row in the training set to forecast the first value of the test set
        for (p in 1:dim(test)[1]) { # Number of values to forecast
          predictions <- NULL
          for (q in 1:length(MLP$net.result)) { # Use all neural networks that actually converged
            predictions <- c(predictions, compute(MLP, data.frame(tail(input.values, inputs)), rep = q)$net.result)
          }
          output = mean(predictions) # Average the predictions to reduce bias
          input.values <- c(input.values, output) # The previous forecast becomes part of the new inputs
          forecasts <- c(forecasts, output)
        }
        actuals <- c(actuals, test[, dim(test)[2]])
      }
      # Record the MSE for each hidden node keeping the number of input nodes constant
      mse <- mean((actuals - forecasts)^2)
      hidden.MSE <- rbind(hidden.MSE, c(inputs = inputs, hidden = hidden, mse = mse))
    }
    MSE <- rbind(MSE, hidden.MSE)  # (# input units, # hidden units, mse)
  }
  detach("package:neuralnet", unload = T)
  return(data.frame(MSE))
}