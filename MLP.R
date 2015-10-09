#*************************************************************************************************#
#     This function returns the time series forecasts using a MLP model     #
#     Depends on CrossValidation.R to find the best NN structure     #
#*************************************************************************************************#

# x: time series to forecast
# structure: best NN structure dataframe
# steps: number of forecasts desired
# epochs: number of neural networks to train
ts.MLP <- function(x, structure, steps, epochs) {
  library(neuralnet)
  
  # Apply a z-score normalization to the time series
  mu <- mean(x)
  sd <- sd(x)
  z <- (x - mu) / sd 
  
  # "best" network structure for forecasting the time series 
  best <- structure[structure$mse == min(structure$mse), ]
  
  
  # Fit the best MLP the desired number of epochs
  lagged <- NULL
  for (j in best$inputs: (length(z) - 1)) {
    lagged <- rbind(lagged, z[(j - best$inputs + 1): (j + 1)])
  }
  lagged <- data.frame(lagged)
  
  # Create a formula for neuralnet()... the y ~ . convention does not work (neuralnet package bug) 
  names <- names(lagged)
  formula <- as.formula(paste(paste(tail(names, 1), "~ "), 
                              paste(names[-length(names)], collapse = " + "), sep = ""))
  
  MLP <- neuralnet(formula, data = lagged, hidden = best$hidden, stepmax = 1e5,
                   rep = epochs, algorithm = "rprop+", act.fct = "logistic",
                   linear.output = T, likelihood = T)
  
  # MLP fitted values:
  mlp.fit <- matrix(unlist(MLP$net.result), length(MLP$net.result[[1]]), length(MLP$net.result))
  mlp.fit <- apply(mlp.fit, 1, mean)
  mlp.fit <- mlp.fit * sd + mu # original scale
  
  
  # MLP Forecasts
  inputs <- z
  forecast <- rep(0, steps)
  for (j in 1:steps) {
    predictions <- NULL # Holds 1-day ahead forecast from each neural network
    for(i in 1:length(MLP$net.result)) { # Use all fitted mlps
      predictions <- c(predictions, 
                       compute(MLP, t(tail(inputs, best$inputs)), rep = i)$net.result)
    }
    output <- mean(predictions)
    inputs <- c(inputs, output)
    
    forecast[j] <- output * sd + mu
  }
  
  # Plot the results
  ts.plot(ts(x), ts(c(mlp.fit, forecast), start = best$inputs + 1), lty = c(1:2), 
          main = "Actual vs MLP Fitted + Forecasts")
  legend("topleft", legend = c("Actual", "MLP"), lty = c(1:2))
  
  detach("package:neuralnet", unload = T)
  return(forecast)
}