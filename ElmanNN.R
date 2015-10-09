#*************************************************************************************************#
#     This function returns the time series forecasts using an Elman NN model     #
#     Depends on CrossValidation.R to find the best NN structure     #
#*************************************************************************************************#

# x: time series to forecast
# structure: best NN structure dataframe
# steps: number of forecasts desired
# epochs: number of neural networks to train
# maxit: maximum number of iterations to take
ts.Elman <- function(x, structure, steps, epochs, maxit = 1e4) {
  library(neuralnet)
  library(RSNNS)
  
  # Apply a z-score normalization to the time series
  mu <- mean(x)
  sd <- sd(x)
  z <- (x - mu) / sd 
  
  
  # best MLP structure for the time series 
  best <- structure[structure$mse == min(structure$mse), ]
  
  
  # Elman neural networks (fitted using 3 times as many hidden nodes as an MLP )
  lagged <- NULL
  for (j in best$inputs: (length(z) - 1)) {
    # outcome variable is in the first column
    lagged <- rbind(lagged, z[(j - best$inputs + 1): (j + 1)])
  }
  lagged <- data.frame(lagged)
  
  ENN <- NULL
  for (j in 1:epochs){
    enn <- elman(x = lagged[, 1:best$inputs], y = lagged[, (best$inputs + 1)], 
                 size = (3 * best$hidden), 
                 maxit = maxit, learnFunc = "JE_Rprop", learnFuncParams = 0.1)
    ENN <- c(ENN, list(enn))
  }
  
  # Elman fitted values
  elman.fit <- NULL
  for (j in 1:epochs) {
    elman.fit <- cbind(elman.fit, ENN[[j]]$fitted.values)
  }
  elman.fit <- apply(elman.fit, 1, mean) * sd + mu # original scale
  
  
  # Elman NN Forecasts
  inputs <- z
  forecast <- rep(0, steps)
  for (j in 1:steps) {
    predictions <- NULL # Holds 1-day ahead forecast from each neural network
    for(i in 1:epochs) {
      predictions <- c(predictions, predict(ENN[[i]], tail(inputs, best$inputs)))
    }
    output <- mean(predictions)
    inputs <- c(inputs, output)
    
    forecast[j] <- output * sd + mu
  }
  
  
  # Plot the results
  ts.plot(ts(x), ts(c(elman.fit, forecast), start = best$inputs + 1), lty = c(1:2), 
          main = "Actual vs Elman Fitted + Forecasts")
  legend("topleft", legend = c("Actual", "Elman"), lty = c(1:2))
  
  detach("package:neuralnet", unload = T)
  detach("package:RSNNS", unload = T)
  return(forecast)
}