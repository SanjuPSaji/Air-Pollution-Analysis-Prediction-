# Load required libraries (if not loaded already)
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)
library(ranger)
library(randomForest)
library(pROC)
library(tree)

# Set seed for reproducibility
set.seed(123)

# Import the dataset with column names
air_pollution_data <- read_csv("E:/College/Oracle/Air Pollution Analysis/Dataset/air_pollution_data.csv")

# Data Preprocessing
# Handle missing values (if needed)
air_pollution_data <- na.omit(air_pollution_data)

# Data Splitting (80% for training, 20% for testing)
splitIndex <- createDataPartition(air_pollution_data$aqi, p = 0.8, list = FALSE)
train_data <- air_pollution_data[splitIndex, ]
test_data <- air_pollution_data[-splitIndex, ]

# Exploratory Data Analysis (EDA)

# Univariate Analysis: Create a histogram for AQI
ggplot(data = air_pollution_data, aes(x = aqi)) +
  geom_histogram(fill = "blue", bins = 30) +
  labs(title = "Distribution of AQI", x = "AQI")


# Convert the "date" column to a Date object
air_pollution_data$date <- as.Date(air_pollution_data$date, format = "%d-%m-%Y")

# Univariate Analysis:Create a histogram with date on the y-axis and CO levels grouped by city on the x-axis
ggplot(air_pollution_data, aes(x = city, y = co, fill = city)) +
  geom_bar(stat = "identity", width = .70) +
  labs(
    title = "CO Levels by City",
    x = "City",
    y = "CO Level"
  ) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.95, size = 10))+ # Rotate x-axis labels for better visibility
  coord_flip()+ theme(axis.text.x = element_text(angle = 0, hjust = .51, vjust = 0.5))

# Bivariate Analysis: Create a scatter plot for Date vs CO
ggplot(data = air_pollution_data, aes(x = date, y = co)) +
  geom_point() +
  labs(title = "Scatter Plot: Date vs. CO", x = "Date", y = "CO")

# Correlation Heatmap: Create a correlation heatmap for numeric variables
corr_matrix <- cor(air_pollution_data[, c("aqi", "co", "o3")])
corrplot(corr_matrix, method = "color")

# Encode city names (one-hot encoding)
encoded_cities <- model.matrix(~ city - 1, data = air_pollution_data)
air_pollution_data <- cbind(air_pollution_data, encoded_cities)

# Checking if "aqi" is present in the columns of train_data
if (!"aqi" %in% colnames(train_data)) {
  stop("The 'aqi' column is missing in the train_data.")
}

# Fitting a Random Forest Regression Model
num_trees <- 1000
randomForest_Model <- ranger(aqi ~ ., data = train_data, num.trees = num_trees)

# Predict AQI values on the test dataset
rf_predict <- predict(randomForest_Model, data = test_data)$predictions

# Evaluate the regression model
mae <- mean(abs(rf_predict - test_data$aqi))
mse <- mean((rf_predict - test_data$aqi)^2)
rsquared <- 1 - (sum((rf_predict - test_data$aqi)^2) / sum((test_data$aqi - mean(test_data$aqi))^2))

# Print evaluation metrics
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("Mean Squared Error (MSE):", mse, "\n")
cat("R-squared (R²):", rsquared, "\n")

# Summary of the Random Forest model
print(randomForest_Model)


# Fitting a Linear Regression Model
linear_regression_model <- lm(aqi ~ ., data = train_data)

# Print evaluation metrics for Linear Regression
linear_mae <- mean(abs(predict(linear_regression_model, newdata = test_data) - test_data$aqi))
linear_mse <- mean((predict(linear_regression_model, newdata = test_data) - test_data$aqi)^2)
linear_rsquared <- summary(linear_regression_model)$r.squared
cat("Linear Regression Model:\n")
cat("Mean Absolute Error (MAE):", linear_mae, "\n")
cat("Mean Squared Error (MSE):", linear_mse, "\n")
cat("R-squared (R²):", linear_rsquared, "\n")

# Fitting a Decision Tree Model
decision_tree_model <- tree(aqi ~ ., data = train_data)

# Predict AQI values on the test dataset using the Decision Tree model
dt_predict <- predict(decision_tree_model, newdata = test_data)

# Evaluate the Decision Tree model
dt_mae <- mean(abs(dt_predict - test_data$aqi))
dt_mse <- mean((dt_predict - test_data$aqi)^2)
dt_rsquared <- 1 - (sum((dt_predict - test_data$aqi)^2) / sum((test_data$aqi - mean(test_data$aqi))^2))
cat("Decision Tree Model:\n")
cat("Mean Absolute Error (MAE):", dt_mae, "\n")
cat("Mean Squared Error (MSE):", dt_mse, "\n")
cat("R-squared (R²):", dt_rsquared, "\n")

# Summary of the Decision Tree model
summary(decision_tree_model)

# Define the KNN model with k=3
knn_model <- train(aqi ~ co + o3, data = train_data, method = "knn", trControl = trainControl(method = "cv", number = 5), preProcess = c("center", "scale"), tuneGrid = data.frame(k = 5))

# Make predictions on the test dataset
knn_predictions <- predict(knn_model, test_data)

# Calculate MAE and MSE
mae <- mean(abs(knn_predictions - test_data$aqi))
mse <- mean((knn_predictions - test_data$aqi)^2)

# Calculate R-squared for the KNN model
rsquared <- 1 - sum((knn_predictions - test_data$aqi)^2) / sum((test_data$aqi - mean(test_data$aqi))^2)

# Print evaluation metrics
cat("K-Nearest Neighbors Model:\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("Mean Squared Error (MSE):", mse, "\n")
cat("R-squared (R²):", rsquared, "\n")
