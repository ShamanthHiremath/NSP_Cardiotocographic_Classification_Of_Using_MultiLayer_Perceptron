# Load the necessary library
library(keras)

# Install Keras if not already installed
# install_keras()

# Read data
data <- read.csv("dataset/Cardiotocographic.csv", header = TRUE)

# Tells the structure of the data
str(data)

# View the first few rows of the data
head(data)

# Convert to matrix and remove column names
data <- as.matrix(data)
dimnames(data) <- NULL

# Normalize the features
data[, 1:21] <- normalize(data[, 1:21])
data[, 22] <- as.numeric(data[, 22]) - 1
summary(data)

# Data partitioning
set.seed(1234)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.75, 0.25))
training <- data[ind == 1, 1:21]
test <- data[ind == 2, 1:21]
training_target <- data[ind == 1, 22]
test_target <- data[ind == 2, 22]

# One-hot encoding of target variable
train_Labels <- to_categorical(training_target)
test_Labels <- to_categorical(test_target)
print(test_Labels)

# Create sequential model
model <- keras_model_sequential()

# Add layers to the model
model %>%
  layer_dense(units = 8, activation = 'relu', input_shape = c(21)) %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)

# Compile the model
model %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# Fit the model
history <- model %>%
  fit(training,
      train_Labels,
      epochs = 200,
      batch_size = 32,
      validation_split = 0.2)
plot(history)

# Evaluate the model with test data
model1 <- model %>%
  evaluate(test, test_Labels)

# Prediction & confusion matrix - test data
prob <- model %>%
  predict_classes(test)

pred <- model %>%
  predict_classes(test)

# Confusion matrix  
table1 <- table(Predicted = pred, Actual = test_target)
model1
cbind(prob, pred, test_target)

# Fine-tune the model by adding more layers
model <- keras_model_sequential()
model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(21)) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)

# Compile the fine-tuned model
model %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

# Fit the fine-tuned model
history <- model %>%
  fit(training,
      train_Labels,
      epochs = 200,
      batch_size = 32,
      validation_split = 0.2)
plot(history)

# Evaluate the fine-tuned model with test data
model2 <- model %>%
  evaluate(test, test_Labels)

# Prediction & confusion matrix - test data
prob <- model %>%
  predict_classes(test)

pred <- model %>%
  predict_classes(test)

# Confusion matrix  
table1 <- table(Predicted = pred, Actual = test_target)
model2
cbind(prob, pred, test_target)
'''

Todos:
Plot graphs for accuracy and also the datset features, demonstrate realistically

'''
