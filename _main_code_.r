# Install packages
library(keras)
# install_keras()

# Read data
data <- read.csv("dataset/Cardiotocographic.csv", header = T)
str(data)

# View the first few rows of the data
head(data)

# Change to matrix
data <- as.matrix(data)
dimnames(data) <- NULL

# Normalize
data[, 1:21] <- normalize(data[, 1:21])
data[,22] <- as.numeric(data[,22]) -1
summary(data)

# # Data partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.75, 0.25))
training <- data[ind==1, 1:21]
test <- data[ind ==2, 1:21]
training_target <- data[ind==1, 22]
test_target <- data[ind==2, 22]

# # One Hot Encoding
train_Labels <- to_categorical(training_target)
test_Labels <- to_categorical(test_target)
print(test_Labels)

# # Create sequential model
model <- keras_model_sequential()
# Pipe function to add layers
model %>%
         layer_dense(units=8, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units = 3, activation = 'softmax')
summary(model)

# # Compile
model %>%
         compile(loss = 'categorical_crossentropy',
                 optimizer = 'adam',
                 metrics = 'accuracy')

# # Fit model
history <- model %>%
         fit(training,
             train_Labels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history)

# # Evaluate model with test data
model1 <- model %>%
         evaluate(test, test_Labels)

# # Prediction & confusion matrix - test data
prob <- model %>%
         predict_classes(test)

pred <- model %>%
         predict_classes(test)
#Confusion matrix  
table1 <- table(Predicted = pred, Actual = test_target)
model1
cbind(prob, pred, test_target)

# # Fine-tune model
# Pipe function to add layers
model %>%
         layer_dense(units=50, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units = 25, activation = 'relu') %>%
         layer_dense(units = 3, activation = 'softmax')
summary(model)

# # Compile
model %>%
         compile(loss = 'categorical_crossentropy',
                 optimizer = 'adam',
                 metrics = 'accuracy')

# # Fit model
history <- model %>%
         fit(training,
             train_Labels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history)

# # Evaluate model with test data
model2 <- model %>%
         evaluate(test, test_Labels)

# # Prediction & confusion matrix - test data
prob <- model %>%
         predict_classes(test)

pred <- model %>%
         predict_classes(test)
#Confusion matrix  
table1 <- table(Predicted = pred, Actual = test_target)
model2
cbind(prob, pred, test_target)

