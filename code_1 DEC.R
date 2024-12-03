library(tidyverse)
library(tidymodels)
library(themis)
library(DALEX)
library(DALEXtra)

data <- read_csv("/Users/papoo/Desktop/ปอเอก/01-MachineLearning/projectML/final_merged_10NOV.csv")
data1 <- read_csv("/Users/papoo/Desktop/ปอเอก/01-MachineLearning/projectML/behav.csv")


glimpse(data1)
library(dplyr)
full <- data %>%
  full_join(data1, by = "student_id")
summary(full) 
dat <-full |> select(-sec, -IRT_Difficulty_Level2, -IRT_Discrimination_Level2,-CTT_Difficulty_Level,
            -CTT_Discrimination_Level, -IRT_Difficulty2,-IRT_Discrimination2,-CTT_Discrimination,
          -CTT_Difficulty,-research_score,-gender )

glimpse(dat)
is.na(dat)


dat$correct <-as.factor(dat$correct)
na_summary <- colSums(is.na(dat))
print("จำนวน NA ในแต่ละคอลัมน์:")
print(na_summary[na_summary > 0])  

# ตรวจสอบแถวที่มี NA
rows_with_na <- sum(!complete.cases(dat))
print(paste("\nจำนวนแถวที่มี NA:", rows_with_na))
print(paste("สัดส่วนแถวที่มี NA (%):", round(rows_with_na/nrow(dat)*100, 2)))

data <- na.omit(dat)
glimpse(data)

set.seed(123)
split<-initial_split(data, prop = 0.8, strata = correct)
train_data <- training(split)
test_data <- testing(split)


##glmnet
library(themis)
library(embed)
glmnet_rec <- recipe(correct ~ ., data = train_data) %>% 
  step_rm(student_id,...1,IRT_Difficulty_Level1,IRT_Discrimination_Level1 ) |>
  step_lencode_glm(department, outcome = 'correct') %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(correct) 

prep <- recipe(correct ~ ., data = train_data) %>%
  step_rm(student_id, ...1, IRT_Difficulty_Level1, IRT_Discrimination_Level1) %>% 
  step_lencode_glm(department, outcome = "correct") %>%  # เพิ่ม outcome = "correct"
  step_dummy(all_nominal_predictors()) %>%
  step_smote(correct) %>% 
  prep()

glmnet_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")


## glmnet workflow and tuning
glmnet_workflow <- workflow() %>% 
  add_recipe(glmnet_rec) %>% 
  add_model(glmnet_spec)

set.seed(123)
folds <- vfold_cv(train_data, v = 5, repeats = 3, strata = correct)

library(future)
plan(multisession, workers = 8)
glmnet_res <- glmnet_workflow %>% 
  tune_grid(
    resamples = folds, 
    grid = 10,
    metrics = metric_set(roc_auc, f_meas, accuracy, sens, spec,brier_class),
    control = control_grid(save_pred = TRUE))


glmnet_res %>% autoplot()
best_auc <- glmnet_res %>% select_best(metric = "roc_auc")


## final model
glmnet_lastfit <- glmnet_workflow %>% 
  finalize_workflow(best_auc) %>% 
  last_fit(split,
  metrics = metric_set(roc_auc, f_meas, accuracy, sens, spec,brier_class)) 


glmnet_lastfit  %>% collect_metrics()
glmnet_lastfit %>% collect_predictions() %>% 
  conf_mat(truth = correct, estimate = .pred_class) %>% summary()

glmnet_final <- glmnet_workflow %>%
    finalize_workflow(best_auc) %>% 
    fit(train_data)

glmnet_final %>% predict(new_data = test_data,
                         type = "prob")

predict_fun <- function(model, newdata) {
  predict(model, new_data = newdata, type = "prob")$.pred_fail
}
library(vip)
glmnet_final %>% vip(pred_wrapper = predict_fun)

#glmnet_final %>% predict(new_data = test_data %>% filter(student_id == "150327"),
                         type = "prob")


install.packages('keras')
library(keras)

# 1. Linear Regression Model
## Recipe
lm_rec <- recipe(correct ~ ., data = train_data) %>%
  step_rm(student_id, ...1, IRT_Difficulty_Level1, IRT_Discrimination_Level1) %>%
  step_lencode_glm(department, outcome = "correct") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors())

## Model Specification
lm_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

## Workflow and Cross-validation
lm_workflow <- workflow() %>%
  add_recipe(lm_rec) %>%
  add_model(lm_spec)

set.seed(123)
folds <- vfold_cv(train_data, v = 5, repeats = 3)

lm_res <- lm_workflow %>%
  fit_resamples(
    resamples = folds,
    metrics = metric_set(roc_auc, f_meas, accuracy, sens, spec,brier_class),
    control = control_resamples(save_pred = TRUE)
  )

# 2. Logistic Regression Model (with SMOTE)
## Recipe
logistic_rec <- recipe(correct ~ ., data = train_data) %>%
  step_rm(student_id, ...1, IRT_Difficulty_Level1, IRT_Discrimination_Level1) %>%
  step_lencode_glm(department, outcome = "correct") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(correct)

## Model Specification
logistic_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

## Workflow and Tuning
logistic_workflow <- workflow() %>%
  add_recipe(logistic_rec) %>%
  add_model(logistic_spec)

set.seed(123)
class_folds <- vfold_cv(train_data, v = 5, repeats = 3, strata = correct)

plan(multisession, workers = 8)
logistic_res <- logistic_workflow %>%
  tune_grid(
    resamples = class_folds,
    grid = 10,
    metrics = metric_set(roc_auc, f_meas, accuracy, sens, spec, brier_class),
    control = control_grid(save_pred = TRUE)
  )

logistic_res %>% autoplot()
best_auc <- logistic_res %>% select_best(metric = "roc_auc")


## final model
logistic_lastfit <- logistic_workflow %>% 
  finalize_workflow(best_auc) %>% 
  last_fit(split,
  metrics = metric_set(roc_auc, f_meas, accuracy, sens, spec,brier_class)) 


logistic_lastfit  %>% collect_metrics()
logistic_lastfit %>% collect_predictions() %>% 
  conf_mat(truth = correct, estimate = .pred_class) %>% summary()

logistic_final <- logistic_workflow %>%
    finalize_workflow(best_auc) %>% 
    fit(train_data)

logistic_final %>% predict(new_data = test_data,
                         type = "prob")

predict_fun <- function(model, newdata) {
  predict(model, new_data = newdata, type = "prob")$.pred_fail
}
library(vip)
logistic_final %>% vip(pred_wrapper = predict_fun)

# 3. Neural Network Model
# ติดตั้ง tensorflow ใน R
install.packages("tensorflow")
install.packages("keras")
library(tensorflow)
library(keras)

# ติดตั้ง tensorflow ใน Python environment
install_tensorflow()

# หรือถ้าต้องการระบุ version เฉพาะ
install_tensorflow(version = "2.12")

# ตรวจสอบการติดตั้ง
tensorflow::tf_config()

# ถ้าต้องการใช้ GPU (ถ้ามี)
install_tensorflow(gpu = TRUE)

# หรือถ้าต้องการใช้ conda environment
install_tensorflow(method = "conda")


## Recipe
nn_rec <- recipe(correct ~ ., data = train_data) %>%
  step_rm(student_id, ...1, IRT_Difficulty_Level1, IRT_Discrimination_Level1) %>%
  step_lencode_glm(department, outcome = "correct") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_smote(correct)

## Model Specification
nn_spec <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_engine("keras") %>%
  set_mode("classification")

## Workflow and Tuning
nn_workflow <- workflow() %>%
  add_recipe(nn_rec) %>%
  add_model(nn_spec)

# Define tuning grid for neural network
nn_grid <- grid_regular(
  hidden_units(range = c(5, 15)),
  penalty(range = c(-5, -1)),
  epochs(range = c(50, 100)),
  levels = 3
)

nn_res <- nn_workflow %>%
  tune_grid(
    resamples = class_folds,
    grid = nn_grid,
    metrics = metric_set(roc_auc, f_meas, accuracy, sens, spec, brier_class),
    control = control_grid(save_pred = TRUE)
  )

# Model Evaluation and Selection
## Linear Regression
lm_res %>% collect_metrics()

## Logistic Regression
best_logistic <- logistic_res %>% select_best(metric = "roc_auc")
logistic_final <- logistic_workflow %>%
  finalize_workflow(best_logistic) %>%
  fit(train_data)

## Neural Network
best_nn <- nn_res %>% select_best(metric = "roc_auc")
nn_final <- nn_workflow %>%
  finalize_workflow(best_nn) %>%
  fit(train_data)

# Final Model Evaluation
## For Logistic Regression
logistic_lastfit <- logistic_workflow %>%
  finalize_workflow(best_logistic) %>%
  last_fit(split,
    metrics = metric_set(roc_auc, f_meas, accuracy, sens, spec, brier_class)
  )

## For Neural Network
nn_lastfit <- nn_workflow %>%
  finalize_workflow(best_nn) %>%
  last_fit(split,
    metrics = metric_set(roc_auc, f_meas, accuracy, sens, spec, brier_class)
  )

# Compare Results
logistic_lastfit %>% collect_metrics()
nn_lastfit %>% collect_metrics()

# Confusion Matrices
logistic_lastfit %>% 
  collect_predictions() %>%
  conf_mat(truth = correct, estimate = .pred_class) %>% 
  summary()

nn_lastfit %>% 
  collect_predictions() %>%
  conf_mat(truth = correct, estimate = .pred_class) %>% 
  summary()
