# ------------------------- dplyr -------------------------
library(dplyr)
d <- data.frame( name = c("Abhi", "Bhavesh", "Chaman", "Dimri"),
                 age = c(7, 5, 9, 16),
                 ht = c(46, NA, NA, 69),
                 school = c("Yes", "Yes", "No", "No") )

d
d.name <- arrange(d, age)
print(d.name)

# ------------------------- mlr3 -------------------------
library(mlr3)
task_penguins = as_task_classif(species ~ ., data = palmerpenguins::penguins)
task_penguins
learner = lrn("classif.rpart", cp = 0.1)
split = partition(task_penguins, ratio = 0.67)
learner$train(task_penguins, split$train_set)
prediction = learner$predict(task_penguins, split$test_set)
prediction$confusion
measure = msr("classif.acc")
prediction$score(measure)

# ------------------------- ggplot 2 -------------------------
library(ggplot2)
ggplot(data = mtcars, aes(x = hp, y = mpg, col = disp)) +
  geom_point() + 
  labs(title = "Miles per Gallon v/s Horsepower",
       x = "Horsepower",
       y = "Miles per Gallon")

