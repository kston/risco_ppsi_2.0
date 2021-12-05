
# PROJECT NAME

![PICTURE NAME](figures/PATH)
Image by WHO, courtesy of NAME

# Introduction:

## The Problem

Describe the problem

## The Solution

Describe the solution

## Directory:

[Report Notebook](path)

[Exploratory Data Analysis](path)

[Shallow Model Development](npath)

[Deep Model Development](path)

[Data](path)

[Collection of Model Candidates](path)

[Various Assorted Charts](path)

[Working Environment](path)

[Presentation](path)


# The Data

Describe the data.

# Features for Predictive Modeling

Describe the features

**Features:**

*Activity Statistics*
*Assessment Statistics*

1. Average assessment score
2. percentage of assessments completed.

## Correlations in the Data

![image](/figures/path)

# Data Preparation
I've discussed some of the preparation I did, but here is a more comprehensive description: 

1. Say something

2. Once those are done, I drop the `code_module` column because I don't want my model considering that in its predictions.


# Model Development

### Custom Tools

The preprocessing needs for this project are unique and don't work well with off the shelf Sci-Kit Learn and Imbalanceed Learn pipelines, cross-validation, and grid search tools.  My preprocessing steps need the `code_module` feature in order to scale and balance each course separately, but that feature should not be passed to the model.  I also wanted to cross-validate by presentation rather than by randomly assembled folds.  Because of this I had to code my own versions of each of these that would fit a scaler to each course in the training set, use that fit scaler to scale both the training and test set, and then use the Imbalanced Learn SMOTE over-sampler to balance the classes in only the training set.  I also needed custom gridsearch and cross-validation tools to properly apply the proprocessing to prevent data leakage and to cross-validate by presentation rather than by randomly sampled observations.  Once I had these coded, my model development could proceed.

### Logistic Regression

I used a logistic regression model for a baseline model.  A logistic regresson model seeks the best fit line to model the linear relationship between the predictor variables, X, and the target variable, y, which is a linear regression. It then applies a sigmoid function to that line to assign probabilities that each observation belongs in one class or the other.  For my purposes, if the probability of an observation belonging a class is greater than .5, then I will predict that it belongs to that class.  A true baseline would be a randomly guessing model with an accuracy equal to the rate of class imbalance, or, in this case, about 34%.  My hyperparameter-tuned logistic regression model was able to achieve a 74% accuracy on the 2014J presentations and 79% average cross-validation scores.  There is some amount of variance in its performance on each fold, from 69% to 83%, and was 70% accurate on one presentation of module GGG and 80% accurate on another.  This indicates to me that some cohorts are harder to predict that others.  The lower accuracy on the hold-out dataset may indicate that it was generally a harder year for prediction.  In my final model I will explore accuracy across modules within the holdout set, and I find a similar amount of variation across modules, with no apparent pattern between social science and STEM modules

This and most of my shallow models tend to overpredict that students will pass the course, with lower accuracy on students who will not.

### K-Nearest Neighbors

A KNN model uses a tuneable distance metric to determine the distance between observations in n-dimensional space, where n is the number of independent variable features in the training set.  The new observation is classified according to the classes of the K (a tuneable hyper-parameter) nearest training observations.  This classification technique hypothesizes that observations of the same class will cluster together in the n-dimensional space.

My KNN model, after hyperparameter tuning, achieved almost identical results as the logistic regression model.  One benefit of both models, however, is that they train and predict quite quickly.  KNN is a great model to use when you have a low number of features because when classes do cluster together, they do so more tightly in lower dimensional space.

### Decision Tree

The logistic regression model was pretty successful, being twice as accurate as random guessing, but I thought perhaps the problem was not as linear as that model likes.  A decision tree is a promising candidate for a less linear relationship between independent and dependent variables because it does not assume the independence of the features or linear relationship between features and target. Instead it seeks to find the best way to divide and subdivide the data in a tree structure based on the values of different variables.  Once the tree is built, predictions are made by sending an observations down the tree on a path to the predicted class as it reaches each split in the tree is sent in one or the other direction.  You can think of it like a deterministic Pachinko machine.

This model averaged 77% on cross-validation scores, and 73% on the validation set.  Apparently linearity of relationships between the variables is not the only hurdle to overcome in modeling student success.

### Random Forest

This is an interesting extension to the decision tree model.  It creates a whole forest of decision trees and trains each one on a subset of the data and a subset of the features.  This is a technique called bagging, or [Boostrap AGGregation](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/) (check the link for more on this).  It works on the principle that a bunch of bad predictors, on average will be more accurate than one good predictor.  This worked for Francis Galton in [guessing the weight of an ox](https://crowdsourcingweek.com/blog/using-the-crowd-to-predict/), maybe it will work here!

In fact, this bagging strategy performed a little better than the single decision tree.  It showed a 78% accuracy across training folds and 73% on the testing presentations.  Specifically it was a little more accurate in predicting which students needed intervention (71% vs 69%).  This model was less prone to overpredicting students to pass.

### eXtreme Gradient Boosting

XGBoost models have gained a lot of popularity recently and won a lot of Kaggle competitions.  It uses another popular idea called [boosting](https://en.wikipedia.org/wiki/Gradient_boosting).  That's a pretty involved wikipedia article, but the TLDR is that it's a ensemble method like random forest, but whereas random forest trains many trees in parallel and takes the aggregate of their predictions, boosting stacks the trees on top of each other.  Each one tries to improve on the one before it by predicting the previous one's mistakes.  I think of it as like a line of poor students each grading the last one's paper, which is an analysis of the previous one's paper, etc.  Each one gets a lot wrong, but something right so the right answers percolate forward and some of the wrong answers get corrected at each step.

![XGBoost Confusion Matrix](/figures/final_modelconfmatrix.png)

The XGBoost model again performed similarly to the other shallow models with an average accuracy of 77% across validation folds and 74% accuracy on the 2014J modules.

### Final Model: Deep Neural Network

![Dense Prediction Network](figures/dense8plot.png)

I had the best luck implementing a 10 layer deep neural network for predictions.  This model predicted student success at 77% accuracy on the validation set and 76% on the hold-out set.  I did not use cross validation with the dense model, as this would have proven unfeasible given the training time and the number of cross validation folds I was using with other model.  Instead I reserved 20% of the training set, presentations before October of 2014, for validation and conducted final testing on the 2014J modules.  It continues, however to have a lot of variance in the accuracy across modules, and to overpredict student success.

In the end my models did not generalize across courses as well as I had hoped, however they can all predict student success with a minimum of 67% on any of the presentations of any of the modules in this dataset.  This indicates to me that there is some validity in my approach, using data about how students interact with their virtual learning environments to predict their success using stastistical modeling.  The fact that the variance between presentations and modules is only about 20% shows that some patterns hold true between social sciences and STEM courses and between years and cohorts.

Below you can find the confusion matrices for the October 2014 cohorts in each module in the dataset and the associated overall accuracy for that cohort.  You can see that, with the exception of Module AAA, which has the least number of students, this deep, dense model predicted with between 73% and 78% accuracy for each module.  However, for modules CCC and DDD it drastically overpredicted that students would pass the course.  Even at 50% accuracy in module DDD, however, it was better than random, which would have been about 33% for this unbalanced dataset.

![Confusion Matrices Between Modules](/figures/best_dense_course_confmats.png)

# Error Analysis

### Patterns in the Predictions

![True, Predicted, and Error means](/figures/true_pred_error_var_means.png)

1. There are significant differences in the true means of each of these variables between those who pass the courses and those who don't.  The black lines represent standard deviations within each group.

2. This model tended to exaggerate the difference it seemed to split on.  The average values in the predicted success groups are all a little higher than the average of the true success groups and the average failing values a little lower.  

3. The means of the predictor variables for the students that the model misclassified show they are true outliers for their class.  Their numbers are much closer to their opposite class, and it makes sense why the model misclassified them.  The information present in the model may simply be insufficient to accurate predict the outcomes for the students, and the model error may represent irreducible error with the data used.  It's possible that these are the cases where demographic data may tell a deeper story, but also may just be unique circumstances.

# Future deployment

This model can be useful in flagging students for intervention in online courses.  However, preprocessing data correctly is crucial for model accuracy.  In order to properly scale new data the CourseScaler transformer would need to be fitted on at least one entire course worth of data, or would at least need to be provided with the mean and standard deviation of each feature for that course.  This model does not lend itself to web deployment, but could be integrated into the backend analytics of an online education provider to help target students for support.

# Summary:
I built a dense neural network that could predict student success in an online course with, on average, 76% accuracy.  The variation in accuracy between presentations in both the cross-validation presentations and the testing presentations was significant, however, ranging from 67% to 78%.  This model is somewhat generalizable across courses with different student graduation rates, different course requirements.  However, in some cases it struggles to predict which students will need intervention.  

# Next Steps:
1. Overall accuracy may not be the best metric to train or evaluate my models on.  Evaluating using a precision-recall area-under-the-curve graph might lend insight into how different probability thresholds could improve the accuracy of this model or balance the accuracy of predictions across classes.

2. The dense modeling process has room for more experimentation with different regularization techniques and model structures.

3. I would love to try this model and preprocessing routine out on new datasets to see how well it generalizes.

4. I could try removing some of the outliers or scaling my data in different ways.

5. Finally, I could dig deeper into the errors my model is making to see if I can see what trips it up. What are the commonalities among students that are misclassified

# References: 
Data Sourcing: Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).

<sup>1</sup> https://nces.ed.gov/fastfacts/display.asp?id=80

<sup>2</sup> https://journals.sagepub.com/doi/pdf/10.1177/2158244015621777#:~:text=Online%20courses%20have%20a%2010,Smith%2C%202010)
