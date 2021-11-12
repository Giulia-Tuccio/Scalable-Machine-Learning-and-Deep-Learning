// Databricks notebook source
// MAGIC %md
// MAGIC # Machine Learning With Spark ML
// MAGIC In this lab assignment, you will complete a project by going through the following steps:
// MAGIC 1. Get the data.
// MAGIC 2. Discover the data to gain insights.
// MAGIC 3. Prepare the data for Machine Learning algorithms.
// MAGIC 4. Select a model and train it.
// MAGIC 5. Fine-tune your model.
// MAGIC 6. Present your solution.
// MAGIC 
// MAGIC As a dataset, we use the California Housing Prices dataset from the StatLib repository. This dataset was based on data from the 1990 California census. The dataset has the following columns
// MAGIC 1. `longitude`: a measure of how far west a house is (a higher value is farther west)
// MAGIC 2. `latitude`: a measure of how far north a house is (a higher value is farther north)
// MAGIC 3. `housing_,median_age`: median age of a house within a block (a lower number is a newer building)
// MAGIC 4. `total_rooms`: total number of rooms within a block
// MAGIC 5. `total_bedrooms`: total number of bedrooms within a block
// MAGIC 6. `population`: total number of people residing within a block
// MAGIC 7. `households`: total number of households, a group of people residing within a home unit, for a block
// MAGIC 8. `median_income`: median income for households within a block of houses
// MAGIC 9. `median_house_value`: median house value for households within a block
// MAGIC 10. `ocean_proximity`: location of the house w.r.t ocean/sea
// MAGIC 
// MAGIC ---
// MAGIC # 1. Get the data
// MAGIC Let's start the lab by loading the dataset. The can find the dataset at `data/housing.csv`. To infer column types automatically, when you are reading the file, you need to set `inferSchema` to true. Moreover enable the `header` option to read the columns' name from the file.

// COMMAND ----------

val housing = spark.read.option("inferSchema", "true").option("header","true").csv("/FileStore/tables/housing.csv")

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 2. Discover the data to gain insights
// MAGIC Now it is time to take a look at the data. In this step we are going to take a look at the data a few different ways:
// MAGIC * See the schema and dimension of the dataset
// MAGIC * Look at the data itself
// MAGIC * Statistical summary of the attributes
// MAGIC * Breakdown of the data by the categorical attribute variable
// MAGIC * Find the correlation among different attributes
// MAGIC * Make new attributes by combining existing attributes

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.1. Schema and dimension
// MAGIC Print the schema of the dataset

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

housing.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC Print the number of records in the dataset.

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

housing.count()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.2. Look at the data
// MAGIC Print the first five records of the dataset.

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

housing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC Print the number of records with population more than 10000.

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

housing.filter($"population" > 10000).count()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.3. Statistical summary
// MAGIC Print a summary of the table statistics for the attributes `housing_median_age`, `total_rooms`, `median_house_value`, and `population`. You can use the `describe` command.

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

housing.describe("housing_median_age", "total_rooms", "median_house_value", "population").show()

// COMMAND ----------

// MAGIC %md
// MAGIC Print the maximum age (`housing_median_age`), the minimum number of rooms (`total_rooms`), and the average of house values (`median_house_value`).

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.sql.functions._

housing.select(max("housing_median_age"), min("total_rooms"), avg("median_house_value")).show()


// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.4. Breakdown the data by categorical data
// MAGIC Print the number of houses in different areas (`ocean_proximity`), and sort them in descending order.

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

housing.groupBy("ocean_proximity").count().show()
housing.groupBy("ocean_proximity").count().sort(desc("count")).show()


// COMMAND ----------

// MAGIC %md
// MAGIC Print the average value of the houses (`median_house_value`) in different areas (`ocean_proximity`), and call the new column `avg_value` when print it.

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

housing.groupBy("ocean_proximity").avg("median_house_value").withColumnRenamed("avg(median_house_value)","avg_value").show()

// COMMAND ----------

// MAGIC %md
// MAGIC Rewrite the above question in SQL.

// COMMAND ----------

//TODO: Replace <FILL IN> with appropriate code

housing.createOrReplaceTempView("df")
val housingSQL = spark.sql("SELECT ocean_proximity, AVG(median_house_value) FROM df GROUP BY ocean_proximity")
housingSQL.show()
housingSQL.withColumnRenamed("avg(median_house_value)", "avg_value").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.5. Correlation among attributes
// MAGIC Print the correlation among the attributes `housing_median_age`, `total_rooms`, `median_house_value`, and `population`. To do so, first you need to put these attributes into one vector. Then, compute the standard correlation coefficient (Pearson) between every pair of attributes in this new vector. To make a vector of these attributes, you can use the `VectorAssembler` Transformer.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.feature.VectorAssembler

val va = new VectorAssembler().setInputCols(Array("housing_median_age", "total_rooms","median_house_value", "population")).setOutputCol("features")

val housingAttrs = va.transform(housing)

housingAttrs.show(5)

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

val Row(coeff: Matrix) = Correlation.corr(housingAttrs, "features").head

println(s"The standard correlation coefficient:\n ${coeff}")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.6. Combine and make new attributes
// MAGIC Now, let's try out various attribute combinations. In the given dataset, the total number of rooms in a block is not very useful, if we don't know how many households there are. What we really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very useful, and we want to compare it to the number of rooms. And the population per household seems like also an interesting attribute combination to look at. To do so, add the three new columns to the dataset as below. We will call the new dataset the `housingExtra`.
// MAGIC ```
// MAGIC rooms_per_household = total_rooms / households
// MAGIC bedrooms_per_room = total_bedrooms / total_rooms
// MAGIC population_per_household = population / households
// MAGIC ```

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

val housingCol1 = housing.withColumn("rooms_per_household", col("total_rooms")/col("households"))
val housingCol2 = housingCol1.withColumn("bedrooms_per_room", col("total_bedrooms")/col("total_rooms"))
val housingExtra = housingCol2.withColumn("population_per_household", col("population") / col("households"))

housingExtra.select("rooms_per_household", "bedrooms_per_room", "population_per_household").show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC ## 3. Prepare the data for Machine Learning algorithms
// MAGIC Before going through the Machine Learning steps, let's first rename the label column from `median_house_value` to `label`.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

val renamedHousing = housingExtra.withColumnRenamed("median_house_value", "label")

// COMMAND ----------

// MAGIC %md
// MAGIC Now, we want to separate the numerical attributes from the categorical attribute (`ocean_proximity`) and keep their column names in two different lists. Moreover, sice we don't want to apply the same transformations to the predictors (features) and the label, we should remove the label attribute from the list of predictors. 

// COMMAND ----------

// label columns
val colLabel = "label"

// categorical columns
val colCat = "ocean_proximity"

// numerical columns
val colNum = renamedHousing.columns.filter(_ != colLabel).filter(_ != colCat)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 3.1. Prepare continuse attributes
// MAGIC ### Data cleaning
// MAGIC Most Machine Learning algorithms cannot work with missing features, so we should take care of them. As a first step, let's find the columns with missing values in the numerical attributes. To do so, we can print the number of missing values of each continues attributes, listed in `colNum`.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

for (c <- colNum) {
    val missing = renamedHousing.filter(col(c).isNull).count()
    println(s"Number of missing values in column ${c}: ${missing}")
}

// COMMAND ----------

// MAGIC %md
// MAGIC As we observerd above, the `total_bedrooms` and `bedrooms_per_room` attributes have some missing values. One way to take care of missing values is to use the `Imputer` Transformer, which completes missing values in a dataset, either using the mean or the median of the columns in which the missing values are located. To use it, you need to create an `Imputer` instance, specifying that you want to replace each attribute's missing values with the "median" of that attribute.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.feature.Imputer

val imputer = new Imputer().setStrategy("median").setInputCols(Array("total_bedrooms","bedrooms_per_room")).setOutputCols(Array("total_bedrooms","bedrooms_per_room"))                                
val imputedHousing = imputer.fit(renamedHousing).transform(renamedHousing)

imputedHousing.select("total_bedrooms", "bedrooms_per_room").show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Scaling
// MAGIC One of the most important transformations you need to apply to your data is feature scaling. With few exceptions, Machine Learning algorithms don't perform well when the input numerical attributes have very different scales. This is the case for the housing data: the total number of rooms ranges from about 6 to 39,320, while the median incomes only range from 0 to 15. Note that scaling the label attribues is generally not required.
// MAGIC 
// MAGIC One way to get all attributes to have the same scale is to use standardization. In standardization, for each value, first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the variance so that the resulting distribution has unit variance. To do this, we can use the `StandardScaler` Estimator. To use `StandardScaler`, again we need to convert all the numerical attributes into a big vectore of features using `VectorAssembler`, and then call `StandardScaler` on that vactor.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

val va = new VectorAssembler().setInputCols(colNum).setOutputCol("features")

val featuredHousing = va.transform(imputedHousing)

val scaler = new StandardScaler().setInputCol("features").setOutputCol("fetaures").setWithMean(true)
val scaledHousing = scaler.fit(featuredHousing).transform(featuredHousing)

scaledHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 3.2. Prepare categorical attributes
// MAGIC After imputing and scaling the continuse attributes, we should take care of the categorical attributes. Let's first print the number of distict values of the categirical attribute `ocean_proximity`.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

renamedHousing.select(countDistinct("ocean_proximity")).show

// COMMAND ----------

// MAGIC %md
// MAGIC ### String indexer
// MAGIC Most Machine Learning algorithms prefer to work with numbers. So let's convert the categorical attribute `ocean_proximity` to numbers. To do so, we can use the `StringIndexer` that encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), ordered by label frequencies, so the most frequent label gets index 0.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.feature.StringIndexer

val indexer = new StringIndexer().setInputCol("ocean_proximity").setOutputCol("index_ocean_proximity")
val idxHousing = indexer.fit(renamedHousing).transform(renamedHousing)

idxHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we can use this numerical data in any Machine Learning algorithm. You can look at the mapping that this encoder has learned using the `labels` method: "<1H OCEAN" is mapped to 0, "INLAND" is mapped to 1, etc.

// COMMAND ----------

indexer.fit(renamedHousing).labelsArray

// COMMAND ----------

// MAGIC %md
// MAGIC ### One-hot encoding
// MAGIC Now, convert the label indices built in the last step into one-hot vectors. To do this, you can take advantage of the `OneHotEncoderEstimator` Estimator.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.feature.OneHotEncoder

val encoder = new OneHotEncoder().setInputCols(Array("index_ocean_proximity")).setOutputCols(Array("ocean_one_hot"))
val ohHousing = encoder.fit(idxHousing).transform(idxHousing)

ohHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 4. Pipeline
// MAGIC As you can see, there are many data transformation steps that need to be executed in the right order. For example, you called the `Imputer`, `VectorAssembler`, and `StandardScaler` from left to right. However, we can use the `Pipeline` class to define a sequence of Transformers/Estimators, and run them in order. A `Pipeline` is an `Estimator`, thus, after a Pipeline's `fit()` method runs, it produces a `PipelineModel`, which is a `Transformer`.
// MAGIC 
// MAGIC Now, let's create a pipeline called `numPipeline` to call the numerical transformers you built above (`imputer`, `va`, and `scaler`) in the right order from left to right, as well as a pipeline called `catPipeline` to call the categorical transformers (`indexer` and `encoder`). Then, put these two pipelines `numPipeline` and `catPipeline` into one pipeline.

// COMMAND ----------

// TODO: Replace <FILL IN> with appropriate code

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}

val numPipeline = <FILL IN>
val catPipeline = <FILL IN>
val pipeline = new Pipeline().setStages(Array(numPipeline, catPipeline))
val newHousing = pipeline.fit(renamedHousing).transform(renamedHousing)

newHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC Now, use `VectorAssembler` to put all attributes of the final dataset `newHousing` into a big vector, and call the new column `features`.

// COMMAND ----------

// MAGIC %python
// MAGIC // TODO: Replace <FILL IN> with appropriate code
// MAGIC 
// MAGIC val va2 = new VectorAssembler().<FILL IN>
// MAGIC val dataset = va2.transform(newHousing).select("features", "label")
// MAGIC 
// MAGIC dataset.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 5. Make a model
// MAGIC Here we going to make four different regression models:
// MAGIC * Linear regression model
// MAGIC * Decission tree regression
// MAGIC * Random forest regression
// MAGIC * Gradient-booster forest regression
// MAGIC 
// MAGIC But, before giving the data to train a Machine Learning model, let's first split the data into training dataset (`trainSet`) with 80% of the whole data, and test dataset (`testSet`) with 20% of it.

// COMMAND ----------

// MAGIC %python
// MAGIC // TODO: Replace <FILL IN> with appropriate code
// MAGIC 
// MAGIC val Array(trainSet, testSet) = dataset.<FILL IN>

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.1. Linear regression model
// MAGIC Now, train a Linear Regression model using the `LinearRegression` class. Then, print the coefficients and intercept of the model, as well as the summary of the model over the training set by calling the `summary` method.

// COMMAND ----------

// MAGIC %python
// MAGIC // TODO: Replace <FILL IN> with appropriate code
// MAGIC 
// MAGIC import org.apache.spark.ml.regression.LinearRegression
// MAGIC 
// MAGIC // train the model
// MAGIC val lr = <FILL IN>
// MAGIC val lrModel = lr.<FILL IN>
// MAGIC val trainingSummary = lrModel.summary
// MAGIC 
// MAGIC println(s"Coefficients: <FILL IN>, Intercept: <FILL IN>")
// MAGIC println(s"RMSE: <FILL IN>")

// COMMAND ----------

// MAGIC %md
// MAGIC Now, use `RegressionEvaluator` to measure the root-mean-square-erroe (RMSE) of the model on the test dataset.

// COMMAND ----------

// MAGIC %python
// MAGIC // TODO: Replace <FILL IN> with appropriate code
// MAGIC 
// MAGIC import org.apache.spark.ml.evaluation.RegressionEvaluator
// MAGIC 
// MAGIC // make predictions on the test data
// MAGIC val predictions = lrModel.<FILL IN>
// MAGIC predictions.select("prediction", "label", "features").show(5)
// MAGIC 
// MAGIC // select (prediction, true label) and compute test error.
// MAGIC val evaluator = new RegressionEvaluator().<FILL IN>
// MAGIC val rmse = evaluator.<FILL IN>
// MAGIC println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.2. Decision tree regression
// MAGIC Repeat what you have done on Regression Model to build a Decision Tree model. Use the `DecisionTreeRegressor` to make a model and then measure its RMSE on the test dataset.

// COMMAND ----------

// MAGIC %python
// MAGIC // TODO: Replace <FILL IN> with appropriate code
// MAGIC 
// MAGIC import org.apache.spark.ml.regression.DecisionTreeRegressor
// MAGIC import org.apache.spark.ml.evaluation.RegressionEvaluator
// MAGIC 
// MAGIC val dt = new DecisionTreeRegressor().<FILL IN>
// MAGIC 
// MAGIC // train the model
// MAGIC val dtModel = dt.<FILL IN>
// MAGIC 
// MAGIC // make predictions on the test data
// MAGIC val predictions = dtModel.<FILL IN>
// MAGIC predictions.select("prediction", "label", "features").show(5)
// MAGIC 
// MAGIC // select (prediction, true label) and compute test error
// MAGIC val evaluator = new RegressionEvaluator().<FILL IN>
// MAGIC val rmse = evaluator.<FILL IN>
// MAGIC println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.3. Random forest regression
// MAGIC Let's try the test error on a Random Forest Model. Youcan use the `RandomForestRegressor` to make a Random Forest model.

// COMMAND ----------

// MAGIC %python
// MAGIC // TODO: Replace <FILL IN> with appropriate code
// MAGIC 
// MAGIC import org.apache.spark.ml.regression.RandomForestRegressor
// MAGIC import org.apache.spark.ml.evaluation.RegressionEvaluator
// MAGIC 
// MAGIC val rf = new RandomForestRegressor().<FILL IN>
// MAGIC 
// MAGIC // train the model
// MAGIC val rfModel = rf.<FILL IN>
// MAGIC 
// MAGIC // make predictions on the test data
// MAGIC val predictions = rfModel.<FILL IN>
// MAGIC predictions.select("prediction", "label", "features").show(5)
// MAGIC 
// MAGIC // select (prediction, true label) and compute test error
// MAGIC val evaluator = new RegressionEvaluator().<FILL IN>
// MAGIC val rmse = evaluator.<FILL IN>
// MAGIC println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.4. Gradient-boosted tree regression
// MAGIC Fianlly, we want to build a Gradient-boosted Tree Regression model and test the RMSE of the test data. Use the `GBTRegressor` to build the model.

// COMMAND ----------

// MAGIC %python
// MAGIC // TODO: Replace <FILL IN> with appropriate code
// MAGIC 
// MAGIC import org.apache.spark.ml.regression.GBTRegressor
// MAGIC import org.apache.spark.ml.evaluation.RegressionEvaluator
// MAGIC 
// MAGIC val gb = new GBTRegressor().<FILL IN>
// MAGIC 
// MAGIC // train the model
// MAGIC val gbModel = gb.<FILL IN>
// MAGIC 
// MAGIC // make predictions on the test data
// MAGIC val predictions = gbModel.<FILL IN>
// MAGIC predictions.select("prediction", "label", "features").show(5)
// MAGIC 
// MAGIC // select (prediction, true label) and compute test error
// MAGIC val evaluator = new RegressionEvaluator().<FILL IN>
// MAGIC val rmse = evaluator.<FILL IN>
// MAGIC println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 6. Hyperparameter tuning
// MAGIC An important task in Machie Learning is model selection, or using data to find the best model or parameters for a given task. This is also called tuning. Tuning may be done for individual Estimators such as LinearRegression, or for entire Pipelines which include multiple algorithms, featurization, and other steps. Users can tune an entire Pipeline at once, rather than tuning each element in the Pipeline separately. MLlib supports model selection tools, such as `CrossValidator`. These tools require the following items:
// MAGIC * Estimator: algorithm or Pipeline to tune (`setEstimator`)
// MAGIC * Set of ParamMaps: parameters to choose from, sometimes called a "parameter grid" to search over (`setEstimatorParamMaps`)
// MAGIC * Evaluator: metric to measure how well a fitted Model does on held-out test data (`setEvaluator`)
// MAGIC 
// MAGIC `CrossValidator` begins by splitting the dataset into a set of folds, which are used as separate training and test datasets. For example with `k=3` folds, `CrossValidator` will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. To evaluate a particular `ParamMap`, `CrossValidator` computes the average evaluation metric for the 3 Models produced by fitting the Estimator on the 3 different (training, test) dataset pairs. After identifying the best `ParamMap`, `CrossValidator` finally re-fits the Estimator using the best ParamMap and the entire dataset.
// MAGIC 
// MAGIC Below, use the `CrossValidator` to select the best Random Forest model. To do so, you need to define a grid of parameters. Let's say we want to do the search among the different number of trees (1, 5, and 10), and different tree depth (5, 10, and 15).

// COMMAND ----------

// MAGIC %python
// MAGIC // TODO: Replace <FILL IN> with appropriate code
// MAGIC 
// MAGIC import org.apache.spark.ml.tuning.ParamGridBuilder
// MAGIC import org.apache.spark.ml.evaluation.RegressionEvaluator
// MAGIC import org.apache.spark.ml.tuning.CrossValidator
// MAGIC 
// MAGIC val paramGrid = new ParamGridBuilder().<FILL IN>
// MAGIC 
// MAGIC val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName(<FILL IN>)
// MAGIC val cv = new CrossValidator().<FILL IN>
// MAGIC val cvModel = cv.<FILL IN>
// MAGIC 
// MAGIC val predictions = cvModel.<FILL IN>
// MAGIC predictions.select("prediction", "label", "features").show(5)
// MAGIC 
// MAGIC val rmse = evaluator.<FILL IN>
// MAGIC println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 7. An End-to-End Classification Test
// MAGIC As the last step, you are given a dataset called `data/ccdefault.csv`. The dataset represents default of credit card clients. It has 30,000 cases and 24 different attributes. More details about the dataset is available at `data/ccdefault.txt`. In this task you should make three models, compare their results and conclude the ideal solution. Here are the suggested steps:
// MAGIC 1. Load the data.
// MAGIC 2. Carry out some exploratory analyses (e.g., how various features and the target variable are distributed).
// MAGIC 3. Train a model to predict the target variable (risk of `default`).
// MAGIC   - Employ three different models (logistic regression, decision tree, and random forest).
// MAGIC   - Compare the models' performances (e.g., AUC).
// MAGIC   - Defend your choice of best model (e.g., what are the strength and weaknesses of each of these models?).
// MAGIC 4. What more would you do with this data? Anything to help you devise a better solution?

// COMMAND ----------


