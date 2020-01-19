using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SentimentAnalysis
{
    public class Utility
    {
        public void getSvmPrediction() {
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var dataPoints = GenerateRandomDataPoints(1000);

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);

            // Define trainer options.
            var options = new LinearSvmTrainer.Options
            {
                BatchSize = 10,
                PerformProjection = true,
                NumberOfIterations = 10
            };

            // Define the trainer.
            var pipeline = mlContext.BinaryClassification.Trainers
                .LinearSvm(options);

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different
            // from training data.
            var testData = mlContext.Data
                .LoadFromEnumerable(GenerateRandomDataPoints(500, seed: 123));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data
                .CreateEnumerable<Prediction>(transformedTestData,
                reuseRowObject: false).ToList();

            // Print 5 predictions.
            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.Label}, "
                    + $"Prediction: {p.PredictedLabel}");

            // Expected output:
            //   Label: True, Prediction: True
            //   Label: False, Prediction: True
            //   Label: True, Prediction: True
            //   Label: True, Prediction: True
            //   Label: False, Prediction: False

            // Evaluate the overall metrics.
            var metrics = mlContext.BinaryClassification
                .EvaluateNonCalibrated(transformedTestData);

            PrintMetrics(metrics);

            // Expected output:
            //   Accuracy: 0.85
            //   AUC: 0.95
            //   F1 Score: 0.86
            //   Negative Precision: 0.91
            //   Negative Recall: 0.80
            //   Positive Precision: 0.80
            //   Positive Recall: 0.92
            //
            //   TEST POSITIVE RATIO:    0.4760 (238.0/(238.0+262.0))
            //   Confusion table
            //             ||======================
            //   PREDICTED || positive | negative | Recall
            //   TRUTH     ||======================
            //    positive ||      218 |       20 | 0.9160
            //    negative ||       53 |      209 | 0.7977
            //             ||======================
            //   Precision ||   0.8044 |   0.9127 |
        }
        public  ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet,String classification)
        {
            classification = classification.ToLower();
            if (classification.Equals("binary"))
            {
                var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
                Console.WriteLine("=============== Create and Train the Model ===============");
                var model = estimator.Fit(splitTrainSet);

                //var crossValidationResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(splitTrainSet, estimator, numberOfFolds: 5, labelColumnName: "Label");
                //Console.WriteLine("=============== End of training ===============");
                //Console.WriteLine();
                //Console.WriteLine(crossValidationResults.ToString());

                return model;
            }
            else if (classification.Equals("svm"))
            {
                var svmEstimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
              .Append(mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label", featureColumnName: "Features"));
                Console.WriteLine("=============== Create and Train the Model  SVM ===============");
                var model = svmEstimator.Fit(splitTrainSet);
                Console.WriteLine("=============== End of training SVM ===============");
                Console.WriteLine();
                var cvResults = mlContext.Regression.CrossValidate(splitTrainSet, svmEstimator, numberOfFolds: 5);
                ITransformer[] models = cvResults
            .OrderByDescending(fold => fold.Metrics.RSquared)
            .Select(fold => fold.Model)
            .ToArray();

                // Get Top Model
                ITransformer topModel = models[0];
                Console.WriteLine(topModel);
                return model;
            }
            else if (classification.Equals("naive"))
            {
                var naiveEstimator = mlContext.Transforms.Conversion.MapValueToKey(nameof(SentimentData.SentimentText))
             .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes());
                //var naiveEstimator = mlContext.Transforms.Conversion
                //.MapValueToKey(nameof(SentimentData.SentimentText))
                //// Apply NaiveBayes multiclass trainer.
                //.Append(mlContext.MulticlassClassification.Trainers
               // .NaiveBayes());
                Console.WriteLine("=============== Create and Train the Model  Naive Bayes ===============");
                var model = naiveEstimator.Fit(splitTrainSet);
                Console.WriteLine("=============== End of training Naive Bayes ===============");
                Console.WriteLine();

                return model;
            }
            throw new Exception("Enter a classificator name");
               

        }
        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
           int seed = 0)

        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat() > 0.5f;
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    // For data points with false label, the feature values are
                    // slightly increased by adding a constant.
                    Features = Enumerable.Repeat(label, 50)
                        .Select(x => x ? randomFloat() : randomFloat() +
                        0.1f).ToArray()

                };
            }
        }
        private class DataPoint
        {
            public bool Label { get; set; }
            [VectorType(50)]
            public float[] Features { get; set; }
        }

        // Class used to capture predictions.
        private class Prediction
        {
            // Original label.
            public bool Label { get; set; }
            // Predicted label from the trainer.
            public bool PredictedLabel { get; set; }
        }
        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}

