using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace SentimentAnalysis
{
    public class Utility
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "dataset.txt");
       
        public  ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet,String classification)
        {
            classification = classification.ToLower();
            if (classification.Equals("binary"))
            {
                var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
                Console.WriteLine("=============== Create and Train the Model ===============");
                var model = estimator.Fit(splitTrainSet);

                var crossValidationResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(splitTrainSet, estimator, numberOfFolds: 5, labelColumnName: "Label");
                Console.WriteLine("=============== End of training ===============");
                Console.WriteLine();
                Console.WriteLine(crossValidationResults.ToString());

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
            
                return model;
            }
            else if (classification.Equals("naive"))
            {
                IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, separatorChar: '|', hasHeader: false);
                var naiveEstimator = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(SentimentData.Sentiment)))
             .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: "Label", featureColumnName: "Features"))
             .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
               
                Console.WriteLine("=============== Create and Train the Model  Naive Bayes With Split Train Set ===============");
                
                var model = naiveEstimator.Fit(splitTrainSet);
                Console.WriteLine("=============== End of training Naive Bayes With Split Train Set ===============");
                Console.WriteLine();

                return model;
            }
            else if (classification.Equals("crossbayes"))
            {
                IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, separatorChar: '|', hasHeader: false);
                var naiveEstimator = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
             .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(SentimentData.Sentiment)))
              .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: "Label", featureColumnName: "Features"))
              .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                Console.WriteLine("=============== Create and Train the Model  Naive Bayes With Cross-validation ===============");
                var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(data: dataView, estimator: naiveEstimator, numberOfFolds: 6, labelColumnName: "Label");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");

                Console.WriteLine(crossValidationResults.ToString());
                var model = naiveEstimator.Fit(dataView);
                Console.WriteLine("=============== End of training Naive Bayes With crossValidation ===============");
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

