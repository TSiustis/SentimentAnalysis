using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
namespace SentimentAnalysis
{
    class Program { 
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "dataset.txt");
    
        static void Main(string[] args)
        {
            MLContext MLContext = new MLContext();
            TrainTestData splitDataView = LoadData(MLContext);
            Utility utility = new Utility();

            Console.WriteLine("Using binary classification:\n");
            ITransformer model = utility.BuildAndTrainModel(MLContext, splitDataView.TrainSet,"Binary");
            EvaluateNonCalibrated(MLContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(MLContext, model);
            UseModelWithBatchItems(MLContext, model);

            Console.WriteLine("Using SVM classification:\n");
            ITransformer modelSvm = utility.BuildAndTrainModel(MLContext, splitDataView.TrainSet, "SVM");
            EvaluateNonCalibrated(MLContext, modelSvm, splitDataView.TestSet);
            UseModelWithSingleItem(MLContext, modelSvm);
            UseModelWithBatchItems(MLContext, modelSvm);

            Console.WriteLine("Using Naive Bayes classification:\n");
            ITransformer modelBayes = utility.BuildAndTrainModel(MLContext, splitDataView.TrainSet, "naive");
            EvaluateNonCalibrated(MLContext, modelBayes, splitDataView.TestSet);
            UseModelWithSingleItem(MLContext, modelBayes);
            UseModelWithBatchItems(MLContext, modelBayes);
        }
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
{
    new SentimentData
    {
        SentimentText = "This was a horrible meal"
    },
    new SentimentData
    {
        SentimentText = "This is not a good show"
    } };
    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: true);
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This show is great"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }
     //   public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
     //   {
            
     //       var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
     //       .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
     //      // var svmEstimator =  mlContext.Transforms.NormalizeBinning("Price", maximumBinCount: 2);
     //     //  var svmEstimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
     ////.Append(mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label", featureColumnName: "Features"));
     //       Console.WriteLine("=============== Create and Train the Model ===============");
     //       var model = estimator.Fit(splitTrainSet);
     //       Console.WriteLine("=============== End of training ===============");
     //       Console.WriteLine();

     //       return model;
     //   }
        public static void EvaluateNonCalibrated(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            BinaryClassificationMetrics metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }
        public static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, separatorChar: '|', hasHeader:false);
 
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }
    }
}
