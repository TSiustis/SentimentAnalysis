using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Transforms;
using System.Text;

namespace SentimentAnalysis
{
    class Program { 
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "dataset.txt");
    
        static void Main(string[] args)
        {
            MLContext MLContext = new MLContext();
            TrainTestData splitDataView = LoadData(MLContext);
            TrainTestData splitDataViewBayes = LoadData(MLContext);
            Utility utility = new Utility();
            for (int i = 0; i < 5; i++)
            {
                using (System.IO.StreamWriter file =
                 new System.IO.StreamWriter(@"C:\Users\siust\OneDrive\Desktop\test.txt", true))
                {
                    file.WriteLine($"Iteration: {i}\n");
                    file.WriteLine("Using binary classification:\n");
                }
                Console.WriteLine("Using binary classification:\n");
                ITransformer model = utility.BuildAndTrainModel(MLContext, splitDataView.TrainSet, "Binary");
                EvaluateNonCalibrated(MLContext, model, splitDataView.TestSet);
                UseModelWithSingleItem(MLContext, model);
                UseModelWithBatchItems(MLContext, model);
                using (System.IO.StreamWriter file =
                new System.IO.StreamWriter(@"C:\Users\siust\OneDrive\Desktop\test.txt", true))
                {
                    file.WriteLine("Using binary classification with cross-validation:\n");
                }
                Console.WriteLine("Using binary classification with cross-validation:\n");
                ITransformer modelCross = utility.BuildAndTrainModel(MLContext, splitDataView.TrainSet, "Binary");
                //  EvaluateNonCalibrated(MLContext, model, splitDataView.TestSet);
                UseModelWithSingleItem(MLContext, model);
                UseModelWithBatchItems(MLContext, model);
                using (System.IO.StreamWriter file =
                new System.IO.StreamWriter(@"C:\Users\siust\OneDrive\Desktop\test.txt", true))
                {
                    file.WriteLine("Using SVM classification:\n");
                }
                Console.WriteLine("Using SVM classification:\n");
                ITransformer modelSvm = utility.BuildAndTrainModel(MLContext, splitDataView.TrainSet, "SVM");
                EvaluateNonCalibrated(MLContext, modelSvm, splitDataView.TestSet);
                UseModelWithSingleItem(MLContext, modelSvm);
                UseModelWithBatchItems(MLContext, modelSvm);
            }
           // Console.WriteLine("Using Naive Bayes classification:\n");
           // ITransformer modelBayes = utility.BuildAndTrainModel(MLContext, splitDataViewBayes.TrainSet, "naive");
           // EvaluateMultiClass(MLContext, modelBayes, splitDataViewBayes.TestSet);
           // UseModelWithSingleItemMulti(MLContext, modelBayes);
           // UseModelWithBatchItemsMulti(MLContext, modelBayes);

           // Console.WriteLine("Using Naive Bayes classification with cross-validation:\n");
           // ITransformer modelBayesCross = utility.BuildAndTrainModel(MLContext, splitDataViewBayes.TrainSet, "crossbayes");
           //// EvaluateNonCalibrated(MLContext, modelBayes, splitDataView.TestSet);
           // UseModelWithSingleItemMulti(MLContext, modelBayesCross);
           // UseModelWithBatchItemsMulti(MLContext, modelBayesCross);
        }
        public static void UseModelWithBatchItemsMulti(MLContext mlContext, ITransformer model)
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
    } ,
            new SentimentData
            {
                SentimentText = "This could have been better but it's alright"
            },
        new SentimentData
    {
        SentimentText = "Worst movie since Attack of The Clones"
    },
         new SentimentData {
        SentimentText = "Wow!"
    }
};
IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPredictionMulti> predictedResults = mlContext.Data.CreateEnumerable<SentimentPredictionMulti>(predictions, reuseRowObject: true);
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (SentimentPredictionMulti prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative ")} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }
        private static void UseModelWithSingleItemMulti(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPredictionMulti> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPredictionMulti>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This show is great"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")}  ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
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
    },
    new SentimentData {
        SentimentText = "This was a horrible meal"
    },
    new SentimentData
    {
        SentimentText = "This is not a good show"
    } ,
            new SentimentData
            {
                SentimentText = "This could have been better but it's alright"
            },
        new SentimentData
    {
        SentimentText = "Worst movie since Attack of The Clones"
    },
         new SentimentData {
        SentimentText = "Wow!"
     } };
    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: true);
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative ")} ");

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
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")}  ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
            using (System.IO.StreamWriter file =
             new System.IO.StreamWriter(@"C:\Users\siust\OneDrive\Desktop\test.txt", true))
            {
                file.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")}  "); ;
            }
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
            StringBuilder writeFile = new StringBuilder();
            writeFile.Append("Model quality metrics evaluation\n");
            writeFile.Append($"Accuracy: {metrics.Accuracy:P2}");
            writeFile.Append($"Auc: {metrics.AreaUnderRocCurve:P2}");
            writeFile.Append($"F1Score: {metrics.F1Score:P2}");
            writeFile.Append("=============== End of model evaluation ===============\n");
            using (System.IO.StreamWriter file =
            new System.IO.StreamWriter(@"C:\Users\siust\OneDrive\Desktop\test.txt", true))
            {
                file.WriteLine(writeFile);
            }
        }
        public static void EvaluateMultiClass(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            // BinaryClassificationMetrics metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(predictions, "Label");
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            // Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            //Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.MicroAccuracy:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }
        public static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, separatorChar: '|', hasHeader:false);
            TrainTestData splitDataView;
            
                splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
       
            return splitDataView;
        }
        public static TrainTestData LoadDataBayes(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentDataBayes>(_dataPath, separatorChar: '|', hasHeader: false);
            TrainTestData splitDataView;

            splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }

    }
}
