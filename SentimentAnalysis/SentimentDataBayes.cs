using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SentimentAnalysis
{
    public class SentimentDataBayes
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1),ColumnName("Label")]
        public bool Sentiment;
    }

    public class SentimentPredictionBayes : SentimentDataBayes
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        // public float Probability { get; set; }

        //public float[] Score1 { get; set; }
        public float Score { get; set; }
    }
}
