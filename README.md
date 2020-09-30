# MCC-F1 Curve and Metrics
MCC-F1 curve: a performance evaluation technique for binary classification

Based on the paper - The MCC-F1 curve: a performance evaluation technique for binary classification (Cao, Chicco, & Hoffman, 2020), wherein the authors combine two single-threshold metrics i.e. Matthews correlation coefficient (MCC) and the 𝐹1 score.
into a MCC-F1 curve and also compute a metric that integrates the MCC-F1 curve inorder to compare classifier performance across varying thresholds.

The code computes the MCC-F1 curve and its relevant metrics.
* Based on 2 input values - Ground truths and Predicted values (given by a binary classifer);
* The MCC-F1 function calculates the MCC and F1 scores across varying thresholds.
* The MCC-F1 metric provides a measure to compare classifers, and provides the the best threshold 𝑇 the point on the MCC-𝐹1 curve closest to the point of perfect performance (1,1)
* Plotting the MCC-F1 curve.

## The MCC-F1 function:
Based on the inputs of ground truths and predicted values; we can calculate Matthews correlation coefficient (MCC) and the 𝐹1 scores which are scoring classifiers. 
This results in a real-valued prediction score 𝑓(𝑥𝑖) for each element, and then assigning positive predictions (𝑦𝑖̂ = 1) when the score exceeds some threshold 𝜏, or negative predictions (𝑦𝑖̂ = 0).

## The MCC-F1 metric:
Based on the MCC-F1 scores calulated we can compute the MCC-F1 Metric based on the following steps:
* Divide the normalized MCC in the curve [min𝑖 𝑋𝑖, max𝑖 𝑋𝑖] into 𝑊 = 100 sub-ranges, each of width 𝑤 = (max𝑖 𝑋𝑖 − min𝑖 𝑋𝑖)/𝑊.
* calculate the mean Euclidean distance between points with MCC in each sub-range to the point of perfect performance (1,1).
* Calculate grand average i.e. averaged the mean distances amongst subranges.
* Better classifiers have MCC-𝐹1 curves closer to the point of perfect performance (1,1), and have a larger MCC-𝐹1 metric.
