# Recommendation_Systems
Recommendation systems built for Movie Lens dataset using user and item based collaborative filtering and model based approach

  
Execution: 
spark-2.2.1-bin-hadoop2.7\bin\spark-submit Tanay_Shankar_task1_Jaccard.py ratings.csv 
spark-2.2.1-bin-hadoop2.7\bin\spark-submit Tanay_Shankar_task2_ModelBasedCF.py ratings.csv testing_small.csv 
spark-2.2.1-bin-hadoop2.7\bin\spark-submit Tanay_Shankar_task2_UserBasedCF.py ratings.csv testing_small.csv 
spark-2.2.1-bin-hadoop2.7\bin\spark-submit Tanay_Shankar_task2_ItemBasedCF.py ratings.csv testing_small.csv 
 
(Note: All filenames can be replaced by file path as well) 
Baseline 
 
 
 
Jaccard Pairs 
Total pairs = 360405 
Hash# = 60 Rows = 3 Bands = 20 
Precision = 1.0000 
Recall = 0.9379 
Time taken :141.083000183 

 
Model Based Small  
Root Mean Squared Error = 0.945888630096 
{0: 13959, 1: 3950, 2: 693, 3: 123, 4: 8} 
Time taken :92.0870001316 
Big 
Root Mean Squared Error = 0.828714545399 
{0: 3206075, 1: 745786, 2: 86074, 3: 8176, 4: 220} 
Time taken :686.105604009 
(Note: Big Output File not included due to size) 
User Based 
Root Mean Squared Error = 0.947287744943 
{0: 14967, 1: 4429, 2: 722, 3: 127, 4: 11} 
Time taken :152.724999905 
Item Based 
Root Mean Squared Error without LSH = 1.1469380972 
Predictions without LSH 
{0: 13739, 1: 4628, 2: 1329, 3: 452, 4: 108} 
Time taken :269.595000029 
 
Root Mean Squared Error with LSH = 0.968559328515 
Predictions with LSH 
{0: 14825, 1: 4412, 2: 871, 3: 143, 4: 5} 
Time taken :64.3229999542 
LSH is much faster, so saves time calculating all similar movies to a movie. I used LSH (Jaccard similarity) to get candidates for Pearson Similarity. They can be precomputed and used for prediction. I also achieved higher RMSE with LSH pairs than N=20 neighbors. 
