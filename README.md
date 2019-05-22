# HamSpam

This is an exercise to train a binary classifier on Ham/Spam given a set of labeled docs from http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html

# Summary of Approach

A vector similarity approach was used and then clustering of the resulting sparse term vectors was performed to allow use of the resulting centroids to perform speedy prediction via k-nearest neighbors.

# Details of implementation

## Data Munging
* Parse each "document" by:
   * optionally removing any punctuation
   *  splitting on whitespace
## Training
* Apply Bag of Words to each doc
*  Optionally apply  n-grams of length 1 to 4
        * add each of the combinations of length 1 to 4 to the terms list
* Determine the overall set of unique terms and create an empty vector with one dimension representing each term
* Create a sparse term vector from each "document" - which is represented by one file on disk that has been parsed by word
* Kmeans cluster the resulting sparse term vectors:  one set of clusters for Ham and another set of clusters for Spam

## Testing
* Try different combinations of data munging and hyperparameter values provide the best accuracy on the provided dataset. 
    * Grid search on the following configurable model settings: 
        Text processing
         * Case insensitive or not
         *  with and without removing punctuation including trying removing different subsets
         * with and without removing numerals since there were a lot of bare numbers in the dataset
         *  Different ranges of ngram lengths
      * Value for K - the number of clusters
      * Value of k - the number of nearest neighbours for voting
          
     * Preferably do a cross fold validation so t hat the Testing data and Training data are disjoint.  I did not have time to do this proper train-test split
  * Different approaches to scoring:  I simply set up  accuracy  But it could be argued that the Spam or Ham should be more heavily weighted than the other - leading to different scoring approaches

## Prediction
* Now we will use k nearest neighbors to predict Ham/Spam for any new documents:
     -  Do data munging and generate the sparse term vector for the new document in the same way they were generated for the training
     -  Find the k-nearest neighbors from the KMeans cluster centroids. 
      - Select the higher voted label between Ham and Spam

# Performance notes

I elected to use the more data-sciency python - vs my scala go-to language.  The performance difference is pronounced.  Python is interpreted and is mostly only  single threaded. There are ways to get some multiprocessing (/multithreading)  to speed up key portions and that's what I started working on.
