# Item Cold-Start Recommendation Using Soft-Cluster Embeddings

The following modules needs to be installed

```
    python2.7
    scikit-learn
    numpy
    scipy
    joblib
    logging

```

### Installing
```
pip install sklearn
pip install numpy
pip install scipy
pip install joblib

```
unzip the file and run the code

## Running the Code

To run the algorithm on the MovieLens and Yahoo! Movies dataset, execute the command
```

./main_cv.py

```
The above command runs the algorithm on both datasets, MovieLens and Yahoo! Movies.  The results will be printed on standard
output.
Results are the average over 5-fold split of the dataset.
We did not include the dataset for MovieLens 20M and MovieLens 10M. It can be obtained from grouplens.org.
We rename the original ratings and movies file in the MovieLens 20M/10M dataset to process using the script process_ml.py to get the new set of dataset to be used in our experiment.
The code comes with default set of hyperparameters for the MovieLens 1M dataset.  For other datasets, please use the hyperparameters mentioned in the paper.
```
