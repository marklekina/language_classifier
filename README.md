# language classifier

## Description
A simple Naive Bayes classifier to distinguish between Swahili, Sheng and English languages

## Modules
``build_dataset.py``
- read data from three language sources (sheng, swahili and english);
- balance and merge the data to create a combined dataset;
- write dataset to .csv file.

``compile_chat_data.py`` 
- read chat data from files in a source directory and compile it into a single .csv file.

``manual_labeler.py``
- read unlabelled data from a source file;
- display text data line-by-line and prompt user to provide appropriate label;
- write labelled data to a new file.

``classifier.py``
- extracts useful features from mixed language dataset;
- train a Naive Bayes classifier model from extracted features;
- assess the performance of the model (accuracy score, most informative features, classification report);
- display a histogram distribution of the predicted language groupings by mean sentence length.