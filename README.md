# Optical Character Recognition on the IAM words dataset

Based on the blog post
https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5,
I use a CNN to extract features from images of words, with a bidirectional RNN that reads these features in sequence
and outputs a sequence of classifications, which are compared against the ground-truth using CTC loss.

Currently I do not do any sort of data augmentation or pretraining. I do use optuna to search for hyperparameters
(learning rate, number of CNN and RNN layers, kernel sizes, etc.) that perform best after 20 epochs. I then train
a model using the best hyperparameters for 100 epochs, obtaining a word error rate of 16%.
