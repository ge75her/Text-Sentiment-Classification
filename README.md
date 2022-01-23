# Text-Sentiment-Classification
This is a classic text sentiment analysis task. Given a sentence, determine whether it is positive or negative. It obtains three .txt file:

training_label.txt: training data with label (0 or 1)

training_nolabel.txt: training data without label (only sentences), can be used to do semi-supervised learning

testing_data.txt: testing data, which should be predicted the label

A LSTM net is used to complete this task.

# environment
numpy  ==  1.19.5

torch  ==  1.10.0

torchvision ==  0.9.1

gensim  ==  3.6