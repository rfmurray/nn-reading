# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
'''Feel free to contact me (zarieamir@gmail.com) if you have any questions about the code. I'd also be happy to
collaborate on projects you may be interested in (both for learning purposes, or if you're considering applying it to
your research). I've included a few simple practice problems at the bottom of this document for those interested in
playing around with the model! :) '''

from sklearn import datasets # Used to import the data set.
from keras.utils import np_utils # Used to do one-hot-encoding on the target data (it is required in this program,
# but not always depending on the data).
from sklearn.preprocessing import StandardScaler # Used to standardize the data (highly recommended and in some cases
# NECESSARY).
from keras.models import Sequential # Used to stack the neural network layers of the model.
from keras.layers import Dense # Used to create fully connected layer of "neurons" or nodes.
import seaborn as sns # Used to make the final plots look better (can also be used for direct plotting as well).
import matplotlib.pyplot as plt # Used for plotting/visualization.
from sklearn.model_selection import train_test_split # Used to split the data into training and testing set.
'''We have to train the model on the training data and evaluate the performance of the trained model on the testing 
data. Based on the performance of the model on the testing data, we further tweak the model and optimize it. Ideally, 
you would also want to have a validation data set. This data set will be used to evaluate the performance of the 
FINALIZED model. The reason why we have the validation data set is that we are still modifying the model based on the 
testing data set, therefore, the performance of the model on the testing data set does not reflect the performance of 
the model on data it has not seen (the model is not truly blind to the testing data set). For learning purposes, 
we may not include a validation data set. But, it is best practice to have such a data set for more rigorous work and 
report the performance of the model on the validation data set. '''
sns.set()

# I highly recommend reading about the data set and familiarizing yourself with it (the links are provided at the top
# of the document).
data, target = datasets.load_breast_cancer(return_X_y=True) # Fetches the data and places them in the appropriate
# variables. sklearn (and other Python libraries) come with some great data sets (faces, houses, diabetes, wine, etc.)
target = np_utils.to_categorical(target, 2) # Converts the target into binary values since we cannot work with strings.
data = StandardScaler().fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=70)

#######################################################################################################################
'''The neural network below has two hidden layers, each with 16 nodes. Furthermore, a ReLU activation function is 
used. As for the final layer, there are 2 nodes. I have used a softmax function for the final output prediction. 
Lastly, I've used a cross entropy loss function, and the 'adam' optimizer (to fine tune the hyperparameters such as 
learning rate). After the model is fully trained, you can use it to make predictions. Alternatively, you could also 
save the model and load it later on (pass it on to someone else so they can use the trained model). '''
classifier = Sequential()
classifier.add(Dense(units=16, kernel_initializer='uniform', bias_initializer='uniform', activation='relu', input_dim=30))
classifier.add(Dense(units=16, kernel_initializer='uniform', bias_initializer='uniform', activation='relu'))
classifier.add(Dense(units=2, kernel_initializer='uniform', bias_initializer='uniform', activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=8, epochs=15)
#######################################################################################################################

# All the code below is related to the final plotting of the performance of the model.
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#######################################################################################################################
'''The following are some things to consider in evaluating the performance of your model (there are more, 
but this should be good for practice. Subtle hint: plots, plots, and plots): accuracy of training and testing set, 
pattern of accuracy in the training and testing set, pattern of loss/cost in the training and testing set, 
is the model overfitting/underfitting the data?

NOTE: DO NOT CHANGE THE UNITS (NODES) IN THE FINAL OUTPUT LAYER. IT IS SET TO 2 BECAUSE WE ARE DOING A BINARY 
CLASSIFICATION. 

Problem 1: Decrease/increase the number of nodes only in the first layer (original number = 16) and observe the 
change in the performance of your model. 

Problem 2: Decrease/increase the number of nodes only in the final hidden layer (original number = 16, not the output 
layer) and observe the difference in the performance of your model. 

Problem 3: Change the activation functions of the hidden layers (original AF = 'relu') and observe the difference in 
the performance of your model. 

Problem 4: Change the optimizer (original optimizer = 'adam') to a different optimizer and observe the difference in 
the performance of your model. 

Problem 5: Change the loss/cost function (original loss/cost function = 'categorical_crossentropy') to a different 
loss/cost function and observe the difference in the performance of your model.

Problem 6: Remove one of the hidden layers and observe the difference in the performance of your model.

Challenging problems:
- Add a dropout layer.
- Add another hidden layer.
- Referring to all Problems: Try to understand and explain why the model's performance changed the way it did when you 
modified its parameters (e.g. did you see underfitting when you REALLY simplified the neural network (and vice versa?).

Final question: What was the simplest model you found that had the highest accuracy (without overfitting/underfitting)?
This would potentially be the most optimal model!! :)
'''