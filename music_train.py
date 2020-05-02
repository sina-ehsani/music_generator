# python music_train.py melody_training_dataset.npz melody_test_dataset.npz 


# Imports

from music21 import converter, instrument, note, chord, stream, midi
import glob
import time
import numpy as np
import keras.utils as utils
import pandas as pd

import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout , GRU , Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt


# Training Hyperparameters:
VOCABULARY_SIZE = 130 # known 0-127 notes + 128 note_off + 129 no_event
SEQ_LEN = 30
BATCH_SIZE = 64
HIDDEN_UNITS = 64
EPOCHS = 50
DROPOUT = 0.3
SEED = 2345  # 2345 seems to be good.
np.random.seed(SEED)




def slice_sequence_examples(sequence, num_steps):
    """Slice a sequence into redundant sequences of lenght num_steps."""
    xs = []
    for i in range(len(sequence) - num_steps - 1):
        example = sequence[i: i + num_steps]
        xs.append(example)
    return xs

def seq_to_singleton_format(examples):
    """
    Return the examples in seq to singleton format.
    """
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs,ys)

def slice_data(dataset):
	slices = []
	for seq in dataset:
	    slices +=  slice_sequence_examples(seq, SEQ_LEN+1)

	# Split the sequences into Xs and ys:
	X, y = seq_to_singleton_format(slices)
	# Convert into numpy arrays.
	X = np.array(X)
	y = np.array(y)
	return(X,y)


def plot_data(dataset):
	# Do some stats on the Train corpus.
	all_notes = np.concatenate(dataset)
	print("Number of notes:")
	print(all_notes.shape)
	all_notes_df = pd.DataFrame(all_notes)
	print("Notes that do appear:")
	unique, counts = np.unique(all_notes, return_counts=True)
	print(unique)
	print("Notes that don't appear:")
	print(np.setdiff1d(np.arange(0,129),unique))

	print("Plot the relative occurences of each note:")
	import matplotlib.pyplot as plt
	# %matplotlib inline

	#plt.style.use('dark_background')
	plt.bar(unique, counts)
	plt.yscale('log')
	plt.xlabel('melody RNN value')
	plt.ylabel('occurences (log scale)')


def model():
	# build the model: 2-layer network.
	# Using Embedding layer and sparse_categorical_crossentropy loss function 
	# Added early stopping!


	print('Build model...')
	model_train = Sequential()
	model_train.add(Embedding(VOCABULARY_SIZE, HIDDEN_UNITS, input_length=SEQ_LEN))

	# LSTM part
	model_train.add(Bidirectional(GRU(HIDDEN_UNITS, return_sequences=True, dropout=DROPOUT)))
	model_train.add(GRU(HIDDEN_UNITS , dropout=DROPOUT))

	# Project back to vocabulary
	model_train.add(Dense(VOCABULARY_SIZE, activation='softmax'))
	model_train.compile(loss='sparse_categorical_crossentropy', optimizer='adam' , metrics=['mse', 'mae', 'mape', 'cosine'])

	es = EarlyStopping(monitor='val_loss', verbose=1 ,   patience=5) # Findes early stopping using for the loss on the validation dataset, adding a delay of 5 to the trigger in terms of the number of epochs on which we would like to see no improvement. 
	mc = ModelCheckpoint('music_bigru_3layer.h5', monitor='val_loss', verbose=1 ,save_best_only=True) #The callback will save the model to file, the one with best overall performance.


	model_train.summary()

	return(model_train , es , mc)




def main():
	# Load up some melodies I prepared earlier...
	train_dataset = './' + sys.argv[1]

	with np.load(train_dataset, allow_pickle=True) as data:
		train_set = data['train']

	# Slice the melody for training:
	X, y = slice_data(train_set)

	plot_data(train_set)

	model_train , es , mc =model()

	# if we have a test data:
	if len(sys.argv)>2: 

		test_dataset = './' + sys.argv[2]

		with np.load(test_dataset, allow_pickle=True) as data:
			test_set = data['train']

		# Slice the melody for test:
		X_test, y_test = slice_data(test_set)

		plot_data(test_set)

		history=  model_train.fit(X, y,validation_data=(X_test,y_test) ,  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es,mc],verbose=1)

	else:
		history=  model_train.fit(X, y,validation_split=0.10 ,  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es,mc],verbose=1)
	
	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	model_train.save("music_model.h5")

if __name__ == '__main__':
	main()









