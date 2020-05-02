# music_generator
Generating music using deep neural nets!

*If you want to generate music right away and try the pre-trained model jump to [3. Music Genration](#3.Music-Genration).*


## Needed Libreries:

1. Music21
2. Keras
3. Numpy
4. Pandas
5. matplotlib (for training)

## 1. MIDI processing

For the MIDI processing use the follwoing code 

```python midi_processing.py <folder_of_folders_containg_midi> <output_name(optional)>```

This model will decode all the MIDI files in the containing folder of folders to Numopy arrays of music sequences. 

If you have different datasets for training and validation, do the preprocessing for each separately.


## 2. Training Model

For training your own model use the following:

```python music_train.py <processed_train_data> <processed_val_data(optional)>```

If you do not have diffrent dataset for the validation, the model will train on a 90%/10% training/validation spilit.

## 3.Music-Genration

1. Given a MIDI music it will generate notes that are likely to follow your input melody. 
2. If no inputs are given it will generate music from scratch.

```python generate_music.py <MIDI_input_path(optional)> <output_length(optional)> <tempreture(optional)>```

- **MIDI_input_file**: you can insert any MIDI file as the input, if no inputs is given, the model will generate a music from scratch.
- **output_length**: The number of quarter notes you want to be generated.
- **Tempreture**: Temperature is a parameter used for sampling in the last layer of the neural network. You can think of it as controlling randomness: higher values produce more variation and some- times even chaos, while lower values are more conservative in their predictions.
