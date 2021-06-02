import random
import sys

import numpy as np
import pandas as pd
from music21 import chord, converter, midi, note, stream
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dense
from tensorflow.keras.layers.embeddings import Embedding

VOCABULARY_SIZE = 130  # known 0-127 notes + 128 note_off + 129 no_event
SEQ_LEN = 30
BATCH_SIZE = 64
HIDDEN_UNITS = 64
EPOCHS = 5
# SEED = 2345  # 2345 seems to be good.
# input_midi_size =5
# np.random.seed(SEED)


# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.


def streamToNoteArray(stream):
    """
    Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
        0-127 - note on at specified pitch
        128   - note off
        129   - no event
    """
    # Part one, extract from stream
    total_length = np.int(np.round(stream.flat.highestTime / 0.25))  # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append(
                [np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.pitch.midi]
            )
        elif isinstance(element, chord.Chord):
            stream_list.append(
                [
                    np.round(element.offset / 0.25),
                    np.round(element.quarterLength / 0.25),
                    element.sortAscending().pitches[-1].midi,
                ]
            )
    np_stream_list = np.array(stream_list, dtype=np.int)
    df = pd.DataFrame({"pos": np_stream_list.T[0], "dur": np_stream_list.T[1], "pitch": np_stream_list.T[2]})
    df = df.sort_values(["pos", "pitch"], ascending=[True, False])  # sort the dataframe properly
    df = df.drop_duplicates(subset=["pos"])  # drop duplicate values
    # part 2, convert into a sequence of note events
    output = np.zeros(total_length + 1, dtype=np.int16) + np.int16(
        MELODY_NO_EVENT
    )  # set array full of no events by default.
    # Fill in the output list
    for i in range(total_length):
        if not df[df.pos == i].empty:
            n = df[df.pos == i].iloc[0]  # pick the highest pitch at each semiquaver
            output[i] = n.pitch  # set note on
            output[i + n.dur] = MELODY_NOTE_OFF
    return output


def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df["offset"] = df.index
    df["duration"] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    df.duration = df.duration.diff(-1) * -1 * 0.25  # calculate durations and change to quarter note fractions
    df = df.fillna(0.25)
    return df[["code", "duration"]]


def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            new_note = note.Rest()  # bit of an oversimplification, doesn't produce long notes.
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream


def sample(preds, temperature=1.0):
    """helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Sampling function (Generating only based on one sequance)


def sample_model(seed, model_name, length=400, temperature=1.0):
    """Samples a musicRNN given a seed sequence."""
    generated = []
    generated.append(seed)
    next_index = seed
    for i in range(length):
        x = np.array([next_index])
        x = np.reshape(x, (1, 1))
        preds = model_name.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        generated.append(next_index)
    return np.array(generated)


# Generating based on diffrenet sequqnces:


def decoder_model(input_midi_size):
    """This function changes the model for doffrent MIDI input size"""

    model_dec = Sequential()
    model_dec.add(
        Embedding(VOCABULARY_SIZE, HIDDEN_UNITS, input_length=input_midi_size, batch_input_shape=(1, input_midi_size))
    )

    # LSTM part
    model_dec.add(Bidirectional(GRU(HIDDEN_UNITS, return_sequences=True)))
    model_dec.add(GRU(HIDDEN_UNITS))

    # project back to vocabulary
    model_dec.add(Dense(VOCABULARY_SIZE, activation="softmax"))
    model_dec.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model_dec.summary()
    # set weights from training model
    # model_dec.set_weights(model_train.get_weights())
    model_dec.load_weights("music_train.h5")

    return model_dec  # Build a decoding model (input length 1, batch size 1, stateful)


def diffrent_size_model(input_sequance=0, length=400, temperature=1.2):
    generated = []
    generated.extend(input_sequance)

    if SEQ_LEN < len(input_sequance):
        input_midi_size = SEQ_LEN
        next_index = generated[-input_midi_size:]
    else:
        input_midi_size = len(input_sequance)
        next_index = input_sequance

    model = decoder_model(input_midi_size)
    print(next_index)
    print(input_midi_size)
    for i in range(length):
        x = np.array([next_index])
        x = np.reshape(x, (1, input_midi_size))
        preds = model.predict(x, verbose=0)[0]
        next_seq = sample(preds, temperature)
        generated.append(next_seq)
        next_index = generated[-input_midi_size:]
    return np.array(generated)


def read_midi(midi_path):
    wm_mid = converter.parse(midi_path)
    wm_mid_rnn = streamToNoteArray(wm_mid)
    return wm_mid_rnn


def main():

    if len(sys.argv) > 1:
        wm_mid_rnn = read_midi(sys.argv[1])
        out = sys.argv[1] + "out" + ".mid"
    else:
        wm_mid_rnn = [random.randint(0, MELODY_NOTE_OFF)]
        out = "generated.mid"

    if len(sys.argv) > 3:
        o = diffrent_size_model(wm_mid_rnn, int(sys.argv[2]), temperature=float(sys.argv[3]))
    elif len(sys.argv) > 2:
        o = diffrent_size_model(wm_mid_rnn, int(sys.argv[2]))
    else:
        o = diffrent_size_model(wm_mid_rnn, length=200)

    melody_stream = noteArrayToStream(o)
    mf = midi.translate.streamToMidiFile(melody_stream)

    mf.open(out, "wb")
    mf.write()
    mf.close()


if __name__ == "__main__":
    main()
