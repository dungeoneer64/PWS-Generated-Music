import os
import numpy as np
import pandas as pd
from tensorflow.python.keras import activations
import tqdm
import datetime
import csv
import subprocess

import data

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def build_model(vocab_size, embedding_dim, dense_layers, units, batch_size, dropout, rdropout):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,None]))
    for i in range(dense_layers):
        model.add(
            tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                recurrent_initializer="glorot_uniform",
                activation=tf.nn.tanh,
                stateful=True,
                dropout=dropout,
                recurrent_dropout=rdropout
            ))
    model.add(tf.keras.layers.Dense(vocab_size))

    return model

# The following two bits of code are from mitdeeplearning (mdl) python package
def get_batch(vectorized_songs, seq_length, batch_size):
    n = len(vectorized_songs)-1
    # List of random indices, batch_size = amount
    idx = np.random.choice(n-seq_length, batch_size)
    
    # An array of numpy arrays, one for every indice
    input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1:i+seq_length+1] for i in idx]

    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    
    return x_batch, y_batch

@tf.function
def train_step(model,optimizer,x,y):
    with tf.GradientTape() as tape:
        pred = model(x)
         # Compute the loss between the labels and the prediction
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred, from_logits=True)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def gettables(vocab):
    char2idx = {b:a for (a,b) in enumerate(vocab)}
    idx2char = np.array(vocab)
    return char2idx, idx2char

def create(model_name="DEFAULT"):
    os.system("mkdir " + data.get_model_dir(model_name))
    os.system("mkdir " + os.path.join(data.get_model_dir(model_name), "training_data"))
    os.system("mkdir " + os.path.join(data.get_model_dir(model_name), "training_data_exported"))
    os.system("mkdir " + os.path.join(data.get_model_dir(model_name), "exported"))
    os.system("mkdir " + os.path.join(data.get_model_dir(model_name), "snapshots"))
    os.system("mkdir " + os.path.join(data.get_model_dir(model_name), "snapshot_to_load"))

    open(data.get_model_dir(model_name) + "/model_info.csv",mode="w").write(
        "iterations;current_iterations;batch_size;seq_length;learning_rate;embedding_dim;dense_layers;units;vocab_size;optimizer;dropout;recurrent_dropout\n" + 
        str(10000) + ";" + str(0) + ";" + str(32) + ";" + str(64) + ";" + str(2.0) + ";" + str(256) + ";" + str(2) + ";" + str(512) + ";" + str(None) + ";" + "adadelta" + ";" + str(0.15) + ";" + str(0.1)
    )

    print("""Created new model directory and model_info.csv
Configure the hyperparameters in model_info.csv (with the exception of vocab_size) to your liking
Input MIDI or ABC files into training_data/ to train the model""")

def get_optimizer(optimizer, lr):
    if optimizer in ("Adadelta","adadelta"):
        return tf.keras.optimizers.Adadelta(lr)
    elif optimizer in ("Adam","adam"):
        return tf.keras.optimizers.Adam(lr)
    elif optimizer in ("Adagrad","adagrad"):
        return tf.keras.optimizers.Adagrad(lr)
    else:
        print("Optimizer",optimizer,"not available!")
        exit(-1)

def train(model_name="DEFAULT", load_snapshot = False):

    # Hyperparameters
    (iterations,current_iterations,batch_size,seq_length,learning_rate,embedding_dim,dense_layers,units,vocab_size,optimizer_name,dropout,recurrent_dropout) = data.get_hyperparameters(model_name)
    
    # Import data
    songs = data.load_data(model_name)
    # Iterate over songs and join them into one long string
    songs_joined = "\n".join(songs)

    # Define vocab (the unique characters used) and lookup tables
    vocab = sorted(set(songs_joined))
    char2idx, idx2char = gettables(vocab)

    # Translate the string into indices using char2idx
    vectorized_songs = np.array([char2idx[char] for char in songs_joined])

    print("VECTORIZED SONG LENGTH: ", len(vectorized_songs))

    # Overwrite vocab_size
    vocab_size = len(vocab)

    # Write model info file

    print("Model name:          ",model_name)
    print("Iterations:          ",iterations)
    print("Batch size:          ",batch_size)
    print("Sequence length:     ",seq_length)
    print("Learning rate:       ",learning_rate)
    print("Embedding dimensions:",embedding_dim)
    print("Deep layers:         ",dense_layers)
    print("Units:               ",units)
    print("Vocab size:          ",units)
    print("Optimizer:           ",optimizer_name)
    print("Dropout:             ",dropout)
    print("Recurrent dropout:   ",recurrent_dropout)


    model = build_model(vocab_size, embedding_dim, dense_layers, units, batch_size, dropout, recurrent_dropout)

    if load_snapshot:
        model.load_weights(tf.train.latest_checkpoint(data.get_model_dir(model_name) + "/snapshot_to_load"))
        model.build(tf.TensorShape([1,None]))
    else:
        current_iterations = 0

    open(data.get_model_dir(model_name) + "/model_info.csv",mode="w").write(
        "iterations;current_iterations;batch_size;seq_length;learning_rate;embedding_dim;dense_layers;units;vocab_size;optimizer;dropout;recurrent_dropout\n" + 
        str(iterations) + ";" + str(current_iterations) + ";" + str(batch_size) + ";" + str(seq_length) + ";" + str(learning_rate) + ";" + str(embedding_dim) + ";" + str(dense_layers) + ";" + str(units) + ";" + str(vocab_size) + ";" + optimizer_name + ";" + str(dropout) + ";" + str(recurrent_dropout)
    )

    model.summary()

    optimizer = get_optimizer(optimizer_name,learning_rate)

    # Clear any progress bars
    if hasattr(tqdm,"_instances"): tqdm._instances.clear()

    with open(data.get_model_dir(model_name) + "/loss_over_iter.csv",mode="a") as file:
        if os.stat(data.get_model_dir(model_name) + "/loss_over_iter.csv").st_size == 0:
            writer = csv.writer(file)
            fields=["iter","mean loss"]
            writer.writerow(fields)

    bar = tqdm.tqdm(range(current_iterations+1,iterations+1),initial=current_iterations,dynamic_ncols=True,colour='green')
    last_snapshot = "-"

    # Iterate and train
    for iter in bar:
        

        x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
        loss = train_step(model, optimizer, x_batch, y_batch)

        mean_loss = loss.numpy().mean()
        bar.set_description(" Mean loss: " + str(mean_loss.round(5)) + ", last snapshot: " + str(last_snapshot))

        with open(data.get_model_dir(model_name) + "/loss_over_iter.csv",mode="a") as file:
            if iter != current_iterations or iter == 0:
                writer = csv.writer(file)
                fields=[iter,mean_loss]
                writer.writerow(fields)

        if iter % 200 == 0 and iter != 0 and iter != current_iterations:

            bar.set_description(" Mean loss: " + str(mean_loss.round(5)) + " [GENERATING SONG]")

            last_snapshot = iter
            # Save in archive, and in snapshot_to_load
            model.save_weights(data.get_model_dir(model_name) + "/snapshots/" + str(iter) + "_" + str(mean_loss.round(2)) + "L_" + str(datetime.datetime.now()) + "/snapshot")
            model.save_weights(data.get_model_dir(model_name) + "/snapshot_to_load/snapshot")
            # write current iter count to model_info
            current_iterations = iter
            open(data.get_model_dir(model_name) + "/model_info.csv",mode="w").write(
                "iterations;current_iterations;batch_size;seq_length;learning_rate;embedding_dim;dense_layers;units;vocab_size;optimizer;dropout;recurrent_dropout\n" + 
                str(iterations) + ";" + str(current_iterations) + ";" + str(batch_size) + ";" + str(seq_length) + ";" + str(learning_rate) + ";" + str(embedding_dim) + ";" + str(dense_layers) + ";" + str(units) + ";" + str(vocab_size) + ";" + optimizer_name + ";" + str(dropout) + ";" + str(recurrent_dropout)
            )
            subprocess.run(['python', 'pws.py', 'gen_debug', model_name],stdout=subprocess.DEVNULL)

    model.save_weights(data.get_model_dir(model_name) + "/snapshots/snapshot" + "_final" + "/snapshot")
    model.save_weights(data.get_model_dir(model_name) + "/snapshot_to_load/snapshot")