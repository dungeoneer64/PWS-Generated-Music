import training
import data
import tqdm
import os
import tensorflow as tf

def generate_text(model, vocab, start_string, generation_length=1000, progress_bar=True):

    char2idx, idx2char = training.gettables(vocab)

    input_eval = [char2idx[char] for char in start_string]
    # Expand dimensions, so every integer has its own
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()

    if progress_bar:
        for iter in tqdm.tqdm(range(generation_length)):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, axis=0)
            predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
            input_eval = tf.expand_dims([predicted_id],0)
            text_generated.append(idx2char[predicted_id])
    else:
        for i in range(generation_length):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, axis=0)
            predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
            input_eval = tf.expand_dims([predicted_id],0)
            text_generated.append(idx2char[predicted_id])

    text_generated = start_string + "".join(text_generated)

    # text_generated_array = data.extract_songs(text_generated)

    return ([text_generated])


# def generate(model_name, generation_length=500, prompt="X: 1\nT:song_name\nL:1/16\nQ:1/4=60\nK:C % 0 sharps\nV:1\n"):
def generate(model_name, generation_length=500, amount=10, prompt="X: 1\nT:song_name\nK:C", progress_bar=True):
      
    (iterations,current_iterations,batch_size,seq_length,learning_rate,embedding_dim,dense_layers,units,vocab_size,optimizer_name,dropout,recurrent_dropout) = training.data.get_hyperparameters(model_name)

    model = training.build_model(vocab_size, embedding_dim, dense_layers, units, batch_size=1,dropout=dropout,rdropout=recurrent_dropout)
    model.load_weights(tf.train.latest_checkpoint(training.data.get_model_dir(model_name) + "/snapshot_to_load"))
    model.build(tf.TensorShape([1,None]))

    model.summary()

    # Get vocab
    songs = training.data.load_data(model_name)
    songs_joined = "\n".join(songs)
    vocab = sorted(set(songs_joined))

    export_path = "/exported/batch"
    if progress_bar == False:
        export_path = "/exported/single"

    j = 1
    while(True):
        if os.path.isdir(training.data.get_model_dir(model_name) + export_path + "_" + str(current_iterations)  + "iter_" + str(j)):
            j += 1
        else: break
    batch_dir = training.data.get_model_dir(model_name) + export_path + "_" + str(current_iterations)  + "iter_" + str(j)
    os.system("mkdir " + batch_dir)

    # Generate music!
    for i in range(amount):
        generated_text_array = generate_text(model, vocab, start_string=prompt, generation_length=generation_length, progress_bar=progress_bar)

        print("generated_text_array has size " + str(len(generated_text_array)))

#         generated_text = generate_text(model, vocab, start_string="""X: 1
# T: from prompt.mid
# M: 3/8
# L: 1/16
# Q:1/4=120
# K:C % 0 sharps
# V:1
# %%MIDI program 0
# ac' ae ae| \\
# ce cA cA| \\
# """, generation_length=generation_length)

        # Create a new directory for the exported files

        x = 1
        for generated_text in generated_text_array:

            open(batch_dir + "/gen" + str(i+1) + ".abc",mode="w").write(generated_text)

        # training.data.export_to_wav(model_name, generated_text, str(i))
        # os.system("abc2midi " + batch_dir + "/generated" + str(i+1) + ".abc" + " -o " + batch_dir + "/midi" + str(i+1) + ".mid")
            os.system("./scripts/convert_abc2wav " + batch_dir + "/gen" + str(i+1) + ".abc")
            x += 1

        # generated_songs = training.data.extract_songs(generated_text)

        # for a,song in enumerate(generated_songs):
        #     print("exporting...")
        #     training.data.export_to_wav(song, str(a) + "_" + str(i))