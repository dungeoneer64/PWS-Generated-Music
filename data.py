import os
import pandas as pd
import regex

# Get current working directory
dir = os.path.dirname(__file__)

def get_model_dir(model_name):
    return os.path.join(dir, "M_" + model_name)

# Exporting to wav proved to be a futile effort
# def export_to_wav(model_name, abc_text, file_name="temp"): 
#     # Export song to a text file so abc2wav can use it
#     open(get_model_dir(model_name) + "/exported/" + file_name + ".abc",mode="w").write(abc_text)

#     path = os.path.join(dir, "scripts", "convert_abc2wav")
#     # Execute command with os.system in a shell
#     os.system(path + " " + get_model_dir(model_name) + "/exported/" + file_name + ".abc")

# Extract songs in ABC format, return them as a list
def extract_songs(text):
    # Regex from mitdeeplearning (mdl) python package! (god forbid I attempt to learn regex)
    # print(text)
    pattern = '(^|\n\n)(.*?)\n\n'
    results = regex.findall(pattern, text, overlapped=True, flags=regex.DOTALL)
    songs = [song[1] for song in results]
    # print("songs has size " + str(len(songs)))
    return songs


# Extract one song from a MIDI file and convert it into ABC format
def extract_and_convert_song(model_name, file):
    # Convert midi 2 abc and store it in temp.abc
    os.system("midi2abc " + os.path.join(get_model_dir(model_name),"training_data",file) + " -o " + get_model_dir(model_name) + "/temp.abc")
    # Read text from temp.abc
    # Try, because the conversion might have failed!
    try:
        song = open(os.path.join(get_model_dir(model_name),"temp.abc"), mode="r").read()
    except:
        print("Could not convert song!")
        return ""
    # Delete temp.abc
    # os.system("rm " + get_model_dir(model_name) + "/temp.abc")
    return song

def load_data(model_name):
    """Load all data in the training directory into a list of ABC-formatted strings"""

    files = os.listdir(os.path.join(get_model_dir(model_name),"training_data"))
    songs = []

    for file in files:

        if (file in (".DS_Store")): continue

        print("Importing song(s) from", file)
        if (file[-3:] == "abc"):
            text = open(os.path.join(get_model_dir(model_name),"training_data",file), mode="r").read()
            extracted_songs = extract_songs(text)
            songs.extend(extracted_songs)
        elif (file[-3:] == "mid" or file[-3:] == "MID" or file[-4:] == "midi" or file[-4:] == "MIDI"):
            songs.append(extract_and_convert_song(model_name, file))
            
    if len(songs) > 0:
        print("Found a total of", len(songs), "songs")
    else:
        print("Found no valid songs in training_data!")
        exit(-1)

    return songs


def get_hyperparameters(model_name):

    data = pd.read_csv(get_model_dir(model_name) + "/model_info.csv",sep=";")
        
    return (
        data["iterations"].values[0],
        data["current_iterations"].values[0],
        data["batch_size"].values[0],
        data["seq_length"].values[0],
        data["learning_rate"].values[0],
        data["embedding_dim"].values[0],
        data["dense_layers"].values[0],
        data["units"].values[0],
        data["vocab_size"].values[0],
        data["optimizer"].values[0],
        data["dropout"].values[0],
        data["recurrent_dropout"].values[0]
    )

# def export_all(model_name):
#     # Improvement: use regex to get song names and use those as export titles instead of an int
#     songs = load_data()
#     i = 0
#     for song in songs:
#         export_to_wav(song, str(i))
#         i += 1