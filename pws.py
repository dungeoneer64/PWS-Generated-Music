import training
import generation

import os
import sys
import pandas as pd
import numpy as np

# Much of the TensorFlow code was aided by the first lab from MIT's fantastic Introduction to Deep Learning course!

# TODO
# Automatically create track information (even if it means losing the beatiful titles T-T)
# User interface

def print_instructions():
    print("""How to use:
    create MODEL_NAME (c):            create folder structure for a new model
    train_new MODEL_NAME (tn):        start training a model from scratch
    train_load MODEL_NAME (tl):       continue training a model from the latest snapshot
    generate MODEL_NAME (gen, g):     generate a new batch of songs
    prompt MODEL_NAME (p):            generate a new batch of songs prompted by a file named prompt.mid (in the home folder)
    export (e):                       export all songs in the training data folder to WAV
Information:
    Iterations start counting at 1; the 1st iteration is iteration no. 1.    
    If you interrupt training, and restart it using the "train_load" command, the iterations after the latest snapshot will still be in loss_over_iter.csv. These entries must be removed manually.
    """)
def separator(len=40): print(len*"=")

# Main program

separator()
print("TensorFlow music generator")
separator()

# This code is a little messy... but it works

if (len(sys.argv) > 2):
    # Generate music
    if (sys.argv[1] in ("generate","gen","g")): 
        generation.generate(sys.argv[2], 3000, amount=20)
    if (sys.argv[1] in ("generate_debug","gen_debug","g_debug")): 
        generation.generate(sys.argv[2], 1000, 1, progress_bar=False)
    elif (sys.argv[1] in ("prompt", "p")):
        os.system("midi2abc prompt.mid -o prompt.abc")
        prompt = open("prompt.abc", mode="r").read()
        os.system("rm prompt.abc")
        generation.generate(sys.argv[2], 800, prompt=prompt)
    # Train a model
    elif (sys.argv[1] in ("train_load", "tl")):
        training.train(sys.argv[2], True)
    elif (sys.argv[1] in ("train_new", "tn")):
        training.train(sys.argv[2])

    elif (sys.argv[1] in ("create", "c")):
        training.create(sys.argv[2])
    else: print("ERROR: Unknown argument")

elif (len(sys.argv) > 1):
    if (sys.argv[1] in ("export", "e")):
        training.data.export_all()
    else: print("ERROR: Unknown argument")

else: print_instructions()

separator()