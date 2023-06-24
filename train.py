#import dataset related packages

import librosa
import os
import json

panjang_sampel = 22050*1 # sampling rate, remember nyquist theorem (default is 44100 hz so half of it?) * 1 means one second
panjang_sampel

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):

    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }
    count=0

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)
            
                # drop audio files with less than pre-decided number of samples
                if len(signal) >= panjang_sampel:

                    # ensure consistency of the length of the signal
                    signal = signal[:panjang_sampel]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal,sr=sample_rate,n_mfcc=40)

                    # store data for analysed track
                    count=count+1
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print(str(count)+" {}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

preprocess_dataset('dataset/train/', 'mixedformat.json')