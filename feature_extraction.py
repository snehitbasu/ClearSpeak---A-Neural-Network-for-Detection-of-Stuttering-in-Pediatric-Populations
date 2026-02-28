

import os
import numpy as np
import librosa


# Define folders

project_dir = "/content/drive/MyDrive/stuttering_project"
raw_dir = os.path.join(project_dir, "raw_clips")
os.makedirs(raw_dir, exist_ok=True)

X_path = os.path.join(project_dir, "X_padded.npy")
y_path = os.path.join(project_dir, "y.npy")
labels_path = os.path.join(project_dir, "labels.csv")

print("Project directory:", project_dir)
print("Raw audio directory:", raw_dir)


#  Check if raw audio files exist

raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".wav")]
print("\nFound", len(raw_files), "raw audio clips.")

if len(raw_files) == 0:
    print("\n❗ No raw audio clips found.")
    print("👉 Upload your original .wav files into:", raw_dir)
    raise SystemExit()


#  Check if MFCC features already processed

X_exists = os.path.isfile(X_path)
y_exists = os.path.isfile(y_path)

if X_exists and y_exists:
    print("\n✔ MFCC features found — loading saved X_padded and y...")
    X_padded = np.load(X_path)
    y = np.load(y_path)
    print("Loaded shapes:", X_padded.shape, y.shape)
    print("\nYou can skip MFCC extraction and proceed to modeling.")
else:
    print("\n❗ Saved MFCC features not found. Running MFCC extraction...")

    import pandas as pd

    # Load labels.csv from Drive
    labels_df = pd.read_csv(labels_path)
    print("Loaded labels:", labels_df.shape)

    
    #  Extract MFCC features for each audio clip
    
    X_features = []
    y_labels = []

    for idx, row in labels_df.iterrows():
        filepath = row["filepath"]
        full_path = os.path.join(project_dir, filepath)

        try:
            audio, sr = librosa.load(full_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            X_features.append(mfcc.T)
            y_labels.append(row[1:].values.astype(float))
        except Exception as e:
            print("Error loading", full_path, ":", e)

    print("\nExtracted MFCCs for", len(X_features), "clips.")

    
    #  Padding to max length

    max_len = max([f.shape[0] for f in X_features])
    print("Max frame length:", max_len)

    def pad(array, maxlen):
        if array.shape[0] < maxlen:
            pad_width = maxlen - array.shape[0]
            return np.pad(array, ((0, pad_width), (0, 0)), mode='constant')
        else:
            return array[:maxlen]

    X_padded = np.array([pad(f, max_len) for f in X_features])
    y = np.array(y_labels)

    print("After padding:", X_padded.shape)

    
    #  Save permanently
    
    np.save(X_path, X_padded)
    np.save(y_path, y)

    print("\n✔ Saved X_padded.npy and y.npy to Google Drive!")
    print("Location:", project_dir)
