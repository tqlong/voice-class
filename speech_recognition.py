import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm

def get_mfcc(file_path):
    y, sr = librosa.load(file_path)
    hop_length = math.floor(sr*0.010)
    win_length = math.floor(sr*0.025)
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, hop_length=hop_length, n_fft=1024, win_length=win_length)
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1))
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    X = np.concatenate([mfcc, delta1, delta2], axis=0)

    return X.T

def get_class_data(data_dir):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir,f)) for f in files if f.endswith(".wav")]
    return mfcc

def clustering(X, n_clusters=8):    
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    print("centers", kmeans.cluster_centers_.shape)
    return kmeans  

if __name__ == "__main__":
    class_names = ["one", "two"]
    dataset = {}
    for cname in class_names:
        dataset[cname] = get_class_data(os.path.join("data", cname))

    all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in dataset.items()], axis=0)
    print("vectors", all_vectors.shape)
    kmeans = clustering(all_vectors)

    models = {}
    for cname in class_names:
        class_vectors = dataset[cname]
        dataset[cname] = list([kmeans.predict(v).reshape(-1,1) for v in dataset[cname]])
        hmm = hmmlearn.hmm.MultinomialHMM(
            n_components=6, random_state=0, n_iter=1000, verbose=True,
            # startprob_prior=np.array([0.7,0.2,0.1,0.0,0.0,0.0]),
            # transmat_prior=np.array([
            #     [0.7,0.2,0.1,0.0,0.0,0.0,],
            #     [0.0,0.7,0.2,0.1,0.0,0.0,],
            #     [0.0,0.0,0.7,0.2,0.1,0.0,],
            #     [0.0,0.0,0.0,0.7,0.2,0.1,],
            #     [0.0,0.0,0.0,0.0,0.8,0.2,],
            #     [0.0,0.0,0.0,0.0,0.1,0.9,],
            # ]),
        )
        X = np.concatenate(dataset[cname])
        lengths = list([len(x) for x in dataset[cname]])
        print("class", cname)
        print(X.shape, lengths, len(lengths))
        hmm.fit(X, lengths=lengths)
        models[cname] = hmm

    for true_cname in class_names:
        for O in dataset[true_cname]:
            score = {cname : model.score(O, [len(O)]) for cname, model in models.items() }
            print(true_cname, score)

    print("Training done")
