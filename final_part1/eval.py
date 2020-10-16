import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


def calculate_APs(df):
    '''
    preds: predictions
    n: number of all images
    m: number of images of the label of interest
    '''

    m = df['truth'].sum()  # number of images of the label of interest
    ranked = df.sort_values(by='votes', ascending=False).reset_index(drop=True)
    ranked.index += 1
    ranked['cumulative_truth'] = ranked['truth'].cumsum()
    precision = ranked['truth'] * ranked['cumulative_truth'] / ranked.index
    return precision.sum() / m, ranked


def calculate_mAP(preds, show=True):
    '''
    preds: prediction dataframe
    '''

    aps = []
    top_bottoms = []
    for pred in preds:
        ap, top_bottom = calculate_APs(pred)
        aps.append(ap)
        top_bottoms.append(top_bottom)
    if show:
        print(f"Average precisions per class: {aps}")
    top_bottoms = pd.concat(top_bottoms, axis=0).sort_values(by='votes', ascending=False)
    for idx, path in enumerate(top_bottoms.head()['paths']):
        plt.subplot(2,5,idx+1)
        plt.imshow(plt.imread(path))

    for idx, path in enumerate(top_bottoms.tail()['paths']):
        plt.subplot(2,5,idx+6)
        plt.imshow(plt.imread(path))
    plt.show()
    return np.mean(aps)


def run_loaded_experiments(labels=[1,2,9,7,3], cluser_size=400):

    models = [pickle.load(open(f'svm_{i}.pkl', 'rb')) for i in range(len(labels))]
    X_test = np.loadtxt('X_test.txt')
    Y_test = np.loadtxt('Y_test.txt')
    paths_test = np.loadtxt('paths_test.txt', dtype=str)

    dfs = []
    for i in range(len(models)):
        preds = models[i].predict(X_test)
        p = np.array(models[i].decision_function(X_test))
        df = pd.DataFrame(list(zip(preds, Y_test[:, i], preds == Y_test[:, i], p, paths_test)), columns=['preds', 'truth', 'correct', 'votes', 'paths'])
        dfs.append(df)
    print(f"mAP for this setting: {calculate_mAP(dfs)}")