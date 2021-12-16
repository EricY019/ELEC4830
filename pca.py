from mat4py.loadmat import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def main():
    mat = loadmat('ELEC4830_Final_project.mat')
    data_spike = np.array(mat['trainSpike']).T.tolist()
    data_state = mat['trainState']

    spike, state = [], []
    for idx in range(len(data_spike)):
        if not math.isnan(data_state[idx]): # state[idx] is 0 or 1
            spike.append(data_spike[idx])
            state.append(data_state[idx])

    # transform array to DataFrame
    spike = pd.DataFrame(spike)
    state = np.array(state)[:, np.newaxis]
    state = pd.DataFrame(state, columns=['state'])
    df = pd.concat([spike, state], axis = 1)
    
    # separate out spike
    spike_columns = np.arange(16)
    spike = df.loc[:, spike_columns].values

    # standardize
    # spike = StandardScaler().fit_transform(spike)
    
    # 2d pca
    pca2d = PCA(n_components=2)
    principle_components_2d = pca2d.fit_transform(spike)
    principledf_2d = pd.DataFrame(data=principle_components_2d
                , columns=['component 1', 'component 2'])
    finaldf_2d = pd.concat([principledf_2d, df[['state']]], axis=1)
    
    # 3d pca
    # pca3d = PCA(n_components=3)
    # principle_components_3d = pca3d.fit_transform(spike)
    # principledf_3d = pd.DataFrame(data=principle_components_3d
    #             , columns=['component 1', 'component 2', 'component 3'])
    # finaldf_3d = pd.concat([principledf_3d, df[['state']]], axis=1)

    # visualize
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d') 
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    # ax.set_zlabel('Component 3', fontsize = 15)
    ax.set_title('2D PCA', fontsize = 20)

    targets = [0.0, 1.0]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indices = finaldf_2d['state'] == target
        ax.scatter(finaldf_2d.loc[indices, 'component 1']
                , finaldf_2d.loc[indices, 'component 2']
                # , finaldf_3d.loc[indices, 'component 3']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()


if __name__ == '__main__':
    main()