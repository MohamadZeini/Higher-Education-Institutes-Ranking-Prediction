###########################################
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def feature_plot(importances, x_train, y_train):

    n = len(importances)
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = x_train.columns.values[indices[:n]]+1
    values = importances[indices][:n]

    # Creat the plot
    fig = plt.figure(figsize = (12,6))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 20)
    plt.bar(np.arange(n), values, width = 0.4, align="edge", color = 'navy', \
          label = "Feature Weight")
    plt.bar(np.arange(n) - 0.4, np.cumsum(values), width = 0.4, align = "edge", color = 'cornflowerblue', \
          label = "Cumulative Feature Weight")
    plt.xticks(np.arange(n), columns, rotation='vertical')
    plt.xlim((-0.5, n-0.5))
    plt.ylabel("Weight", fontsize = 20)
    plt.xlabel("Feature Dimension", fontsize = 20)
    
    #pl.legend(loc = 'upper right')
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., 
              shadow=True, fontsize='x-large')
    plt.tight_layout()
    plt.show()  
    return fig
    #fig.savefig("04.png", dpi=300,bbox_inches='tight');

    
def pca_results(data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,6))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar', width = 0.8,);
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., 
              shadow=True, fontsize='large')

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained\n Variance\n   %.4f"%(ev))
    
    #fig.savefig("05.png", dpi=600,bbox_inches='tight');

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize = (13,7))

    # Constants
    bar_width = 0.2
    colors = ['navy', 'dodgerblue', 'royalblue', 'lightsteelblue']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'R2_pca_train', 'R2_reduced_train', 'pred_time', 'R2_pca_test', 'R2_reduced_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j/3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j/3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j/3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j/3, j%3].set_xlabel("Training Set Size")
                ax[j/3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("R2 Score")
    ax[0, 2].set_ylabel("R2 score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("R2 Score")
    ax[1, 2].set_ylabel("R2 score")
    
    # Add titles
    ax[0, 0].set_title("Model Training Time")
    ax[0, 1].set_title("R2 Score on PCA Training Subset")
    ax[0, 2].set_title("R2 Score on PCA_reduced Training Subset")
    ax[1, 0].set_title("Model Predicting Time")
    ax[1, 1].set_title("R2 Score on PCA Testing Subset")
    ax[1, 2].set_title("R2 Score on PCA_Reduced Testing Subset")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Four Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.tight_layout()
    #fig.savefig("06.png", dpi=600,bbox_inches='tight');
    plt.show()
    return fig