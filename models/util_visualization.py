import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import auc

def plot_feature_importances(evaluation_df, model_name):
    title_font = {'fontname': 'Arial', 'fontsize': 30, 'fontweight': 'bold'}

    # Get column names ending with "_importance"
    importance_columns = [col for col in evaluation_df.columns if col.endswith("_importance")]

    # Calculate the average importance value for each feature
    importance_avgs = evaluation_df[importance_columns].mean(axis=0)
    cleaned_index = [index.replace("_importance", "") for index in importance_avgs.index]
    importance_avgs = pd.Series(importance_avgs.values, index=cleaned_index)

    plt.figure(figsize=(20, 15))
    plt.yticks(range(len(importance_avgs)), importance_avgs.index, fontsize=23)
    plt.xticks(fontsize=25)
    plt.tight_layout()
    plt.barh(range(len(importance_avgs)), importance_avgs)
    plt.title(f'Feature Importance in {model_name} Model', **title_font)
    plt.ylabel('Features', fontsize=30)
    plt.xlabel('Importance', fontsize=30)
    plt.show()

def plot_roc_curve(evaluation_df, model_name):
    tprs = []
    aucs = evaluation_df['AUC-ROC']
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(len(evaluation_df)):
        interp_tpr = np.interp(mean_fpr, evaluation_df['False Positive Rate'][i], evaluation_df['True Positive Rate'][i])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve for {model_name} Model"
    )
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line

    ax.legend(loc="lower right")
    plt.show()