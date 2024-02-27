import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importances(evaluation_df, model_name):
    # Get column names ending with "_importance"
    importance_columns = [col for col in evaluation_df.columns if col.endswith("_importance")]

    # Calculate the average importance value for each feature
    importance_avgs = evaluation_df[importance_columns].mean(axis=0)
    cleaned_index = [index.replace("_importance", "") for index in importance_avgs.index]
    importance_avgs = pd.Series(importance_avgs.values, index=cleaned_index)

    # Plot the average as a bar chart
    if len(importance_avgs) > 5:
        plt.figure(figsize=(17, 8))
        plt.xticks(range(len(importance_avgs)), importance_avgs.index, fontsize=10, rotation='45')
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 6))
        plt.xticks(range(len(importance_avgs)), importance_avgs.index)
    plt.bar(range(len(importance_avgs)), importance_avgs)
    plt.title(f'Feature Importance in {model_name} Model')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()