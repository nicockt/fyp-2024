import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importances(evaluation_df, model_name):
    # Get column names ending with "_importance"
    importance_columns = [col for col in evaluation_df.columns if col.endswith("_importance")]

    importance_avgs = evaluation_df[importance_columns].mean(axis=0)
    cleaned_index = [index.replace("_importance", "") for index in importance_avgs.index]
    importance_avgs = pd.Series(importance_avgs.values, index=cleaned_index)

    # Plot the column average as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importance_avgs)), importance_avgs)
    plt.title(f'Feature Importance in {model_name} Model')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(range(len(importance_avgs)), importance_avgs.index)  # Set x-axis tick labels
    plt.show()