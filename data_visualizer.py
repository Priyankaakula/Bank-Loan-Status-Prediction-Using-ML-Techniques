import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    @staticmethod
    def visualize_distribution(data, column, save_path=r"C:\Users\pinky\Data Mining\Updated Project"):
        try:
            if save_path and not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')

            if save_path:
                plt.savefig(os.path.join(save_path, f'Distribution_{column}.png'))

            plt.show()
        finally:
            plt.close()

    @staticmethod
    def visualize_count(data, column, save_path=None):
        try:
            if save_path and not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(10, 6))
            sns.countplot(x=data[column], order=data[column].value_counts().index)
            plt.title(f'Count of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)

            if save_path:
                plt.savefig(os.path.join(save_path, f'Count_{column}.png'))

            plt.show()
        finally:
            plt.close()

    @staticmethod
    def visualize_boxplot(data, column, save_path=None):
        try:
            if save_path and not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(10, 6))
            sns.boxplot(y=data[column])
            plt.title(f'Box Plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Value')

            if save_path:
                plt.savefig(os.path.join(save_path, f'Boxplot_{column}.png'))

            plt.show()
        finally:
            plt.close()

    @staticmethod
    def visualize_scatter(data, x_column, y_column, save_path=None):
        try:
            if save_path and not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[x_column], y=data[y_column])
            plt.title(f'{x_column} vs {y_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)

            if save_path:
                plt.savefig(os.path.join(save_path, f'Scatter_{x_column}_vs_{y_column}.png'))

            plt.show()
        finally:
            plt.close()

    @staticmethod
    def visualize_heatmap(data, save_path=None):
        try:
            if save_path and not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(12, 10))
            sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Heatmap of Correlations')

            if save_path:
                plt.savefig(os.path.join(save_path, 'Correlation_Heatmap.png'))

            plt.show()
        finally:
            plt.close()
