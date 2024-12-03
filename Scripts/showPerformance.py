import matplotlib.pyplot as plt
import pandas as pd

file_path_logistic = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\MGL869_LAB\CSV_files\Results\result_logistic.csv'
file_path_forest = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\MGL869_LAB\CSV_files\Results\result_forest.csv'
mardown_path = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\MGL869_LAB\CSV_files\Results\PERFORMANCE.md'

def plot_generator(modelName : str, data : pd.DataFrame):
    fig, axes = plt.subplots(figsize=(10, 6))
    versions = data['Version']
    metrics = ['AUC', 'Precision', 'Recall']
    bar_width = 0.25
    x_indices = range(len(versions))

    for i, metric in enumerate(metrics):
        bars = axes.bar([index + bar_width * i for index in x_indices], data[metric], bar_width, label=metric)

        for bar in bars:
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)


    
    axes.set_xlabel('Version', fontsize=12)
    axes.set_ylabel('Valeur des performances', fontsize=12)
    axes.set_title(f'{modelName} : Comparaison des performances par version', fontsize=14)
    axes.set_xticks([x + bar_width for x in x_indices])
    axes.set_xticklabels(versions)
    axes.legend()


    image_path = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\MGL869_LAB\CSV_files\Results\{}.png'.format(f'{modelName}')
    plt.savefig(image_path)
    plt.close(fig)

    return image_path.split("\\").pop()

def markdown_generator(markdown_path : str, log_path : str, fr_path : str):
    with open(markdown_path, 'w', encoding="utf8") as md_file:
        md_file.write('# Rapport des métriques par version\n\n')
        md_file.write("|**Random Forest** | **Logistic Regression**|\n")
        md_file.write(":-----------------:|:-----------------------:\n")
        md_file.write(f'![Comparaison Logistic]({log_path}) | ![Comparaison Forest]({fr_path})\n')


if __name__ == '__main__':
    data_logistic = pd.read_csv(file_path_logistic)
    logistic_png = plot_generator("Logistic_Regression", data_logistic)

    data_forest = pd.read_csv(file_path_forest)
    forest_png = plot_generator("Random_Forest", data_forest)

    markdown_generator(mardown_path, logistic_png, forest_png)