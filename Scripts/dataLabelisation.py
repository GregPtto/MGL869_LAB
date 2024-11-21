import os
import pandas as pd

bugs_file_path = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\MGL869_LAB\CSV_files\bug_file_association.csv'
metrics_folder_path = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\MGL869_LAB\CSV_files\Metrics'

bugs_file = pd.read_csv(bugs_file_path)

def add_bug_column(metrics_file_path):
    metrics_file = pd.read_csv(metrics_file_path)
    metrics_file['Bogue'] = 0

    for _, bug_row in bugs_file.iterrows():
        bug_versions = str(bug_row['Affected Versions']).split(",")
        bug_path = str(bug_row['File_Path']).split("/").pop()

        for version in bug_versions:
            metrics_file.loc[(metrics_file['Version'].str.endswith(version)) & (metrics_file['Name'] == (bug_path)), 'Bogue'] = 1

    metrics_file.to_csv(metrics_file_path, index=False)


if __name__ == '__main__':    
    for filename in os.listdir(metrics_folder_path):
        metrics_file_path = os.path.join(metrics_folder_path, filename)
        add_bug_column(metrics_file_path)