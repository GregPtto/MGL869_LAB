import pandas as pd
import subprocess
import os
import tempfile
import re
from multiprocessing import Process

git_repo_path = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\hive'
understand_path = r'C:\Program Files\SciTools\bin\pc-win64\und.exe'
metrics_output_master = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\MGL869_LAB\CSV_files\Metrics\metrics_master.csv'
csv_output_template = r'C:\Users\Guillaume\Documents\Ecole\ETS\MGL869\MGL869_LAB\CSV_files\Metrics\{}'

regex_tags = r'.*-([2-3]\.[0-9]\.0)$'

tags_tmp = subprocess.check_output(['git', 'tag', '--list'], cwd=git_repo_path, text=True).strip().split('\n')
tags = [tag for tag in tags_tmp if re.match(regex_tags, tag)]


def understand_execution(Output : str, understand_db : str, temp_repo_path : str):
    subprocess.run([understand_path, 'add', '-db', understand_db, temp_repo_path], check=True)
    subprocess.run([understand_path, 'settings', '-metrics', 'all', understand_db], check=True)
    subprocess.run([understand_path, 'settings', '-metricsOutputFile', Output, understand_db], check=True)
    subprocess.run([understand_path, 'analyze', understand_db], check=True)
    subprocess.run([understand_path, 'metrics', understand_db], check=True)

def analyze_version(tag : str):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_repo_path = os.path.join(temp_dir, 'hive')
        subprocess.run(['git', 'clone', git_repo_path, temp_repo_path], check=True)

        understand_db = os.path.join(temp_dir, 'db.und')
        metrics_output_master= os.path.join(temp_dir, 'metrics_master.csv')

        subprocess.run([understand_path, 'create', '-db', understand_db, '-languages', 'java',  'c++'], check=True)
        understand_execution(metrics_output_master, understand_db, temp_repo_path)
        
        commit_id = subprocess.check_output(['git', 'rev-list', '-n', '1', tag], cwd=temp_repo_path, text=True).strip()
        latest_commit = subprocess.check_output(['git', 'log', '-n', '1', '--format=%H', f'{commit_id}^', '--first-parent'], cwd=temp_repo_path, text=True).strip()
        
        subprocess.run(['git', 'checkout', latest_commit], cwd=temp_repo_path, check=True)

        metrics_output = os.path.join(temp_dir, 'metrics_temp.csv')
        understand_execution(metrics_output, understand_db, temp_repo_path)

        metrics_df = pd.read_csv(metrics_output, low_memory=False)
        metrics_df['Version'] = tag
        metrics_df['CommitId'] = latest_commit

        csv_output = csv_output_template.format(f'metrics-{tag.replace("/", "-")}.csv')
        metrics_df.to_csv(csv_output, index=False)
        
        os.remove(metrics_output)
        os.remove(metrics_output_master)

if __name__ == '__main__':
    processes = []
    max_process = 5
    print("max_process")
    for i, tag in enumerate(tags):
        print(tag)
        p = Process(target=analyze_version, args=(tag, ))
        processes.append(p)
        p.start()
        
        if len(processes) >= max_process:
            for p in processes:
                p.join()
                processes.remove(p)
                break

    for p in processes:
        p.join()