import pandas as pd
import subprocess
import re


bugs_df = pd.read_csv('./CSV_files/jira_issues.csv')
regex_java_files = r'.*\.java$'
regex_cpp_files = r'.*\.(c\+\+|cpp|cxx|cc|c|hpp|hh|hxx|h\+\+|h)$'
regex_versions = r'.*[2-4]\.[0-9]\.[0-9].*'

results = []

for _, row in bugs_df.iterrows():
    bug_id = row['Key']
    all_affected_versions = row['Affected Versions'].split(", ")

    usefull_affected_versions = ",".join([version for version in all_affected_versions if re.match(regex_versions, version)])
    
    # Recherche commits liés à ce bug
    git_log_cmd = ['git', 'log', '--grep=' + bug_id, '--oneline']

    print(f"Commande exécutée : {git_log_cmd}")
    log_output = subprocess.run(git_log_cmd, capture_output=True, text=True, cwd='../hive')
    print(f"Résultat de git log pour {bug_id} : {log_output.stdout}")



    # Extraire les hash de commit
    commits = [line.split()[0] for line in log_output.stdout.splitlines()]

    print(f"Numéro du commit : {commits}")


    # Parcourir chaque commit pour trouver les fichiers modifiés
    for commit in commits:
        git_show_cmd = f"git show --name-only --oneline {commit}"
        show_output = subprocess.run(git_show_cmd, shell=True, capture_output=True, text=True, cwd='../hive')

        # Ajouter chaque fichier modifié dans les résultats
        files = show_output.stdout.splitlines()
        firstLine = True
        for file in files:
            if (firstLine):
                firstLine = False
            else: 
                if ((re.match(regex_java_files, file) or re.match(regex_cpp_files, file)) and ({'Bug_ID': bug_id, 'File_Path': file, 'Affected Versions' :usefull_affected_versions} not in results)):
                    results.append({'Bug_ID': bug_id, 'File_Path': file, 'Affected Versions': usefull_affected_versions})

results_df = pd.DataFrame(results)
results_df.to_csv('./CSV_files/bug_file_association_2.csv', index=False)