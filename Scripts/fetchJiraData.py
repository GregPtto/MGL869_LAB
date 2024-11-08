import requests
import csv

# URL de l'API Jira et la requête JQL pour filtrer les problèmes
url_base = "https://issues.apache.org/jira/rest/api/2/search"
query_jql = "project = HIVE AND issuetype = Bug AND (status = Resolved OR status = Closed) AND affectedVersion >= '2.0.0' AND affectedVersion <= '4.0.0'"
resultats_par_page = 1000
debut = 0
list_bugs = []

# Boucle pour récupérer toutes les issues
while True:
    params = {
        'jql': query_jql,
        'startAt': debut,
        'maxResults': resultats_par_page
    }

    reponse = requests.get(url_base, params=params)

    if reponse.status_code != 200:
        print(f"Erreur lors de la récupération des données : {reponse.status_code}")
        break

    data = reponse.json()
    bugs = data['issues']
    list_bugs.extend(bugs)

    if len(bugs) < resultats_par_page:
        break

    debut += resultats_par_page  

# Exportation des données dans un fichier CSV
chemin_fichier = "CSV_files/jira_issues.csv"
with open(chemin_fichier, mode="w", newline="", encoding="utf-8") as fichier_csv:
    writer = csv.writer(fichier_csv)
    writer.writerow(["Key", "Summary", "Status", "Affected Versions"])

    for bug in list_bugs:
        key = bug["key"]
        summary = bug["fields"]["summary"]
        status = bug["fields"]["status"]["name"]
        affected_versions = [version["name"] for version in bug["fields"]["versions"]]

        writer.writerow([key, summary, status, ", ".join(affected_versions)])

print(f"Export terminé ! Les données sont dans '{chemin_fichier}'")
