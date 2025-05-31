import csv
import json

def csv_to_json(csv_file, json_file):

    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]

    with open(json_file, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)

