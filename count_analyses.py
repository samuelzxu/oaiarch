import os
import json

exp_name = 'experiment_hydro_eng_v4'
anomalies_jsons = list(filter(lambda x: x.endswith(".json"), os.listdir(f"{exp_name}")))
region_keys = set()
for anomaly_json in anomalies_jsons:
    region_name = anomaly_json.split('_')[0]
    region_keys.add(region_name)

region_anomaly_counts = {region: 0 for region in region_keys}
for anomaly_json in anomalies_jsons:
    region_name = anomaly_json.split('_')[0]
    with open(f"{exp_name}/{anomaly_json}", 'r') as f:
        # count the number of instances of the string "description" in the JSON file
        anomaly_data = f.read()
        anomaly_count = anomaly_data.count('"description"')
        region_anomaly_counts[region_name] += anomaly_count
print(f"Anomaly counts for {exp_name}:")
for region, count in region_anomaly_counts.items():
    print(f"{region}: {count} anomalies")
# Average anomaly count
average_anomaly_count = sum(region_anomaly_counts.values()) / len(region_anomaly_counts)
print(f"Average anomaly count: {average_anomaly_count:.2f}")
print(f"Total anomalies: {sum(region_anomaly_counts.values())}")