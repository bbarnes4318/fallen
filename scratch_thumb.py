import json
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('hoppwhistle-facial-uploads')
blob = bucket.blob('topology/network_graph.json')
data = json.loads(blob.download_as_string())

arnold_node = next((n for n in data.get('nodes', []) if 'Schwarzenegger' in n.get('name', '')), {})
print('Arnold thumbnail:', arnold_node.get("thumbnail"))

other_node = next((n for n in data.get('nodes', []) if 'Coria' in n.get('name', '')), {})
print('Coria thumbnail:', other_node.get("thumbnail"))
