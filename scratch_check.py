import json
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('hoppwhistle-facial-uploads')
blob = bucket.blob('topology/network_graph.json')
data = json.loads(blob.download_as_string())

arnold_id = next((n['id'] for n in data.get('nodes', []) if 'Schwarzenegger' in n.get('name', '')), None)
print(f'Arnold ID: {arnold_id}')

links = data.get('links', [])
arnold_links = [l for l in links if l['source'] == arnold_id or l['target'] == arnold_id]
print(f'Arnold has {len(arnold_links)} links')

for l in arnold_links[:5]:
    other_id = l['source'] if l['target'] == arnold_id else l['target']
    other_node = next((n for n in data['nodes'] if n['id'] == other_id), {})
    print(f"{other_node.get('name')} - {l['value']}")
