from google.cloud import storage

client = storage.Client()
bucket = client.bucket("hoppwhistle-facial-uploads")
blobs = list(bucket.list_blobs(prefix="gallery/"))
print(f"Total blobs in gallery/: {len(blobs)}")
