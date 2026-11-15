from rag.vector_store import client, COLLECTION_NAME

if not client.collection_exists(COLLECTION_NAME):
    print("La colección todavía no existe.")
else:
    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )

    points = result[0]

    for point in points:
        print("ID:", point.id)
        print("Payload:", point.payload)
        print("-------------------")