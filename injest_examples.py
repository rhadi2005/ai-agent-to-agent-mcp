#!/usr/bin/env python3

import os
import sys
import argparse
from typing import List, Dict

import vertexai
from vertexai.preview import example_stores
from vertexai.generative_models import Content, Part
from dotenv import load_dotenv

EXAMPLE_STORE_DISPLAY_NAME = "sql_agent_few_shot_examples"
GOOGLE_CLOUD_LOCATION = "us-central1"

load_dotenv()

FEW_SHOT_EXAMPLES = [
    {
        "user_query": "What is the name of the artist with ID 5?",
        "sql_query": "SELECT Name FROM Artist WHERE ArtistId = 5;",
        "schema": """Table: Artist
Columns: ArtistId (INTEGER PRIMARY KEY AUTOINCREMENT), Name (NVARCHAR(120))"""
    },
    {
        "user_query": "List all tracks in the 'Rock' genre.",
        "sql_query": "SELECT T.Name FROM Track T JOIN Genre G ON T.GenreId = G.GenreId WHERE G.Name = 'Rock' LIMIT 10;",
        "schema": """Table: Track
Columns: TrackId (INTEGER PK), Name (NVARCHAR(200)), GenreId (INTEGER), ... FK(GenreId) REFERENCES Genre(GenreId)
Table: Genre
Columns: GenreId (INTEGER PK), Name (NVARCHAR(120))"""
    },
    {
        "user_query": "Which 3 customers have spent the most?",
        "sql_query": """SELECT C.FirstName, C.LastName, SUM(I.Total) AS TotalSpent
FROM Customer C
JOIN Invoice I ON C.CustomerId = I.CustomerId
GROUP BY C.CustomerId
ORDER BY TotalSpent DESC
LIMIT 3;""",
        "schema": """Table: Customer
Columns: CustomerId (INTEGER PK), FirstName, LastName, ...
Table: Invoice
Columns: InvoiceId (INTEGER PK), CustomerId (INTEGER), Total (NUMERIC), ... FK(CustomerId) REFERENCES Customer(CustomerId)"""
    },
    {
        "user_query": "How many albums does the band 'AC/DC' have?",
        "sql_query": """SELECT COUNT(Al.AlbumId)
FROM Album Al
JOIN Artist Ar ON Al.ArtistId = Ar.ArtistId
WHERE Ar.Name = 'AC/DC';""",
        "schema": """Table: Album
Columns: AlbumId (INTEGER PK), Title, ArtistId (INTEGER), ... FK(ArtistId) REFERENCES Artist(ArtistId)
Table: Artist
Columns: ArtistId (INTEGER PK), Name"""
    },
    {
        "user_query": "Find the email address of the customer named Jennifer Peterson.",
        "sql_query": "SELECT Email FROM Customer WHERE FirstName = 'Jennifer' AND LastName = 'Peterson';",
        "schema": """Table: Customer
Columns: CustomerId (INTEGER PK), FirstName, LastName, Email"""
    },
    {
        "user_query": "List all employees who are Sales Support Agents.",
        "sql_query": "SELECT FirstName, LastName FROM Employee WHERE Title = 'Sales Support Agent';",
        "schema": """Table: Employee
Columns: EmployeeId (INTEGER PK), LastName, FirstName, Title"""
    },
    {
        "user_query": "What are the names and composers of tracks on the album 'Let There Be Rock'?",
        "sql_query": """SELECT T.Name, T.Composer
FROM Track T
JOIN Album Al ON T.AlbumId = Al.AlbumId
WHERE Al.Title = 'Let There Be Rock'
LIMIT 10;""",
        "schema": """Table: Track
Columns: TrackId (INTEGER PK), Name, AlbumId (INTEGER), Composer, ... FK(AlbumId) REFERENCES Album(AlbumId)
Table: Album
Columns: AlbumId (INTEGER PK), Title"""
    },
    {
        "user_query": "Which sales agent made the most in sales in 2009?",
        "sql_query": """SELECT e.FirstName, e.LastName, SUM(i.Total) as TotalSales
FROM Employee e
JOIN Customer c ON e.EmployeeId = c.SupportRepId
JOIN Invoice i ON c.CustomerId = i.CustomerId
WHERE i.InvoiceDate LIKE '2009%'
AND e.Title = 'Sales Support Agent'
GROUP BY e.EmployeeId
ORDER BY TotalSales DESC
LIMIT 1;""",
        "schema": """Table: Employee 
Columns: EmployeeId (INTEGER PK), LastName, FirstName, Title
Table: Customer
Columns: CustomerId (INTEGER PK), SupportRepId (INTEGER FK), ... FK(SupportRepId) REFERENCES Employee(EmployeeId)
Table: Invoice
Columns: InvoiceId (INTEGER PK), CustomerId (INTEGER FK), InvoiceDate (DATETIME), Total (NUMERIC), ... FK(CustomerId) REFERENCES Customer(CustomerId)"""
    },
    {
        "user_query": "What is the total number of tracks in each playlist?",
        "sql_query": """SELECT p.Name, COUNT(pt.TrackId) AS TrackCount
FROM Playlist p
JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId
GROUP BY p.PlaylistId
ORDER BY TrackCount DESC;""",
        "schema": """Table: Playlist
Columns: PlaylistId (INTEGER PK), Name (NVARCHAR(120))
Table: PlaylistTrack
Columns: PlaylistId (INTEGER PK), TrackId (INTEGER PK), ... FK(PlaylistId) REFERENCES Playlist(PlaylistId), FK(TrackId) REFERENCES Track(TrackId)"""
    },
    {
        "user_query": "What is the average track length by genre?",
        "sql_query": """SELECT g.Name AS Genre, AVG(t.Milliseconds)/1000 AS AvgLengthSeconds
FROM Track t
JOIN Genre g ON t.GenreId = g.GenreId
GROUP BY g.GenreId
ORDER BY AvgLengthSeconds DESC;""",
        "schema": """Table: Track
Columns: TrackId (INTEGER PK), GenreId (INTEGER FK), Milliseconds (INTEGER), ... FK(GenreId) REFERENCES Genre(GenreId)
Table: Genre
Columns: GenreId (INTEGER PK), Name (NVARCHAR(120))"""
    }
]


def setup_auth():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        print(
            f"Found GOOGLE_APPLICATION_CREDENTIALS environment variable: {creds_path}")
        return

    print("No explicit GOOGLE_APPLICATION_CREDENTIALS found.")
    print("Checking if Application Default Credentials are available...")

    try:
        vertexai.init(project=os.getenv(
            "GOOGLE_CLOUD_PROJECT", ""), location="us-central1")
        print("Successfully authenticated with Application Default Credentials")
    except Exception as e:
        print(f"Authentication error: {e}")
        print("\nPlease authenticate using one of these methods:")
        print("1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to a service account key")
        print("2. Run 'gcloud auth application-default login' to use your user credentials")
        raise


def create_or_get_example_store(embedding_model: str = "text-embedding-004") -> str:
    example_store_name = EXAMPLE_STORE_DISPLAY_NAME

    try:
        example_store = example_stores.ExampleStore(
            example_store_name=os.getenv("EXAMPLE_STORE"))
        print(
            f"Found existing Example Store from environment variable: {example_store_name}")
        return example_store
    except Exception as e:
        print(f"Error finding Example Store {example_store_name}: {e}")
        example_store_name = None

    try:
        print(f"Creating new Example Store: {example_store_name}")
        new_store = example_stores.ExampleStore.create(
            display_name=example_store_name,
            example_store_config=example_stores.ExampleStoreConfig(
                vertex_embedding_model=embedding_model
            )
        )
        print(f"Created new Example Store with name: {example_store_name}")
        return new_store
    except Exception as e:
        print(f"Error creating Example Store: {e}")
        raise


def ingest_examples(example_store: any, examples: List[Dict], force_reingest: bool = False) -> bool:
    try:
        if not force_reingest:
            try:
                existing_examples = example_store.fetch_examples()
                print(existing_examples)

                if isinstance(existing_examples, dict) and 'results' in existing_examples:
                    if len(existing_examples['results']) > 0:
                        print(
                            f"Example Store {EXAMPLE_STORE_DISPLAY_NAME} already has examples. Use --force to reingest.")
                        return False
                elif hasattr(existing_examples, '__iter__') and len(list(existing_examples)) > 0:
                    print(
                        f"Example Store {EXAMPLE_STORE_DISPLAY_NAME} already has examples. Use --force to reingest.")
                    return False

                print("No existing examples found. Proceeding with ingestion.")

            except Exception as e:
                print(f"Warning: Could not check existing examples: {e}")

        formatted_examples = []
        for example in examples:
            user_content = Content(
                role="user",
                parts=[Part.from_text("user_query")]
            )
            assistant_content = Content(
                role="assistant",
                parts=[Part.from_text(example['sql_query'])]
            )
            print(user_content)
            print(assistant_content)

            formatted_example = {
                "contents_example": {
                    "contents": [user_content.to_dict()],
                    "expected_contents": [
                        {"content": assistant_content.to_dict()}
                    ]
                },
                "search_key": example["user_query"]
            }

            formatted_examples.append(formatted_example)

        batch_size = 5
        success_count = 0

        for i in range(0, len(formatted_examples), batch_size):
            batch = formatted_examples[i:i + batch_size]

            try:
                print(
                    f"Uploading batch {i//batch_size + 1} with {len(batch)} examples...")
                example_store.upsert_examples(examples=batch)
                success_count += len(batch)
                print(f"Successfully uploaded batch {i//batch_size + 1}")
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {e}")
                print(f"Details: {e}")

        print(
            f"Successfully uploaded {success_count} out of {len(formatted_examples)} examples")
        return success_count > 0

    except Exception as e:
        print(f"Error during example ingestion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Ingest SQL examples into Vertex AI Example Store.")
    parser.add_argument("--project", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT"),
                        help="Google Cloud Project ID")
    parser.add_argument("--location", type=str, default=GOOGLE_CLOUD_LOCATION)
    parser.add_argument("--store-name", type=str, default=EXAMPLE_STORE_DISPLAY_NAME,
                        help="Display name for the Example Store")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-004",
                        help="Embedding model to use (textembedding-gecko@003, text-embedding-004, or text-multilingual-embedding-002)")
    parser.add_argument("--force", action="store_true",
                        help="Force reingestion even if store already has examples")

    args = parser.parse_args()

    if not args.project:
        print("Error: Google Cloud Project ID is required.")
        print("Either set GOOGLE_CLOUD_PROJECT environment variable or use --project flag.")
        return 1

    try:
        setup_auth()
    except Exception as e:
        print(f"Authentication verification failed: {e}")
        return 1

    vertexai.init(project=args.project, location=args.location)
    print(
        f"Initialized Vertex AI with project {args.project} in {args.location}")

    examples = FEW_SHOT_EXAMPLES
    print(f"Using {len(examples)} examples for ingestion")

    example_store = create_or_get_example_store(args.embedding_model)

    if not example_store:
        print("Failed to create or get Example Store.")
        return 1

    success = ingest_examples(
        example_store=example_store,
        examples=examples,
        force_reingest=args.force
    )

    if success:
        print("\nSuccessfully ingested examples.")
        print(f"Example Store resource name: {EXAMPLE_STORE_DISPLAY_NAME}")
        print(
            f"Set this environment variable to use the Example Store with your SQL agent:")
        return 0
    else:
        print("\nIngestion process not started or failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
