# ingest.py (Modified to load multiple JSON files MANUALLY)
import os
import json
import shutil
import glob
from dotenv import load_dotenv
# Removed JSONLoader import, using built-in json library
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import traceback # Import traceback for detailed error printing

# --- Configuration ---
load_dotenv()
SOURCE_DOCUMENTS_DIR = "./docs"
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_ID = "nomic-embed-text"

# --- Helper Functions to Extract Records from Different JSON Structures ---
# These functions replace the jq_schema logic

def extract_records_auxlocks(data: dict) -> list:
    """Extracts product records from AuxLocksData.json structure."""
    records = []
    for key in data:
        if isinstance(data[key], list):
            records.extend(data[key])
    return records

def extract_records_exitdevices(data: dict) -> list:
    """Extracts product records from ExitDeviceData.json structure."""
    records = []
    if "ExitDevices" in data and isinstance(data["ExitDevices"], dict):
        for category in data["ExitDevices"]:
            if isinstance(data["ExitDevices"][category], list):
                records.extend(data["ExitDevices"][category])
    # Optionally handle baseTrims if needed later, but skip for now
    return records

def extract_records_multipoints(data: dict) -> list:
    """Extracts product records from MultiPointsData.json structure."""
    records = []
    if "MultiPoints" in data and isinstance(data["MultiPoints"], dict):
        for category in data["MultiPoints"]:
            if isinstance(data["MultiPoints"][category], list):
                records.extend(data["MultiPoints"][category])
    # Optionally handle baseTrims if needed later, but skip for now
    return records

def extract_records_thermalpin(data: dict) -> list:
    """Extracts product records from ThermalPinData.json structure."""
    records = []
    if "ThermalPin" in data and isinstance(data["ThermalPin"], dict):
         if "Thermal" in data["ThermalPin"] and isinstance(data["ThermalPin"]["Thermal"], list):
              records.extend(data["ThermalPin"]["Thermal"])
    return records

# --- Map filenames to their specific extractor function ---
JSON_EXTRACTORS = {
    "AuxLocksData.json": extract_records_auxlocks,
    "ExitDeviceData.json": extract_records_exitdevices,
    "MultiPointsData.json": extract_records_multipoints,
    "ThermalPinData.json": extract_records_thermalpin,
    # Add other files and their extractor functions here if structures differ
}

# --- Define metadata extraction for JSON ---
# (Same as before)
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["device"] = record.get("device", "")
    metadata["title"] = record.get("title", "")
    metadata["functions"] = record.get("functions", "")
    # Add seq_num if present (though less likely with manual loading)
    if "seq_num" in record.get("__metadata__", {}):
         seq_num_val = record["__metadata__"]["seq_num"]
         metadata["seq_num"] = str(seq_num_val) if not isinstance(seq_num_val, (str, int, float, bool)) else seq_num_val
    return metadata

# --- Define content formatting ---
# (Same as before, ensure it covers all relevant keys from all files)
def format_json_content(record: dict) -> str:
    content_parts = []
    if record.get("device"): content_parts.append(f"Device: {record['device']}")
    if record.get("title"): content_parts.append(f"Title: {record['title']}")
    if record.get("functions"): content_parts.append(f"Functions: {record['functions']}")
    if record.get("MechOptions"): content_parts.append(f"Mechanical Options: {record['MechOptions']}")
    if record.get("ElecOptions"): content_parts.append(f"Electrical Options: {record['ElecOptions']}")
    if record.get("CylOptions"): content_parts.append(f"Cylinder Options: {record['CylOptions']}")
    if record.get("finishes"): content_parts.append(f"Finishes: {record['finishes']}")
    if record.get("warning"): content_parts.append(f"Warning: {record['warning']}")

    # Add template links (link/text, link1/text1, etc.)
    for i in range(1, 6):
        link_key = f"link{i if i > 1 else ''}"
        text_key = f"text{i if i > 1 else ''}"
        if record.get(link_key) and record.get(text_key):
            content_parts.append(f"Template Info: {record[text_key]} - URL: {record[link_key]}")
        elif record.get(link_key):
            content_parts.append(f"Template URL: {record[link_key]}")

    # Add installation links if present
    if "installation" in record and isinstance(record["installation"], list):
         for item in record["installation"]:
             install_title = item.get("title", "Installation Info")
             # Check for primary link/text
             if item.get("link") and item.get("text"):
                  content_parts.append(f"{install_title}: {item['text']} - URL: {item['link']}")
             elif item.get("link"):
                  content_parts.append(f"{install_title} URL: {item['link']}")
             elif item.get("text"):
                  content_parts.append(f"{install_title}: {item['text']}")

             # Check for nested link1/text1 etc. within installation
             for i_inst in range(1, 6):
                  link_key_inst = f"link{i_inst if i_inst > 0 else ''}"
                  text_key_inst = f"text{i_inst if i_inst > 0 else ''}"
                  if item.get(link_key_inst) and item.get(text_key_inst):
                       content_parts.append(f"  - {item[text_key_inst]} - URL: {item[link_key_inst]}")
                  elif item.get(link_key_inst):
                       content_parts.append(f"  - Installation URL: {item[link_key_inst]}")

    return "\n".join(filter(None, content_parts)) # Filter out empty strings

# --- Main Ingestion Logic ---
def main():
    print(f"Loading JSON documents from: {SOURCE_DOCUMENTS_DIR}")
    all_documents = []
    processed_files_count = 0
    skipped_files = []
    error_files = []

    # --- Find all JSON files ---
    json_files = glob.glob(os.path.join(SOURCE_DOCUMENTS_DIR, "**/*.json"), recursive=True)

    if not json_files:
        print(f"No JSON files found in {SOURCE_DOCUMENTS_DIR}")
    else:
        print(f"Found {len(json_files)} JSON files. Attempting to load...")

    # --- Loop through each JSON file and load it MANUALLY ---
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"\n--- Processing: {file_name} ---")

        # Skip files known to be JavaScript, not JSON
        if file_name in ["MortiseLocksData.json", "BoredLocksData.json"]:
            print(f"Skipping file (detected as JavaScript): {file_name}")
            skipped_files.append(file_name)
            continue

        # Get the specific extractor function for this file
        extractor_func = JSON_EXTRACTORS.get(file_name)
        if not extractor_func:
            print(f"Warning: No specific extractor function found for {file_name}. Skipping file.")
            skipped_files.append(file_name)
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract the list of product records using the specific function
            records = extractor_func(data)

            if not records:
                print(f"Warning: Extractor for {file_name} yielded no records.")
                continue

            # Process each record into a Document
            json_documents_for_file = []
            for record in records:
                if not isinstance(record, dict):
                    print(f"Warning: Expected record to be a dict, got {type(record)} in {file_name}. Skipping record.")
                    continue
                try:
                    # Create metadata - start fresh and add source
                    metadata = {}
                    metadata = metadata_func(record, metadata) # Populate from record
                    metadata['source'] = file_path # Set source

                    # Format content
                    content = format_json_content(record)

                    # Create LangChain Document
                    new_doc = Document(page_content=content, metadata=metadata)
                    json_documents_for_file.append(new_doc)

                except Exception as e_proc:
                    print(f"Error processing a record from {file_name}: {e_proc}")
                    traceback.print_exc() # Print detailed traceback for record processing error
                    continue # Skip this record

            all_documents.extend(json_documents_for_file)
            print(f"Loaded and processed {len(json_documents_for_file)} entries from {file_name}.")
            processed_files_count += 1

        except json.JSONDecodeError as e_json:
             print(f"Error: Could not decode JSON from {file_name}: {e_json}")
             error_files.append(file_name)
        except Exception as e_load:
            print(f"Error loading or extracting data from {file_name}: {e_load}")
            traceback.print_exc() # Print detailed traceback for loading/extraction error
            error_files.append(file_name)

    print("\n--- Loading Summary ---")
    print(f"Successfully processed {processed_files_count} JSON files.")
    if skipped_files:
        print(f"Skipped files (JS or no extractor): {', '.join(skipped_files)}")
    if error_files:
        print(f"Files with loading/extraction errors: {', '.join(error_files)}")
    # --- End JSON Loading ---

    if not all_documents:
        print("\nNo documents loaded successfully. Exiting.")
        return

    print(f"\nLoaded a total of {len(all_documents)} processed document entries.")

    # --- Split documents ---
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    print(f"Split into {len(texts)} text chunks.")

    # --- Initialize Ollama embeddings ---
    print(f"Initializing embeddings with model: {EMBEDDING_MODEL_ID}")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_ID)
    except Exception as e:
         print(f"\n--- Error during embedding initialization ---")
         print(f"An error occurred: {e}")
         print("Please ensure Ollama is running and the model '{EMBEDDING_MODEL_ID}' is available.")
         return # Exit if embeddings fail

    # --- Create and persist the vector store ---
    print(f"Creating/updating vector store at: {PERSIST_DIRECTORY}")
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Removing existing vector store at: {PERSIST_DIRECTORY}")
        shutil.rmtree(PERSIST_DIRECTORY)

    try:
        print("Adding documents to ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print("Persisting vector store...")
        vectorstore.persist()
        print(f"\nIngestion complete. Vector store created/updated with data from processed JSON files.")
    except Exception as e:
        print(f"\n--- Error during vector store creation ---")
        print(f"An error occurred: {e}")
        traceback.print_exc() # Print detailed traceback for ChromaDB error
        print("This might be due to issues with metadata types (ensure all are str, int, float, bool),")
        print("Ollama server problems (RAM, model availability), or ChromaDB.")
        print("Please check Ollama server status, logs, and system resources.")

if __name__ == "__main__":
    main()