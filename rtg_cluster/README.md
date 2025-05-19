# resourceType Clustering 

Processes a CSV file containing the distribution of resourceType data in the DataCite data file to cluster around the controlled list of resourceTypeGeneral values using an embeddings model.

## Installation

```bash
pip instal -r requirements.txt
```

## Usage

```bash
python rtg_cluster.py -i <input_csv> -o <output_csv> -s <summary_csv> [--model MODEL] [--distance_threshold THRESHOLD]
```

### Parameters

- `-i, --input`: Path to input CSV file (with columns: resourceTypeGeneral, resourceType, count)
- `-o, --output`: Path for detailed output CSV file
- `-s, --summary_output`: Path for summary output CSV file
- `--model`: Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `--distance_threshold`: Maximum cosine distance threshold (default: 0.4, meaning similarity >= 0.6)


## Output

- CSV with original data plus assigned categories
- Summary CSV with counts per category and clustered resource types