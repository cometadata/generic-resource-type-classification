# ResourceType ResourceType General Pair Counts

Counts unique pairs of normalized `resourceTypeGeneral` and `resourceType` values from a CSV input dervied from the DataCite data file using [fast-field-parser](https://github.com/adambuttrick/datacite-utils/tree/main/fast-field-parser).

## Compilation

```bash
cargo build --release
````

## Usage

```bash
./target/release/resourcetype_resourcetypegeneral_count -i <input.csv> -o <output_counts.csv>
```

### Parameters

  - `-i, --input <FILE>`: Input CSV file. Must contain headers: `doi`, `subfield_path`, `value`. The script processes `value`s where `subfield_path` is `attributes.types.resourceTypeGeneral` or `attributes.types.resourceType`. These values are normalized (trimmed, lowercased, alphanumeric/whitespace only).
  - `-o, --output <FILE>`: Output CSV file.

## Output

  - A CSV file with columns: `resourceTypeGeneral`, `resourceType`, `count`, sorted by `resourceTypeGeneral` (asc), then `resourceType` (asc), then `count` (desc).
  - Empty strings denote missing general or specific types for a DOI's data.