use clap::Parser;
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use string_interner::{symbol::SymbolU32, StringInterner};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version = "1.2.0",
    about = "Parses CSV by reading 'doi', 'subfield_path', and 'value' columns based on headers. Counts normalized resourceTypeGeneral/resourceType pairs.",
    long_about = "Reads a CSV file, expecting 'doi', 'subfield_path', and 'value' headers in the first row. \
                  It then processes subsequent rows to find resourceTypeGeneral and resourceType values, \
                  normalizes them, and counts unique pairs. Output is a CSV with these pairs and their counts.")]
struct Cli {
    #[arg(short = 'i', long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short = 'o', long, value_name = "FILE")]
    output: PathBuf,
}

type Interner = StringInterner<SymbolU32>;
type StringId = SymbolU32;

const DOI_HEADER: &str = "doi";
const SUBFIELD_PATH_HEADER: &str = "subfield_path";
const VALUE_HEADER: &str = "value";

const GENERAL_TYPE_PATH: &[u8] = b"attributes.types.resourceTypeGeneral";
const SPECIFIC_TYPE_PATH: &[u8] = b"attributes.types.resourceType";

#[derive(Debug, Default, Clone, Copy)]
struct DoiTypes {
    general: Option<StringId>,
    specific: Option<StringId>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();

    let cli = Cli::parse();
    let input_file_path = &cli.input;
    let output_file_path = &cli.output;

    if !input_file_path.exists() {
        eprintln!(
            "Error: Input file not found: {}",
            input_file_path.display()
        );
        process::exit(1);
    }

    println!(
        "Processing input file: {} (expecting headers: '{}', '{}', '{}')",
        input_file_path.display(),
        DOI_HEADER,
        SUBFIELD_PATH_HEADER,
        VALUE_HEADER
    );
    let (counts, interner) = process_csv(input_file_path)?;
    println!(
        "Finished processing. Found {} unique type pairs.",
        counts.len()
    );
    println!("Processing took: {:.2?}", start_time.elapsed());

    let write_start = Instant::now();
    println!("Writing results to: {}", output_file_path.display());
    write_results_csv(&counts, &interner, output_file_path)?;
    println!(
        "Successfully wrote results to {}",
        output_file_path.display()
    );
    println!("Writing results took: {:.2?}", write_start.elapsed());

    Ok(())
}

fn normalize_type_value(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
}

fn process_csv(
    file_path: &Path,
) -> Result<(HashMap<(Option<StringId>, Option<StringId>), u32>, Interner), Box<dyn Error>> {
    let mut interner = Interner::new();
    let file = File::open(file_path)?;
    let buf_reader = BufReader::with_capacity(1024 * 1024, file);

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(buf_reader);

    let mut headers = StringRecord::new();
    if !rdr.read_record(&mut headers)? {
        return Err("Error: Input CSV is empty or header row could not be read.".into());
    }

    let find_idx = |header_name: &str| {
        headers
            .iter()
            .position(|h| h.trim().to_lowercase() == header_name)
            .ok_or_else(|| Box::<dyn Error>::from(format!("Error: Header '{}' not found in input CSV.", header_name)))
    };

    let doi_idx = find_idx(DOI_HEADER)?;
    let subfield_path_idx = find_idx(SUBFIELD_PATH_HEADER)?;
    let value_idx = find_idx(VALUE_HEADER)?;

    println!(
        "Found column indices: '{}' -> {}, '{}' -> {}, '{}' -> {}",
        DOI_HEADER, doi_idx, SUBFIELD_PATH_HEADER, subfield_path_idx, VALUE_HEADER, value_idx
    );

    let mut doi_data: HashMap<StringId, DoiTypes> = HashMap::new();
    let mut record = StringRecord::new();
    let mut data_rows_read: u64 = 0;

    while rdr.read_record(&mut record)? {
        data_rows_read += 1;

        let doi_str = match record.get(doi_idx) {
            Some(s) if !s.is_empty() => s.trim(),
            _ => {
                eprintln!(
                    "Warning: Skipping data row {} (file line {}) due to missing or empty DOI in column '{}' (index {}).",
                    data_rows_read, data_rows_read + 1,
                    DOI_HEADER, doi_idx
                );
                continue;
            }
        };

        let subfield_path_str = record.get(subfield_path_idx).unwrap_or("").trim();
        let value_str = record.get(value_idx).unwrap_or("").trim();

        let doi_id = interner.get_or_intern(doi_str);
        let entry = doi_data.entry(doi_id).or_insert_with(DoiTypes::default);
        let subfield_path_bytes = subfield_path_str.as_bytes();

        match subfield_path_bytes {
            GENERAL_TYPE_PATH => {
                if !value_str.is_empty() {
                    let normalized_value = normalize_type_value(value_str);
                    if !normalized_value.is_empty() {
                        entry.general = Some(interner.get_or_intern(normalized_value));
                    }
                }
            }
            SPECIFIC_TYPE_PATH => {
                if !value_str.is_empty() {
                    let normalized_value = normalize_type_value(value_str);
                    if !normalized_value.is_empty() {
                        entry.specific = Some(interner.get_or_intern(normalized_value));
                    }
                }
            }
            _ => {}
        }
    }
    println!("Attempted to process {} data rows.", data_rows_read);
    println!("Interned {} unique strings.", interner.len());
    println!("Collected type information for {} unique DOIs.", doi_data.len());

    let count_start = Instant::now();
    let mut type_pair_counts: HashMap<(Option<StringId>, Option<StringId>), u32> = HashMap::new();
    doi_data.values().for_each(|doi_types| {
        let pair = (doi_types.general, doi_types.specific);
        *type_pair_counts.entry(pair).or_insert(0) += 1;
    });
    println!("Counting pairs took: {:.2?}", count_start.elapsed());

    Ok((type_pair_counts, interner))
}

fn write_results_csv(
    counts: &HashMap<(Option<StringId>, Option<StringId>), u32>,
    interner: &Interner,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(output_path)?;
    let writer = BufWriter::with_capacity(1024 * 1024, file);
    let mut wtr = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Necessary)
        .from_writer(writer);

    wtr.write_record(&["resourceTypeGeneral", "resourceType", "count"])?;

    let mut sorted_data = Vec::with_capacity(counts.len());
    for ((general_id_opt, specific_id_opt), count) in counts.iter() {
        let general_str = general_id_opt
            .map(|id| interner.resolve(id).unwrap_or("<ERR: Invalid ID>"))
            .unwrap_or("");
        let specific_str = specific_id_opt
            .map(|id| interner.resolve(id).unwrap_or("<ERR: Invalid ID>"))
            .unwrap_or("");
        sorted_data.push((general_str, specific_str, *count));
    }

    sorted_data.sort_by(|a, b| {
        a.0.cmp(b.0)
            .then_with(|| a.1.cmp(b.1))
            .then_with(|| b.2.cmp(&a.2))
    });

    for (general_str, specific_str, count) in sorted_data {
        wtr.write_record(&[general_str, specific_str, &count.to_string()])?;
    }

    wtr.flush()?;
    Ok(())
}