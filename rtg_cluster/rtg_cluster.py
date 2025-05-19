import os
import re
import nltk
import argparse
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def download_nltk_resources():
    print(f"--- Diagnostics from script ---")
    print(f"NLTK version: {nltk.__version__}")
    user_nltk_data_path = os.path.expanduser('~/nltk_data')
    if user_nltk_data_path not in nltk.data.path:
        print(f"Manually appending to NLTK path: {user_nltk_data_path}")
        nltk.data.path.append(user_nltk_data_path)

    print(f"NLTK data search path: {nltk.data.path}")

    resources = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt")
    ]
    all_resources_available = True
    for path_suffix, resource_name in resources:
        print(f"Checking for NLTK resource: '{resource_name}' (expected at suffix: '{path_suffix}')")
        try:
            nltk.data.find(path_suffix)
            print(f"NLTK resource '{resource_name}' FOUND.")
        except LookupError:
            print(f"NLTK resource '{resource_name}' NOT FOUND (LookupError for '{path_suffix}').")
            print(f"Attempting download for '{resource_name}'...")
            try:
                nltk.download(resource_name, quiet=True)
                nltk.data.find(path_suffix)
                print(f"NLTK resource '{resource_name}' downloaded successfully.")
            except Exception as e:
                print(f"Error during download/verification attempt for '{resource_name}': {e}")
                all_resources_available = False
                if resource_name in ["wordnet", "stopwords", "punkt"]:
                    print(f"CRITICAL: NLTK resource '{resource_name}' is still unavailable.")
        except Exception as e_find:
            print(f"An unexpected error occurred while trying to find NLTK resource '{resource_name}': {e_find}")
            all_resources_available = False

    if all_resources_available:
        print("All specified NLTK resources appear to be available.")
    else:
        print("One or more NLTK resources are NOT available.")
    print(f"--- End Diagnostics from script ---")
    return all_resources_available


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Assign resource types to fixed broad categories using embeddings and generate a summary file.")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input CSV file.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the output CSV file for detailed categorized results.")
    parser.add_argument("-s", "--summary_output", required=True,
                        help="Path to the output CSV file for category summaries.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name.")
    parser.add_argument("--distance_threshold", type=float, default=0.4,
                        help="Maximum cosine distance for a resource type to be mapped to a fixed category (e.g., 0.4 means similarity >= 0.6). Lower distance implies higher similarity required.")
    return parser.parse_args()


# Very naive form of classification, but given resourceType values, sufficient for now
def is_likely_english_heuristic(text):
    if not text or not isinstance(text, str) or not text.strip():
        return False
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        return False
    if not any(char.isalpha() for char in text):
        return False
    return True


def preprocess_text_enhanced(text, lemmatizer, stop_words_set):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if len(
        word) > 1 and word not in stop_words_set]
    return " ".join(tokens)


FIXED_LABELS = [
    "audiovisual", "award", "book", "book chapter", "collection", "computational notebook",
    "conference paper", "conference proceeding", "data paper", "dataset", "dissertation", "event",
    "image", "interactive resource", "instrument", "journal", "journal article", "model",
    "output management plan", "peer review", "physical object", "preprint", "project", "report",
    "service", "software", "sound", "standard", "study registration", "workflow"
]
UNCATEGORIZED_LABEL = "Uncategorized"


def main(input_file, output_file, summary_output_file, model_name, distance_threshold):
    if not download_nltk_resources():
        print("Critical NLTK resources are unavailable. Exiting.")
        return
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully read input file: {input_file}")
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}")
        return

    required_cols = ['resourceTypeGeneral', 'resourceType',
                     'count']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input CSV must contain {required_cols} columns.")
        return

    print("Original DataFrame sample:")
    print(df.head())
    print("-" * 30)

    df['assignedBroadCategory'] = UNCATEGORIZED_LABEL

    df['IsLikelyEnglish'] = df['resourceType'].apply(
        is_likely_english_heuristic)

    processable_df = df[df['IsLikelyEnglish']
                        & df['resourceType'].notna()].copy()
    print(f"\nFound {len(processable_df)} likely English, non-null ResourceType entries for potential processing.")

    if not processable_df.empty:
        lemmatizer = WordNetLemmatizer()
        # Set to empty for now as there was no performance difference observed, but if we want to use English stopwords, uncomment
        # stop_words_set = set(stopwords.words('english'))
        stop_words_set = set()

        print("Applying enhanced preprocessing to ResourceTypes...")
        processable_df['processedResourceType'] = processable_df['resourceType'].apply(
            lambda text: preprocess_text_enhanced(
                text, lemmatizer, stop_words_set)
        )

        processable_df = processable_df[processable_df['processedResourceType'].str.strip(
        ) != '']
        print(f"{len(processable_df)} entries remaining after preprocessing.")

        if not processable_df.empty and processable_df['processedResourceType'].nunique() > 0:
            unique_processed_resource_types = processable_df['processedResourceType'].unique(
            ).tolist()
            print(f"Found {len(unique_processed_resource_types)} unique PROCESSED resource types for embedding.")

            print("Preprocessing fixed category labels...")
            processed_fixed_labels_dict = {
                label: preprocess_text_enhanced(label, lemmatizer, stop_words_set)
                for label in FIXED_LABELS
            }
            active_processed_fixed_labels_list = [
                pl for pl in processed_fixed_labels_dict.values() if pl]
            original_labels_for_active_processed = [
                label for label, pl_label in processed_fixed_labels_dict.items() if pl_label in active_processed_fixed_labels_list
            ]

            processed_fixed_labels_for_embedding = [
                processed_fixed_labels_dict[fl] for fl in FIXED_LABELS]

            if not any(processed_fixed_labels_for_embedding):
                print(
                    "Error: All fixed labels became empty after preprocessing. Cannot proceed.")
                df['assignedBroadCategory'] = UNCATEGORIZED_LABEL
            else:
                print(f"Loading sentence transformer model: {model_name}")
                try:
                    model = SentenceTransformer(model_name)

                    print("Generating embeddings for processed resource types...")
                    resource_embeddings = model.encode(
                        unique_processed_resource_types, show_progress_bar=True)

                    print("Generating embeddings for processed fixed labels...")
                    fixed_label_embeddings = model.encode(
                        processed_fixed_labels_for_embedding, show_progress_bar=True)

                    if isinstance(resource_embeddings, np.ndarray) and resource_embeddings.shape[0] > 0 and \
                       isinstance(fixed_label_embeddings, np.ndarray) and fixed_label_embeddings.shape[0] > 0:

                        similarity_threshold = 1.0 - distance_threshold
                        print(f"\nCalculating similarities and assigning to fixed categories (similarity threshold: >={similarity_threshold:.2f})...")

                        similarities = cosine_similarity(
                            resource_embeddings, fixed_label_embeddings)

                        processed_rt_to_assigned_category = {}
                        for i, proc_res_type in enumerate(unique_processed_resource_types):
                            max_similarity_idx = np.argmax(similarities[i])
                            max_similarity_val = similarities[i][max_similarity_idx]

                            if max_similarity_val >= similarity_threshold:
                                if processed_fixed_labels_for_embedding[max_similarity_idx]:
                                    assigned_label = FIXED_LABELS[max_similarity_idx]
                                    processed_rt_to_assigned_category[proc_res_type] = assigned_label
                                else:
                                    processed_rt_to_assigned_category[proc_res_type] = UNCATEGORIZED_LABEL
                            else:
                                processed_rt_to_assigned_category[proc_res_type] = UNCATEGORIZED_LABEL

                        processable_df['assignedBroadCategory'] = processable_df['processedResourceType'].map(
                            processed_rt_to_assigned_category)

                        df.loc[processable_df.index,
                               'assignedBroadCategory'] = processable_df['assignedBroadCategory']

                        num_mapped_successfully = (
                            df['assignedBroadCategory'] != UNCATEGORIZED_LABEL).sum()
                        print(f"Assigned categories for {num_mapped_successfully} resource types based on similarity.")

                    else:
                        print(
                            "\nEmbeddings matrix for resource types or fixed labels is not valid. All processable items will be marked Uncategorized.")
                        df.loc[processable_df.index,
                               'assignedBroadCategory'] = UNCATEGORIZED_LABEL
                except Exception as e:
                    print(f"Error during embedding or category assignment: {e}. All processable items will be marked Uncategorized.")
                    df.loc[processable_df.index,
                           'assignedBroadCategory'] = UNCATEGORIZED_LABEL
        else:
            print("No unique processed resource types to categorize after filtering. All processable items will be marked Uncategorized.")
    else:
        print(f"No likely English resource types found for processing. All items will remain as '{UNCATEGORIZED_LABEL}'.")

    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_output_cols = ['resourceTypeGeneral', 'resourceType',
                          'count', 'assignedBroadCategory', 'IsLikelyEnglish']
        for col in df_output_cols:
            if col not in df.columns:
                df[col] = None if col != 'IsLikelyEnglish' else False
        df.to_csv(output_file, index=False, columns=df_output_cols)
        print(f"\nSuccessfully wrote detailed categorized data to '{os.path.abspath(output_file)}'")
    except Exception as e:
        print(f"\nError writing detailed CSV '{output_file}': {e}")

    print("\nGenerating category summary...")
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)

    if not df.empty:
        def get_category_summary_details(group):
            total_count = group['count'].sum()
            types_list = group['resourceType'].dropna().astype(
                str).unique().tolist()
            included_types_str = ';'.join(sorted(types_list))
            return pd.Series({'totalCount': total_count, 'includedResourceTypes': included_types_str})

        try:
            category_summary = df.groupby('assignedBroadCategory').apply(
                get_category_summary_details, include_groups=False).reset_index()
            category_summary.rename(
                columns={'assignedBroadCategory': 'broadCategoryLabel'}, inplace=True)

            all_expected_labels = FIXED_LABELS[:]
            if UNCATEGORIZED_LABEL in category_summary['broadCategoryLabel'].unique() and UNCATEGORIZED_LABEL not in all_expected_labels:
                all_expected_labels.append(UNCATEGORIZED_LABEL)

            expected_labels_df = pd.DataFrame(
                {'broadCategoryLabel': all_expected_labels})
            category_summary = pd.merge(expected_labels_df, category_summary, on='broadCategoryLabel', how='left').fillna({
                'totalCount': 0,
                'includedResourceTypes': ''
            })
            category_summary['totalCount'] = category_summary['totalCount'].astype(
                int)

            category_summary['broadCategoryLabel'] = pd.Categorical(
                category_summary['broadCategoryLabel'], categories=all_expected_labels, ordered=True)
            category_summary = category_summary.sort_values(
                'broadCategoryLabel')

            print(f"Generated summary for {len(category_summary)} categories.")
            try:
                summary_output_dir = os.path.dirname(summary_output_file)
                if summary_output_dir and not os.path.exists(summary_output_dir):
                    os.makedirs(summary_output_dir)
                summary_cols_order = ['broadCategoryLabel',
                                      'totalCount', 'includedResourceTypes']
                category_summary.to_csv(
                    summary_output_file, index=False, columns=summary_cols_order)
                print(f"Successfully wrote category summary data to '{os.path.abspath(summary_output_file)}'")
            except Exception as e:
                print(f"\nError writing summary CSV '{summary_output_file}': {e}")
        except Exception as e_agg:
            print(f"\nError during summary aggregation: {e_agg}")
    else:
        print("DataFrame is empty, cannot generate summary.")

    print("\nFinal DataFrame sample with Assigned Categories (first 5 rows):")
    print(df[['resourceType', 'assignedBroadCategory', 'count']].head())
    print("-" * 30)
    print(f"Detailed results saved to: '{os.path.abspath(output_file)}'")
    print(f"Category summary saved to: '{os.path.abspath(summary_output_file)}'")


if __name__ == "__main__":
    args = parse_arguments()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    summary_output_path = os.path.abspath(args.summary_output)

    print(f"Running script from: {os.getcwd()}")
    print(f"Input file resolved to: {input_path}")
    print(f"Output file (detailed) resolved to: {output_path}")
    print(f"Output file (summary) resolved to: {summary_output_path}")
    print(f"Using sentence transformer model: {args.model}")
    print(f"Using similarity mapping with cosine distance threshold: {args.distance_threshold} (similarity >= {1.0 - args.distance_threshold:.2f})")

    main(input_path, output_path, summary_output_path,
         args.model, args.distance_threshold)
