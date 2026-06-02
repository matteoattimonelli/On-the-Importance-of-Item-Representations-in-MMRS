import os
import re
import csv
import argparse
import logging
import nltk
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple, Set
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from rich.logging import RichHandler
from rich.progress import track

#######################################################################################################################################

# Preprocessing csv structured as:
# Image Name,Keywords
# B0BHTBS5RM,"[Category] {Pet Accessories}, [Pet Type] {Dog}, [Purpose] {Support}, [Material] {Plastic}, [Usage Context] {Home}"
# B0BX76YVP9,"[Category] {Leashes}, [Pet Type] {Dog}, [Purpose] {Training}, [Material] {Leather}, [Usage Context] {Outdoor}."
# B0BM6V2SH8,"[Category] {Dog Treats}, [Pet Type] {Dog}, [Purpose] {Chewing}, [Material] {Meat}, [Usage Context] {Training Aid}"

# Define the labels for the categories
labels = ['category', 'pet type', 'purpose', 'material', 'usage context']


def download_nltk_data():
    """
    Ensures that required NLTK data packages are downloaded.
    """
    nltk_data_packages = ['wordnet']
    for package in nltk_data_packages:
        try:
            nltk.data.find(f'corpora/{package}')
            logging.info(f"NLTK package '{package}' already exists.")
        except LookupError:
            nltk.download(package)
            logging.info(f"Downloaded NLTK package '{package}'.")


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process and cluster keywords from a CSV file.')
    parser.add_argument('--input_file', type=str, default="pets_keywords.csv", required=False,
                        help='Path to the input CSV file.')
    parser.add_argument('--save_lemmatized', action='store_true', help='Save lemmatized keywords and counts.')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2',
                        help='Name of the SentenceTransformer model to use for clustering. Default is "all-MiniLM-L6-v2".')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the model on ("cuda" or "cpu"). Default is "cuda".')
    parser.add_argument('--frequency_threshold', type=int, default=0,
                        help='Frequency threshold for keywords. Set to 0 to cluster all keywords. Default is 0.')
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='Number of clusters for keywords. Default is None. Either Number of clusters or Distance Treshold.')
    parser.add_argument('--distance_threshold', type=float, default=0.45,
                        help='Distance threshold for clustering. Default is 0.2. Either Number of clusters or Distance Treshold')
    parser.add_argument('--category_limit', type=int, default=50,
                        help='Limit of top N keywords per category. Default is 50.')
    parser.add_argument('--remove_duplicates', action='store_true',
                        help='Remove duplicates in clustered keywords per row.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_file = args.input_file
    save_lemmatized = args.save_lemmatized
    model_name = args.model_name
    device = args.device
    frequency_threshold = args.frequency_threshold
    num_clusters = args.num_clusters
    distance_threshold = args.distance_threshold
    category_limit = args.category_limit
    remove_duplicates = args.remove_duplicates

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    # Set up logging
    input_dir = os.path.dirname(os.path.abspath(input_file))
    input_filename = os.path.basename(input_file)
    clustered_filename = f"distanceTresh{distance_threshold}_CategoryLimit{category_limit}_nClusters{num_clusters}_frqTresh{frequency_threshold}_{input_filename}"
    input_name_no_ext = os.path.splitext(clustered_filename)[0]
    timestamp = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    processed_dir = os.path.join(input_dir, f'processed_{timestamp}')
    os.makedirs(processed_dir, exist_ok=True)
    log_file = os.path.join(processed_dir, f'log_{input_name_no_ext}.log')

    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[
            RichHandler(),               # For console output with Rich
            logging.FileHandler(log_file)  # For file logging
        ]
    )

    logging.info(f"Processing input file: {input_file}")

    # Download NLTK data if necessary
    download_nltk_data()

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()



    
    # Define output file paths
    output_file = os.path.join(processed_dir, f'clustered_{clustered_filename}')
    one_hot_encoded_file = os.path.join(processed_dir, f'onehot_{input_name_no_ext}.tsv')
    lemmatized_counts_output_file = os.path.join(processed_dir, f'lemmatized_keyword_counts.csv')

    # ------------------------------#
    # Step 1: Cleaning              #
    # ------------------------------#
    logging.info("Starting CSV cleaning...")
    df_clean = clean_csv(input_file, labels)
    logging.info("CSV cleaning completed.")

    # ------------------------------#
    # Step 2: Lemmatization         #
    # ------------------------------#
    logging.info("Starting lemmatization...")
    if save_lemmatized:
        per_row_df, keyword_counts = lemmatize_keywords(df_clean, lemmatized_counts_output_file, lemmatizer)
    else:
        per_row_df, keyword_counts = lemmatize_keywords(df_clean, None, lemmatizer)
    # Check if per_row_df is None
    if per_row_df is None:
        logging.error("Error in lemmatization.")
        return

    # ------------------------------#
    # Step 3: Load Model            #
    # ------------------------------#
    model = load_model(model_name, device=device)

    # ------------------------------#
    # Step 4: Clustering            #
    # ------------------------------#
    logging.info("Starting clustering of keywords...")
    clustered_keyword_mapping = perform_clustering_with_threshold(
        keyword_counts, model,
        threshold=frequency_threshold, num_clusters=num_clusters,
        distance_threshold=distance_threshold
    )
    logging.info("Keyword clustering completed.")

    # Extract cluster representatives
    cluster_reps = set(clustered_keyword_mapping.values())
    logging.info(f"Cluster representatives extracted: {len(cluster_reps)} representatives.")

    # ------------------------------#
    # Step 5: Clustered Keywords    #
    # ------------------------------#
    per_row_df['Clustered Keywords'] = per_row_df['Lemmatized Keywords'].apply(
        lambda kws: ', '.join([clustered_keyword_mapping.get(kw, kw) for kw in kws.split(', ')])
    )

    # ------------------------------#
    # Step 6: Categorization        #
    # ------------------------------#
    logging.info("Starting categorization of clustered keywords...")
    category_keyword_counts = categorize_and_count_clustered_keywords(
        per_row_df, labels=labels, category_limit=category_limit
    )
    logging.info("Categorization completed.")

    # ------------------------------#
    # Step 7: Final Processing      #
    # ------------------------------#
    logging.info("Generating final clustered keywords and categories...")
    top_keywords_per_category = {label: set(counts.keys()) for label, counts in category_keyword_counts.items()}

    per_row_df, other_count_rows = generate_clustered_keywords_and_categories(
        per_row_df, clustered_keyword_mapping, top_keywords_per_category, labels=labels,
        remove_duplicates=remove_duplicates, cluster_reps=cluster_reps,
        category_keyword_counts=category_keyword_counts, category_limit=category_limit
    )
    logging.info("Final processing completed.")

    # ------------------------------#
    # Step 8: Save Results          #
    # ------------------------------#
    per_row_df.to_csv(output_file, index=False)
    logging.info(f"Process completed. Output saved to '{output_file}'.")

    # ------------------------------#
    # Step 9: One-Hot Encoding      #
    # ------------------------------#
    logging.info("Starting one-hot encoding of categorical features...")
    encode_categorical_features(
        per_row_df, one_hot_encoded_file,
        id_column='Image Name', remove_extension='.jpg',
        labels=labels,
        sep='\t', header=False, index=False,
        top_keywords_per_category=top_keywords_per_category
    )
    logging.info(f"One-hot encoded features saved to '{one_hot_encoded_file}'.")

    # ------------------------------#
    # Step 10: Log Summary of 'Other' Count Rows #
    # ------------------------------#
    logging.info("\nSummary of 'Other' count rows:")
    for count in sorted(other_count_rows.keys()):
        logging.info(f"Rows with {count} 'Other' keywords: {other_count_rows[count]}")

def clean_csv(input_file: str, labels: List[str]) -> pd.DataFrame:
    """
    Cleans the input CSV file and returns a DataFrame.

    Parameters:
        input_file (str): Path to the input CSV file.
        labels (List[str]): List of labels to process.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'Image Name' and 'Keywords' columns.
    """
    logging.info("Starting to clean the CSV file...")

    # Read the CSV file using pandas
    try:
        df = pd.read_csv(input_file, usecols=['Image Name', 'Keywords'], dtype=str)
    except ValueError as e:
        logging.error(f"Error reading CSV file: {e}")
        return pd.DataFrame(columns=['Image Name', 'Keywords'])
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame(columns=['Image Name', 'Keywords'])

    # Drop rows where 'Keywords' is NaN or empty
    df = df.dropna(subset=['Keywords'])
    df = df[df['Keywords'].str.strip().astype(bool)]

    # Clean the 'Keywords' column further
    df = clean_keywords_in_df(df, labels)

    logging.info(f"Total rows processed: {len(df)}")
    logging.info("CSV cleaning completed successfully.")
    return df


def extract_keywords(line: str, labels: List[str]) -> List[str]:
    """
    Extracts keywords from a line.

    Parameters:
        line (str): The line from which to extract keywords.
        labels (List[str]): List of labels to be removed.

    Returns:
        List[str]: List of extracted keywords.
    """
    # Extract keywords between curly braces
    keyword_groups = re.findall(r'\{([^}]+)\}', line)
    extracted_keywords = []
    for group in keyword_groups:
        # Split by ',' to separate multiple keywords
        parts = group.split(',')
        if parts:
            # Take the first keyword
            first_part = parts[0].strip()
            # Further split by '/' if present
            if '/' in first_part:
                first_part = first_part.split('/')[0].strip()
            # Append the processed keyword
            extracted_keywords.append(first_part)
    return extracted_keywords


def parse_keywords_entry(entry: str, labels: List[str]) -> Dict[str, str]:
    """
    Parses a single keyword entry and returns a dictionary mapping labels to values.
    If no label is present, assigns the value to the 'Keywords' field.

    Parameters:
        entry (str): The keyword entry string.
        labels (List[str]): List of expected labels.

    Returns:
        Dict[str, str]: A dictionary with label-value pairs.
    """
    parsed = {}
    labeled_pattern = re.compile(r'\[(?i)(\w+)\]\s*\{([^}]+)\}')  # Matches [Label] {Value}, case-insensitive
    unlabeled_pattern = re.compile(r'\[([^]]+)\]')  # Matches [Value]

    labeled_match = labeled_pattern.match(entry.strip())
    if labeled_match:
        label, value = labeled_match.groups()
        label = label.lower().strip()
        value = value.strip()
        if label in labels:
            parsed[label] = value
        else:
            # If label is not recognized, assign to 'Keywords'
            parsed.setdefault('Keywords', []).append(value)
    else:
        # Attempt to match unlabeled pattern
        unlabeled_match = unlabeled_pattern.match(entry.strip())
        if unlabeled_match:
            value = unlabeled_match.group(1).strip()
            parsed.setdefault('Keywords', []).append(value)
    return parsed


def clean_keywords_in_df(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    """
    Cleans the 'Keywords' column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the keywords to be cleaned.
        labels (List[str]): List of labels to process.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned 'Keywords' column.
    """
    # Remove trailing periods
    df['Keywords'] = df['Keywords'].str.rstrip('.')
    
    # Create a case-insensitive regex pattern for labels
    labels_pattern = r'(?i)\[(' + '|'.join(map(re.escape, labels)) + r')\]\s*{([^}]+)}'
    
    # Remove labels and keep only the keyword values
    df['Keywords'] = df['Keywords'].str.replace(labels_pattern, r'\2', regex=True)
    
    # Remove the '[] {keyword}' patterns
    df['Keywords'] = df['Keywords'].str.replace(r'(?i)\[\]\s*{([^}]+)}', r'\1', regex=True)
    
    # Replace multiple commas and spaces with a single comma and space
    df['Keywords'] = df['Keywords'].str.replace(r'\s*,\s*', ', ', regex=True)
    df['Keywords'] = df['Keywords'].str.replace(r'\s+', ' ', regex=True)
    
    # Trim leading/trailing spaces and commas
    df['Keywords'] = df['Keywords'].str.strip(' ,')
    
    return df

def clean_column(column: pd.Series) -> pd.Series:
    """
    Cleans a pandas Series by removing '[] {}' patterns and unnecessary punctuation.
    
    Parameters:
        column (pd.Series): The Series to be cleaned.
    
    Returns:
        pd.Series: The cleaned Series.
    """
    # Remove trailing periods
    column = column.str.rstrip('.')
    
    # Remove '[] {keyword}' patterns (case-insensitive)
    column = column.str.replace(r'(?i)\[\]\s*{([^}]+)}', r'\1', regex=True)
    
    # Remove '[Label] {keyword}' patterns (case-insensitive)
    column = column.str.replace(r'(?i)\[.*?\]\s*{([^}]+)}', r'\1', regex=True)
    
    # Replace multiple commas and spaces with a single comma and space
    column = column.str.replace(r'\s*,\s*', ', ', regex=True)
    column = column.str.replace(r'\s+', ' ', regex=True)
    
    # Trim leading/trailing spaces and commas
    column = column.str.strip(' ,')
    
    return column

def lemmatize_keywords(df: pd.DataFrame, lemmatized_counts_output_file: str, lemmatizer: WordNetLemmatizer) -> Tuple[pd.DataFrame, Counter]:
    """
    Lemmatizes the keywords in the DataFrame and returns a DataFrame with lemmatized keywords.
    Optionally saves the lemmatized keyword counts to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Image Name' and 'Keywords' columns.
        lemmatized_counts_output_file (str): Path to save the lemmatized keyword counts CSV.
        lemmatizer (WordNetLemmatizer): NLTK WordNetLemmatizer instance.

    Returns:
        Tuple[pd.DataFrame, Counter]: DataFrame with lemmatized keywords and keyword counts.
    """
    logging.info("Starting lemmatization of keywords...")
    per_row_data = []
    keyword_counts = Counter()
    # Iterate through each row to extract and lemmatize keywords
    for index, row in track(df.iterrows(), total=len(df), description="Lemmatizing keywords..."):
        image_name = row.get('Image Name', '')
        keywords_str = row.get('Keywords', '')
        keywords_str = re.sub(r'\[(.*?)\]', r'\1', keywords_str)  # Remove square brackets

        if pd.isna(keywords_str):
            keywords_str = ''
        # Split keywords by comma and strip whitespace
        extracted_keywords = [kw.strip() for kw in keywords_str.split(',')]
        # Lemmatize keywords
        lemmatized_keywords = [lemmatizer.lemmatize(kw.lower()) for kw in extracted_keywords if kw]
        # Update global counts
        keyword_counts.update(lemmatized_keywords)
        # Store per-row data
        per_row_data.append({
            'Image Name': image_name,
            'Keywords': ', '.join(extracted_keywords),
            'Lemmatized Keywords': ', '.join(lemmatized_keywords)
        })
    # Create a DataFrame from per_row_data
    per_row_df = pd.DataFrame(per_row_data)
    if lemmatized_counts_output_file:
        # Write the lemmatized keyword counts to CSV
        lemmatized_counts = [{'Lemmatized Keyword': kw, 'Count': count}
                             for kw, count in keyword_counts.items()]
        lemmatized_counts_df = pd.DataFrame(lemmatized_counts)
        # Sort by Count in descending order
        lemmatized_counts_df = lemmatized_counts_df.sort_values(by='Count', ascending=False)
        # Write to CSV
        try:
            lemmatized_counts_df.to_csv(lemmatized_counts_output_file, index=False)
            logging.info(f"Lemmatized keyword counts saved to '{lemmatized_counts_output_file}'.")
        except Exception as e:
            logging.error(f"Error saving lemmatized keyword counts: {e}")
            raise
    logging.info("Lemmatization completed.")
    return per_row_df, keyword_counts


def load_model(model_name: str, device: str = 'cuda') -> SentenceTransformer:
    """
    Loads a SentenceTransformer model.
    """
    try:
        logging.info(f"Loading SentenceTransformer model '{model_name}' on device '{device}'. This may take a few minutes...")
        model = SentenceTransformer(model_name, device=device)
        logging.info(f"Model '{model_name}' loaded successfully on device '{device}'.")
        return model
    except Exception as e:
        logging.error(f"Error loading model '{model_name}': {e}")
        raise


def perform_clustering_with_threshold(keyword_counts: Counter, model: SentenceTransformer, threshold: int = 0,
                                      num_clusters: int = None, distance_threshold: float = 0.2) -> Dict[str, str]:
    """
    Clusters keywords using hierarchical clustering.

    Parameters:
        keyword_counts (Counter): Counts of lemmatized keywords.
        model (SentenceTransformer): SentenceTransformer model.
        threshold (int): Frequency threshold to separate frequent and infrequent keywords.
                         Set to 0 to cluster all keywords.
        num_clusters (int): Number of clusters to form if distance_threshold is not specified. Default is None.
        distance_threshold (float): The linkage distance threshold above which clusters will not be merged.

    Returns:
        Dict[str, str]: A mapping from keywords to their cluster representatives.
    """
    logging.info(f"Clustering keywords with frequency threshold: {threshold}")
    # Apply frequency threshold
    if threshold > 0:
        filtered_keywords = [kw for kw, count in keyword_counts.items() if count >= threshold]
    else:
        filtered_keywords = list(keyword_counts.keys())
    logging.info(f"Total number of keywords to cluster: {len(filtered_keywords)}")

    # Cluster all keywords and map them to their representatives
    mapping = cluster_keywords(
        filtered_keywords, keyword_counts, model,
        num_clusters=num_clusters, distance_threshold=distance_threshold
    )
    logging.info(f"Number of cluster representatives: {len(set(mapping.values()))}")
    return mapping


def cluster_keywords(keywords: List[str], keyword_counts: Counter, model: SentenceTransformer,
                    num_clusters: int = None, distance_threshold: float = 0.2) -> Dict[str, str]:
    """
    Clusters keywords and maps them to cluster representatives.

    Parameters:
        keywords (List[str]): List of keywords to cluster.
        keyword_counts (Counter): Counts of lemmatized keywords.
        model (SentenceTransformer): SentenceTransformer model.
        num_clusters (int): Number of clusters to form. Default is None.
        distance_threshold (float): The linkage distance threshold above which clusters will not be merged.

    Returns:
        Dict[str, str]: A mapping from each keyword to its cluster representative.
    """
    logging.info("Clustering keywords...")
    if not keywords:
        logging.warning("No keywords provided for clustering.")
        return {}
    
    # Compute embeddings for all keywords
    embeddings = model.encode(keywords, show_progress_bar=True)
    if len(embeddings) == 0:
        logging.warning("No valid embeddings found for clustering.")
        return {}
    # Compute cosine distance matrix
    from sklearn.metrics.pairwise import cosine_distances
    distance_matrix = cosine_distances(embeddings)
    # Perform Hierarchical Clustering using precomputed distances
    if distance_threshold is not None and distance_threshold > 0:
        logging.info(f"Using distance threshold: {distance_threshold}")
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=distance_threshold
        )
    else:
        logging.info(f"Using number of clusters: {num_clusters}")
        clustering_model = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric='precomputed',
            linkage='average'
        )
    labels = clustering_model.fit_predict(distance_matrix)
    logging.info(f"Clustering completed. Number of clusters formed: {len(set(labels))}")
    # Organize keywords by clusters
    clusters = defaultdict(list)
    for label, keyword in zip(labels, keywords):
        clusters[label].append(keyword)
    # Select the most frequent keyword in each cluster as the representative
    representative_keywords = {}
    for label, words in clusters.items():
        most_frequent = max(words, key=lambda w: keyword_counts[w])
        representative_keywords[label] = most_frequent
    # Create a mapping from keyword to representative
    keyword_to_representative = {}
    for label, words in clusters.items():
        rep = representative_keywords[label]
        for word in words:
            keyword_to_representative[word] = rep
    return keyword_to_representative


def categorize_and_count_clustered_keywords(per_row_df: pd.DataFrame, labels: List[str], category_limit: int = 200) -> Dict[str, Counter]:
    """
    Categorizes clustered keywords and counts them per category.

    Parameters:
        per_row_df (pd.DataFrame): DataFrame containing 'Clustered Keywords'.
        labels (List[str]): List of category labels.
        category_limit (int): Limit of top N keywords per category.

    Returns:
        Dict[str, Counter]: Dictionary of keyword counts per category.
    """
    logging.info("Categorizing and counting clustered keywords...")
    category_keyword_counts = {label: Counter() for label in labels}
    for idx, row in per_row_df.iterrows():
        clustered_keywords = row['Clustered Keywords'].split(', ')
        for label, keyword in zip(labels, clustered_keywords):
            if keyword:
                category_keyword_counts[label][keyword] += 1
    # Limit to top N keywords per category
    for label in labels:
        original_count = len(category_keyword_counts[label])
        category_keyword_counts[label] = Counter(dict(category_keyword_counts[label].most_common(category_limit)))
        logging.info(f"Category '{label}': Reduced from {original_count} to {len(category_keyword_counts[label])} keywords (top {category_limit})")
    logging.info("Categorization and counting completed.")
    return category_keyword_counts


def generate_clustered_keywords_and_categories(per_row_df: pd.DataFrame, clustered_keyword_mapping: Dict[str, str],
                                               top_keywords_per_category: Dict[str, Set[str]], labels: List[str],
                                               remove_duplicates: bool = False, cluster_reps: Set[str] = set(),
                                               category_keyword_counts: Dict[str, Counter] = None,
                                               category_limit: int = 200) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Generates the final DataFrame with clustered keywords and category assignments.

    Parameters:
        per_row_df (pd.DataFrame): DataFrame containing 'Lemmatized Keywords'.
        clustered_keyword_mapping (Dict[str, str]): Mapping of keywords to their cluster representatives.
        top_keywords_per_category (Dict[str, Set[str]]): Top keywords per category.
        labels (List[str]): List of category labels.
        remove_duplicates (bool): Whether to remove duplicates in clustered keywords.
        cluster_reps (Set[str]): Set of cluster representative keywords.
        category_keyword_counts (Dict[str, Counter]): Keyword counts per category.
        category_limit (int): The maximum number of keywords allowed per category.

    Returns:
        Tuple[pd.DataFrame, Dict[int, int]]: Updated DataFrame with 'Clustered Keywords' and category columns,
                                            and a dictionary counting how many rows have a specific number of 'Other' keywords.
    """
    logging.info("Generating clustered keywords and assigning categories...")
    # Initialize lists for categories
    category_keywords_list = {label: [] for label in labels}
    clustered_keywords_list = []
    # Initialize a counter for the number of rows with specific "Other" counts
    max_other_count = len(labels)
    other_count_rows = {i: 0 for i in range(max_other_count + 1)}

    for idx, row in per_row_df.iterrows():
        lemmatized_keywords = row['Lemmatized Keywords'].split(', ')
        clustered_keywords = [clustered_keyword_mapping.get(kw, kw) for kw in lemmatized_keywords]
        if remove_duplicates:
            clustered_keywords = list(dict.fromkeys(clustered_keywords))
        clustered_keywords_list.append(', '.join(clustered_keywords))
        
        # Assign keywords to labels
        assigned_keywords = []
        temp_clustered_keywords = clustered_keywords.copy()  # Make a copy to preserve original
        
        for label in labels:
            if temp_clustered_keywords:
                keyword = temp_clustered_keywords.pop(0)
                if keyword not in top_keywords_per_category[label]:
                    keyword = 'Other'
            else:
                keyword = 'Other'
            assigned_keywords.append(keyword)
            
        # Add assigned keywords to categories
        for label, keyword in zip(labels, assigned_keywords):
            category_keywords_list[label].append(keyword)

    # Update DataFrame with final assignments
    per_row_df['Clustered Keywords'] = clustered_keywords_list
    for label in labels:
        per_row_df[label] = category_keywords_list[label]

    # Calculate row-based "Other" counts from final category assignments
    for i in range(len(per_row_df)):
        row_other_count = sum(1 for label in labels if category_keywords_list[label][i] == 'Other')
        if row_other_count <= max_other_count:
            other_count_rows[row_other_count] += 1

    # Log summary of "Other" counts per category
    other_count = {label: 0 for label in labels}
    logging.info("\nSummary of 'Other' counts per category:")
    for label in labels:
        other_count[label] = sum(1 for keyword in category_keywords_list[label] if keyword == 'Other')
        logging.info(f"{label}: {other_count[label]}")

    logging.info("Clustered keywords and categories generated successfully.")
    return per_row_df, other_count_rows


def encode_categorical_features(input_df: pd.DataFrame, output_tsv_path: str, id_column: str = 'Image Name',
                                remove_extension: str = '.jpg', labels: List[str] = None, sep: str = '\t',
                                header: bool = False, index: bool = False,
                                top_keywords_per_category: Dict[str, Set[str]] = None):
    """
    Encodes categorical features using One-Hot Encoding and saves them to a TSV file.

    Parameters:
        input_df (pd.DataFrame): Input DataFrame containing the categorical features.
        output_tsv_path (str): Path to save the one-hot encoded TSV file.
        id_column (str): Name of the identifier column.
        remove_extension (str): Extension to remove from the identifier column values.
        labels (List[str]): List of columns to encode.
        sep (str): Delimiter to use in the TSV file.
        header (bool): Whether to include the header in the TSV file.
        index (bool): Whether to write row names (index).
        top_keywords_per_category (Dict[str, Set[str]]): Top keywords per category to define the encoder categories.
    """
    logging.info("Starting one-hot encoding of categorical features...")
    if labels is None:
        raise ValueError("Labels for encoding must be provided.")
    if top_keywords_per_category is None:
        raise ValueError("Top keywords per category must be provided.")

    # Normalize label cases in top_keywords_per_category
    top_keywords_per_category = {label.lower(): {kw.lower() for kw in kws} for label, kws in top_keywords_per_category.items()}

    # Build categories per label
    categories_per_label = []
    for label in labels:
        categories = list(top_keywords_per_category[label])
        categories = sorted(categories)  # Sorting for consistency
        if 'other' not in categories:
            categories.append('other')  # Ensure 'Other' is included
        categories_per_label.append(categories)

    # Read only the necessary columns
    columns_to_encode = labels
    columns_to_read = [id_column] + columns_to_encode
    df = input_df[columns_to_read]
    
    # Ensure all categorical entries are in lowercase to match encoding
    for label in labels:
        df[label] = df[label].str.lower()

    # Initialize OneHotEncoder with specified categories
    enc = OneHotEncoder(categories=categories_per_label, handle_unknown='ignore', dtype=np.int8)
    
    # Fit and transform the categorical columns
    emb = enc.fit_transform(df[columns_to_encode])
    
    # Retrieve the names of the one-hot encoded features
    feature_names = []
    for label, categories in zip(columns_to_encode, categories_per_label):
        for category in categories:
            feature_names.append(f"{label.capitalize()}_{category}")
    
    # Create a DataFrame with the encoded features
    emb_df = pd.DataFrame(emb.toarray(), columns=feature_names, dtype=np.int8)
    
    # Concatenate the id_column with the one-hot encoded features
    result_df = pd.concat([df[[id_column]].reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    
    # Remove specified extension from id_column
    if remove_extension:
        result_df[id_column] = result_df[id_column].str.replace(remove_extension, '', regex=False)
    
    # Save to TSV file
    try:
        result_df.to_csv(output_tsv_path, index=index, header=header, sep=sep)
        logging.info(f"One-hot encoded features saved successfully to '{output_tsv_path}'.")
    except Exception as e:
        logging.error(f"Error saving one-hot encoded features: {e}")
        raise


if __name__ == "__main__":
    main()