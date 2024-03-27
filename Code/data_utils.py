import pandas as pd
import numpy as np
import json
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack, vstack
import re
from feature_utils import *


def read_conll(file_path):
    """
    Read the conll file and return a dataframe
    """
    data = []
    max_cols = 0
    sent_id = ''
    with open(file_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # grab the sentence id (useful for grouping sentences)
            if line.startswith('# sent_id'):
                sent_id = line.split()[-1]
            # skip comments and empty lines
            if line.startswith('#') or line == '\n':
                continue
            line = line.strip().split('\t')
            # add the sentence id to the beginning of the line
            line.insert(0, sent_id)
            data.append(line)
            max_cols = max(max_cols, len(line)) 
        df = pd.DataFrame(data, columns=range(max_cols))
        df.columns = ['sent_id', 'token_id', 'token', 'lemma',
                      'POS', 'Universal_POS', 'morph_type', 'distance_head',
                      'dep_label', 'dep_rel', 'space'] + list(df.columns[11:])
        return df

def conll_transform(df):
    """
    Transform the conll df by duplicating sentences with more than one predicate,
    such that each sentence has exactly one predicate.
    Add a random number to the 'sent_id' column for each duplicated sentence.
    """
    regex = '.*\.0\d'
    multi_predicate_sentence_ids = df.groupby('sent_id').filter(
        lambda x: x.iloc[:, 11].str.match(regex).sum() > 1)['sent_id'].unique()
    rows_to_concat = []
    for sent_id in multi_predicate_sentence_ids:
        sentence = df[df['sent_id'] == sent_id]
        predicate_count = sentence.iloc[:, 11].str.match(regex).sum()
        for i in range(1, predicate_count):
            sentence_copy = sentence.copy(deep=True)
            sentence_copy.iloc[:, 12] = sentence_copy.iloc[:, 12+i]
            # Add a random number to the 'sent_id' column
            sentence_copy['sent_id'] = sentence_copy['sent_id'] + '_' + str(i) # add the duplication number for easier grouping
            #print(sentence_copy.columns)
            sentence_copy.loc[sentence_copy[12] != 'V', 11] = '_' # set the predicate to '_' for non-predicate words for the subsequent predicates after the first
            rows_to_concat.append(sentence_copy)
    df = pd.concat([df] + rows_to_concat, ignore_index=True)
    df = df.drop(df.columns[13:], axis=1)
    df = df.rename(columns={11: 'predicate', 12: 'argument_type'})
    df.loc[df['argument_type'] != 'V', 'predicate'] = '_' # set the argument type to '_' for non-predicate words for the first predicate
    df = df.fillna('_')
    return df

def prepare_data(file_path):
    """
    Prepare the data by reading the conll file and transforming it and adding features
    """
    df = read_conll(file_path)
    df = conll_transform(df)
    df = bigram_features(df)
    df = trigram_features(df)
    df = get_ner_tags(df)
    df = morph_features(df)
    # feature to indicate if the token is a predicate; maybe redundant
    df['is_token_predicate'] = (df['predicate'] != '_').astype(int)
    # feature for classification task 1: argument identification
    df['is_token_argument'] = (df['argument_type'].str.startswith('ARG')).astype(int)
    # feature for classification task 2: argument classification
    df['argument_label'] = df['argument_type'].apply(lambda x: x if x.startswith('ARG') else 'O')
    return df


def create_count_vectorizer(data, text_feature_columns):
    """
    Create a count vectorizer from the text features in the data
    """
    count_vectorizer = CountVectorizer()
    combined_text = data[text_feature_columns].astype(str).apply(' '.join, axis=1)
    count_vectorizer.fit(combined_text)
    return count_vectorizer

def process_data(data, vectorizer, numeric_features):
    """
    Vectorize the text features and combine with the numeric features
    """
    token_count = vectorizer.transform(data['token'])
    lemma_count = vectorizer.transform(data['lemma'])
    pos_count = vectorizer.transform(data['POS'])
    universal_pos_count = vectorizer.transform(data['Universal_POS'])
    dep_label_count = vectorizer.transform(data['dep_label'])
    dep_rel_count = vectorizer.transform(data['dep_rel'])
    space_count = vectorizer.transform(data['space'])
    predicate_count = vectorizer.transform(data['predicate'])
    ner_count = vectorizer.transform(data['ner'])
    token_bigram_count = vectorizer.transform(data['token_bigram'])
    token_trigram_count = vectorizer.transform(data['token_trigram'])
    pos_bigram_count = vectorizer.transform(data['POS_bigram'])
    pos_trigram_count = vectorizer.transform(data['POS_trigram'])
    
    X = hstack([token_count, lemma_count, pos_count, universal_pos_count, dep_label_count, dep_rel_count, space_count,
                predicate_count, ner_count, data['is_token_predicate'].values.reshape(-1, 1),
                token_bigram_count, token_trigram_count, pos_bigram_count, pos_trigram_count, data[numeric_features].values])
    
    return X


def calculate_metrics(start_label, end_label, y_dev_encoded_array, y_pred):
    """
    Calculate metrics for the classification task
    For labels 0 and 1: is_token_argument (argument identification)
    For labels 2 to 30: argument_label (argument classification)
    """
    weighted_precision_sum = 0
    weighted_recall_sum = 0
    weighted_f1_sum = 0
    total_support = 0
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(start_label, end_label):
        precision, recall, f1, support = precision_recall_fscore_support(y_dev_encoded_array[:, i], y_pred[:, i], average=None, zero_division=0)
        for j in range(len(precision)):
            weighted_precision_sum += precision[j] * support[j]
            weighted_recall_sum += recall[j] * support[j]
            weighted_f1_sum += f1[j] * support[j]
            precisions.append(precision[j])
            recalls.append(recall[j])
            f1_scores.append(f1[j])
        total_support += sum(support)

    weighted_avg_precision = weighted_precision_sum / total_support if total_support > 0 else 0
    weighted_avg_recall = weighted_recall_sum / total_support if total_support > 0 else 0
    weighted_avg_f1 = weighted_f1_sum / total_support if total_support > 0 else 0

    macro_avg_precision = np.mean(precisions)
    macro_avg_recall = np.mean(recalls)
    macro_avg_f1 = np.mean(f1_scores)
    return weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, macro_avg_precision, macro_avg_recall, macro_avg_f1


def save_dict_to_json(data, filename):
    """
    Save a dictionary to a JSON file. This function handles numpy arrays in the dictionary by converting
    them to lists before saving.
    
    :param data: Dictionary to save.
    :param filename: File path for the JSON file to save the dictionary in.
    """
    # Convert numpy arrays to lists
    data_to_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    # Convert nested dictionaries that may have numpy arrays
    for k, v in data_to_save.items():
        if isinstance(v, dict):
            data_to_save[k] = {k1: (v1.tolist() if isinstance(v1, np.ndarray) else v1) for k1, v1 in v.items()}
    
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=4)
        
        f.close()
        

def calculate_classification_metrics(preds, true_labels):
    """
    Calculate precision, recall, f1 score, and macro average metrics for classification results.
    
    Parameters:
    preds: List of list of predictions from token classification
    true_labels: List of list of true labels from token classification
    return: 
    Dictionary with precision, recall, f1 score for each class and macro averages
    """
    # Flatten the predictions and true labels lists
    preds_flat = [p for sublist in preds for p in sublist]
    true_flat = [t for sublist in true_labels for t in sublist]
    
    # Extract unique classes
    classes = sorted(set(true_flat))
    
    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(true_flat, preds_flat, labels=classes)
    
    # Calculate macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    # Create a dictionary to store the metrics
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro': {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro
        }
    }
    
    return metrics


def fix_lists(formatted_df):
    """
    fixing columns of lists where they were read as strings.
    """
    columns_to_convert = ['sentences', 'is_argument', 'arg_labels']

    for column in columns_to_convert:
        formatted_df[column] = formatted_df[column].apply(ast.literal_eval)
        
    return formatted_df





def aggregate_subtoken_logits(tokenized_data, predictions):
    """
    Aggregates subtoken logits to word level for each example in a tokenized dataset.

    Parameters:
    tokenized_data: A list of tokenized data, where each list is a dictionary containing
                               'sentence' and 'word_ids'.
    predictions: A list of subtoken-level predictions, corresponding to the tokenized examples.
                        Each element in the list is an array of logits for an example.

    Returns:
    list: A list of word-level logits for each example. Each element in the list is an array of aggregated logits,
          corresponding to the words in the example.
    """
    word_level_logits = []

    for index, data in enumerate(tokenized_data):
        tokens = data['sentence']
        word_ids = data['word_ids']
        subtoken_logits = np.array(predictions[index])
        current_word_id = None
        current_word_logits = None
        sentence_logits = []

        for subtoken_index, word_id in enumerate(word_ids):
            if word_id is not None and word_id != current_word_id:
                if current_word_logits is not None:
                    sentence_logits.append(current_word_logits)

                current_word_id = word_id
                current_word_logits = subtoken_logits[subtoken_index].copy()
            elif word_id is not None:
                current_word_logits += subtoken_logits[subtoken_index]

        if current_word_logits is not None:
            sentence_logits.append(current_word_logits)

        word_level_logits.append(np.array(sentence_logits))

    return word_level_logits

def align_labels_with_predictions(tokenized_data):
    """
    Aligns original labels with their corresponding word-level predictions in tokenized data.

    Parameters:
    tokenized_data: A list of tokenized examples, where each example is a dictionary containing
                           'word_ids' and 'labels'. 'word_ids' should be a list of word IDs for each subtoken,
                           and 'labels' should be a list of labels for each subtoken.

    Returns:
    list: A list where each element is a list of aligned labels for the words in the corresponding tokenized example.
    """
    aligned_labels = []

    for item in tokenized_data:
        # Extract word IDs and labels, ignoring special tokens at the start and end
        word_ids = item['word_ids'][1:-1]
        original_labels = item['labels'][1:-1]

        # Aggregate labels based on word IDs
        current_word_id = None
        word_labels = []

        for word_id, label in zip(word_ids, original_labels):
            if word_id is not None and word_id != current_word_id:
                # Start of a new word
                word_labels.append(label)
                current_word_id = word_id

        aligned_labels.append(word_labels)

    return aligned_labels

def remove_special_token_indexes(predictions, labels, label_list):
    """
    removes special token indexes from predictions and gold labels
    
    params:
    predictions: list of predictions.
    labels: list of gold labels.
    label_list: list of unique labels
    """

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_predictions, true_labels

def plot_results(path = None):
    '''
    plot the results of the two models

    parms:
    path: saves the plot to the path
    '''
    baseline_results_path = '..\Results\distilbert-base-uncased-baseline-argument-classificationresults-final.json'
    advanced_results_path = '..\Results\distilbert-base-uncased-advanced-argument-classificationresults-final.json'

    with open(baseline_results_path, 'r') as file:
        baseline_results = json.load(file)

    with open(advanced_results_path, 'r') as file:
        advanced_results = json.load(file)

    labels = baseline_results['classes']
    baseline_f1 = baseline_results['f1']
    advanced_f1 = advanced_results['f1']

    f1_differences = [advanced - baseline for advanced, baseline in zip(advanced_f1, baseline_f1)]

    labels, f1_differences

    plt.figure(figsize=(14, 8))
    plt.barh(labels, f1_differences, color='skyblue')
    plt.xlabel('Difference in F1 Score')
    plt.title('Difference in F1 Score per Label Between Two Models')
    plt.axvline(x=0, color='grey', lw=1.5, linestyle='--')
    plt.tight_layout()

    # Show plot 
    plt.xticks(rotation=45)

    # Save plot
    if path: 
        plt.savefig(path + 'f1_differences.png')

    plt.show()
