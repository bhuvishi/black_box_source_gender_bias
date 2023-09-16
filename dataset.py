import datasets
from datasets import load_dataset
from datasets import load_from_disk
from datetime import datetime

import heapq
import os
import pandas as pd
import pickle
from transformers import AutoTokenizer

from collections import defaultdict


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import gender_bias

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


COUNT_OF_WORDS_TO_REMOVE = 10


def tokenize_into_words(text):
    for x in word_tokenize(text):
        yield x.strip()

def get_dictionary(text):
    word_freqs = defaultdict(int)
    word_tokens = tokenize_into_words(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    for word in filtered_sentence:
        word_freqs[word] += 1
    return word_freqs


def get_unique_words(text):
    word_freqs = get_dictionary(text)
    word_freqs_unique = sorted(word_freqs.items(), key=lambda x: x[1])[
        COUNT_OF_WORDS_TO_REMOVE:]
    unique_words = [key for key, count in word_freqs_unique]
    return remove_punctuations(unique_words)


def remove_punctuations(input_list):
    new_list = []
    for word in input_list:
        if word in string.punctuation:
            continue
        new_list.append(word)
    return new_list


class DataMangler(object):
    def __init__(self, dataset_name=None, load_dir_path=None, max_unique_words=1000, vocab_start_index=0):
        if load_dir_path:
            self._load(load_dir_path)
            return

        self._vocab = self._generate_vocabulary(dataset_name, max_unique_words, vocab_start_index)
        print(self._vocab)

        source_dataset = self._get_hf_dataset(dataset_name)
        self._dataset = self._generate_dataset_for_blockbox_inference(source_dataset)
        print(self._dataset)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        dataset_dir_path, vocab_file_path = self._get_save_file_paths(save_dir)
        if not os.path.exists(dataset_dir_path):
            os.mkdir(dataset_dir_path)
        self._dataset.save_to_disk(dataset_dir_path)

        with open(vocab_file_path, 'wb') as f:
            pickle.dump(self._vocab, f)

    def _load(self, load_dir_path):
        dataset_dir_path, vocab_file_path = self._get_save_file_paths(load_dir_path)
        self._dataset = load_from_disk(dataset_dir_path)

        with open(vocab_file_path, 'rb') as f:
            self._vocab = pickle.load(f)

    def _get_save_file_paths(self, save_dir):
        return (os.path.join(save_dir, 'dataset'), os.path.join(save_dir, 'vocab'))
    
    def _generate_dataset_for_blockbox_inference(self, source_dataset):
        return datasets.Dataset.from_pandas(
            pd.DataFrame(self._generate_blackbox_inference_examples(source_dataset)))

    def _generate_blackbox_inference_examples(self, source_dataset):
        for i, example in enumerate(source_dataset):
            if i % 100000 == 0:
                curr_time = datetime.now().strftime("%H:%M:%S")
                print(F'Read {i} texts. Time: {curr_time}')
            if i > 200000:
            # if i > 5:
                break
            # print(example)
            for inference_example in self._get_inference_examples(example):
                # print(inference_example)
                # print('\n\n')
                yield inference_example

    def _get_inference_examples(self, example):
        text = example['text'].lower()
        masked_str_and_targets = gender_bias.get_masked_str_and_targets(
                text)
        for masked_str, targets in masked_str_and_targets:
            for masked_str_with_word_dropped, word in self._get_modified_text_chunk(masked_str):
                yield {'masked_str': masked_str, 'masked_str_with_word_dropped': masked_str_with_word_dropped, 'word': word, 'targets': targets}

    def _get_modified_text_chunk(self, input_text):
        used_words = set()
        for word in tokenize_into_words(input_text):
            if word not in used_words and word.lower() in self._vocab:
                used_words.add(word)
                new_text = gender_bias.remove_words(input_text, word)
                yield (new_text, word)

    def _generate_vocabulary(self, dataset_name, max_unique_words, vocab_start_index):
        word_freqs = defaultdict(int)
        dataset = self._get_hf_dataset(dataset_name)
        print(dataset)
        for i, text in enumerate(dataset['text']):
            text = text.lower()
            if i % 100000 == 0: 
                curr_time = datetime.now().strftime("%H:%M:%S")
                print(F'Read {i} texts. Time: {curr_time}')
            if i > 100000000:
            # if i > 10000:
                break
            for word in tokenize_into_words(text):
                if not self._should_filter_from_vocab(word):
                    word_freqs[word] += 1
        print('Vocab size: ', len(word_freqs))
        word_count_pairs = heapq.nlargest(
            max_unique_words, word_freqs.items(), key=lambda x: x[1])
        return {word: count for word, count in word_count_pairs[vocab_start_index:]}

    def _should_filter_from_vocab(self, word):
        if word in stop_words or word in string.punctuation or word.lower() == 'mask':
            return True
        return False
    
    def _get_hf_dataset(self, dataset_name):
        return load_dataset(dataset_name, keep_in_memory=True, split=datasets.Split.TRAIN)


def main():
    data_mangler = DataMangler(dataset_name='yelp_review_full', max_unique_words=20000, vocab_start_index=1000)
    data_mangler_save_path = '/Users/geetb/fixed_uncased_nlp_out_vocab_1000_20000_dataset_examples_200K/'
    # data_mangler_save_path = '/Users/geetb/test/'
    data_mangler.save(data_mangler_save_path)

    data_mangler_loaded = DataMangler(load_dir_path=data_mangler_save_path)
    print(data_mangler_loaded._vocab)
    print(data_mangler_loaded._dataset)



if __name__ == "__main__":
    main()