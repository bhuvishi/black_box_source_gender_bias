import csv
import dataset
import fill_mask
import numpy as np
import os
import sys
import traceback
from datetime import datetime


# from transformers import BertTokenizer
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


class WordsCausingBias(object):

    def __init__(self, model_checkpoint, data_path, batch_size=64):
        self._data_mangler = dataset.DataMangler(load_dir_path=data_path)
        self._unmasker = pipeline(
            'fill-mask',
            model=model_checkpoint,
            # tokenizer=tokenizer,
            #device=1,
            #device_map='auto',
            # batch_size=batch_size,
            #targets=['he', 'she']
            )

        results_by_word = {}
        records_processed = 0
        records_dropped = 0
        #  for i, data_elem in enumerate(self._data_mangler._dataset):
        #      if i > 3:
        #          break
        #      print('word: ', data_elem['word'])
        #      print('masked_str: ', len(data_elem['masked_str']), '   ', data_elem['masked_str'])
        #      print('masked_str_with_word_dropped: ', len(data_elem['masked_str_with_word_dropped']), '   ', data_elem['masked_str_with_word_dropped'])
        #      print('\n\n')


        for i, data_elem in enumerate(self._data_mangler._dataset):
            if i % 10000 == 0:
                curr_time = datetime.now().strftime("%H:%M:%S")
                print('Processed ', records_processed, ' examples and dropped', records_dropped, ' curr_time: ', curr_time)
            if records_processed %100000 == 0:
                print('Outputting statistics after processing ', records_processed, ' records')
                self._output_statistics(records_processed, results_by_word, data_path, model_checkpoint)

            try:
                result = self._unmasker(data_elem['masked_str'], targets=data_elem['targets'])
                result_with_word_dropped = self._unmasker(data_elem['masked_str_with_word_dropped'], targets=data_elem['targets'])

                ratio = fill_mask.get_statistics_from_results(result)
                ratio_with_word_dropped = fill_mask.get_statistics_from_results(result_with_word_dropped)
                results_by_word.setdefault(data_elem['word'], []).append(ratio / ratio_with_word_dropped)
                records_processed += 1
            except Exception:
                records_dropped += 1
                #traceback.print_exc(file=sys.stdout)
                #print('word: [', data_elem['word'], '] masked_str_with_word_dropped: ', len(data_elem['masked_str_with_word_dropped']), '   ', data_elem['masked_str_with_word_dropped'])
                continue
        
        print('Processed records #: ', records_processed, results_by_word)
        self._output_statistics(records_processed, results_by_word, data_path, model_checkpoint)

    def _output_statistics(self, records_processed, results_by_word, data_path, model_checkpoint):
        score_statistics = []
        for word, results in results_by_word.items():
            #if len(results) < 10:
            #    continue
            score_statistics.append({
                'word': word,
                'count': len(results),
                'mean': np.mean(results),
                'median': np.median(results)})

        out_file_path = os.path.join(data_path, model_checkpoint + str(records_processed) + '_scores.csv')
        print('outputting to ', out_file_path)
        with open(out_file_path, 'w', newline='') as out_file:
            score_writer = csv.DictWriter(out_file, fieldnames=['word', 'median', 'mean', 'count'])
            score_writer.writeheader()
            for score_statistic in sorted(score_statistics, key=lambda x: x['median']):
                score_writer.writerow(score_statistic)
                # print(score_statistic)

        #for batch_out in self._unmasker(KeyDataset(self._data_mangler._dataset, "masked_str")):
        #    for result in batch_out:
        #        fill_mask.collect_statistics_from_results(result, score_statistics)
        #return score_statistics


def main():
    data_mangler_save_path = '/Users/geetb/fixed_uncased_nlp_out_vocab_size_1000_dataset_examples_100K/'
    model_checkpoint = 'ahmedrachid/FinancialBERT'
    # tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    WordsCausingBias(model_checkpoint=model_checkpoint, data_path=data_mangler_save_path)

if __name__ == "__main__":
    main()