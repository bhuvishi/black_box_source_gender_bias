import csv
import os
import scores_file_constants


class DivergentWords(object):

    def __init__(self, scores_file1_path, scores_file2_path, out_file_dir):
        scores_dict1 = self._load_data(scores_file1_path)
        scores_dict2 = self._load_data(scores_file2_path)
        if not os.path.exists(out_file_dir):
            os.mkdir(out_file_dir)

        self._out_divergent_scores_with_female_bias(
            scores_dict1, scores_dict2, os.path.join(out_file_dir, 'divergence_under_female_bias'))
        self._out_divergent_scores_with_male_bias(
            scores_dict1, scores_dict2, os.path.join(out_file_dir, 'divergence_under_male_bias'))
        self._out_divergent_scores_with_bias_switched_male_to_female(
            scores_dict1, scores_dict2, os.path.join(out_file_dir, 'bias_switched_male_to_female'))
        self._out_divergent_scores_with_bias_switched_female_to_male(
            scores_dict1, scores_dict2, os.path.join(out_file_dir, 'bias_switched_female_to_male'))

    def _load_data(self, scores_file_path):
        scores_dict = {}
        with open(scores_file_path, 'r', newline='') as scores_file:
            reader = csv.DictReader(f=scores_file, fieldnames=scores_file_constants.FIELD_NAMES)
            next(reader) # skip header
            for row in reader:
                scores_dict[row['word']] = float(row['median'])
        return scores_dict

    def _out_divergent_scores(self, scores_dict1, scores_dict2, out_file_path, predicate):
        scores_with_divergence = []
        for word, score1 in scores_dict1.items():
            score2 = scores_dict2.get(word, None)
            if score2 is None:
                continue
            if predicate(score1, score2):
                scores_with_divergence.append((word, score1 / score2, score1, score2))
        with open(out_file_path, 'w', newline='') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=['word', 'divergence', 'score1', 'score2'])
            writer.writeheader()
            for word, divergence, score1, score2 in sorted(scores_with_divergence, key=lambda x: -x[1]):
                writer.writerow({'word': word, 'divergence': divergence, 'score1': score1, 'score2': score2})

    def _out_divergent_scores_with_female_bias(self, scores_dict1, scores_dict2, out_file_path):
        self._out_divergent_scores(scores_dict1, scores_dict2, out_file_path, lambda x, y: x < 1.0 and y < 1.0)

    def _out_divergent_scores_with_male_bias(self, scores_dict1, scores_dict2, out_file_path):
        self._out_divergent_scores(scores_dict1, scores_dict2, out_file_path, lambda x, y: x > 1.0 and y > 1.0)

    def _out_divergent_scores_with_bias_switched_male_to_female(self, scores_dict1, scores_dict2, out_file_path):
        self._out_divergent_scores(scores_dict1, scores_dict2, out_file_path, lambda x, y: x > 1.0 and y < 1.0)

    def _out_divergent_scores_with_bias_switched_female_to_male(self, scores_dict1, scores_dict2, out_file_path):
        self._out_divergent_scores(scores_dict1, scores_dict2, out_file_path, lambda x, y: x < 1.0 and y > 1.0)

def main():
    scores_file1 = '/Users/sujeetba/play/black_box_source_gender_bias/FinancialBERT400000_scores_vocab_size_1000.csv'
    scores_file2 = '/Users/sujeetba/play/black_box_source_gender_bias/hateBERT200000_scores_vocab_size_1000.csv'
    out_file_dir = '/Users/sujeetba/play/black_box_source_gender_bias/FinancialBERT400000_vs_hateBERT200000_scores_vocab_1000/'
    DivergentWords(scores_file1, scores_file2, out_file_dir)


if __name__ == "__main__":
    main()