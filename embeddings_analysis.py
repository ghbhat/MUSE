from src.utils import read_embeddings
from word_translation_modified import *
import argparse
import pandas
import torch
import src.evaluation.word_translation

parser = argparse.ArgumentParser()
# parser.add_argument('--en_embedding_file', default="/projects/tir1/users/gbhat/data/fasttext/wiki.en.vec")
# parser.add_argument('--ru_embedding_file', default="/projects/tir1/users/gbhat/data/fasttext/wiki.ru.vec")
# parser.add_argument('--en_embedding_file', \
#     default="/projects/tir1/users/gbhat/data/muse-data/multilingual_vectors/wiki.multi.en.vec")
# parser.add_argument('--ru_embedding_file', \
#     default="/projects/tir1/users/gbhat/data/muse-data/multilingual_vectors/wiki.multi.ru.vec")
parser.add_argument('--en_embedding_file', \
    default="/projects/tir1/users/gbhat/work/muse_experiments/vectors-en.txt")
parser.add_argument('--ru_embedding_file', \
    default="/projects/tir1/users/gbhat/work/muse_experiments/vectors-ru.txt")
parser.add_argument('--loanwords_file', \
    default="/projects/tir1/users/gbhat/data/russian/en-ru/en_ru_loanwords.csv")
parser.add_argument('--supervised_dict_file', \
    default="/projects/tir1/users/gbhat/data/muse-data/crosslingual/dictionaries/en-ru.0-5000.txt")
parser.add_argument('--flip_dict', action='store_true')
parser.add_argument('--vocab_size', default=1e9)
parser.add_argument('--knn_method', default='csls_knn_10')


#totally unnecessary, load_dictionary does all this analysis
def count_borrowed_words_in_embeddings(args):
    # loading data
    borrowed_pairs = pandas.read_csv(args.loanwords_file)
    en_word2id, en_embeddings = read_embeddings(args.en_embedding_file, n_max=args.vocab_size)
    ru_word2id, ru_embeddings = read_embeddings(args.ru_embedding_file, n_max=args.vocab_size)
    print "Number of embeddings loaded per language: ", args.vocab_size
    
    # find number of words in vocab
    en_words = borrowed_pairs["English"].tolist()
    ru_words = borrowed_pairs["Russian"].tolist()
    en_count = sum([1 if word in en_word2id else 0 for word in en_words])
    print "Number of english words in vocab:", en_count
    ru_count = sum([1 if word in ru_word2id else 0 for word in ru_words])
    print "Number of russian words in vocab:", ru_count


def main():
    args = parser.parse_args()

    en_word2id, en_emb = read_embeddings(args.en_embedding_file, n_max=args.vocab_size)
    ru_word2id, ru_emb = read_embeddings(args.ru_embedding_file, n_max=args.vocab_size)
    en_emb = torch.from_numpy(en_emb)
    ru_emb = torch.from_numpy(ru_emb)

    borrowed_pairs = load_dictionary(args.loanwords_file, en_word2id, ru_word2id)
    get_word_translation_accuracy(borrowed_pairs, en_word2id, en_emb, ru_word2id, ru_emb, args.knn_method)
    
    if args.flip_dict:
        supervised_pairs = src.evaluation.word_translation.load_dictionary(\
                                args.supervised_dict_file, ru_word2id, en_word2id)
        for i in range(supervised_pairs.size()[0]):
            supervised_pairs[i,0],supervised_pairs[i,1] = supervised_pairs[i,1],supervised_pairs[i,0]
        get_word_translation_accuracy(supervised_pairs, en_word2id, en_emb, ru_word2id, \
                                        ru_emb, args.knn_method)
    else:
        supervised_pairs = src.evaluation.word_translation.load_dictionary(\
                                args.supervised_dict_file, en_word2id, ru_word2id)
        get_word_translation_accuracy(supervised_pairs, en_word2id, en_emb, ru_word2id, \
                                        ru_emb, args.knn_method)


if __name__=="__main__":
    main()