from src.utils import read_embeddings
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file')
    args = parser.parse_args()

    word2id, embeddings = read_embeddings(args.embedding_file, n_max=50)
    for word, index in word2id.iteritems():
        print word, index

if __name__=="__main__":
    main()