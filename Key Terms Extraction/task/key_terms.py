import string
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.tree import *
from lxml import etree
# from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')


def remove_stop_words_and_punctuation(tokens):
    idx = [i for i in range(len(tokens)) if tokens[i] in stop_words]
    idx.reverse()
    for i in idx:
        tokens.pop(i)

    idx = [i for i in range(len(tokens)) if tokens[i] in string.punctuation]
    idx.reverse()
    for i in idx:
        tokens.pop(i)


def keep_only_nouns(tokens):
    # tokens = nltk.pos_tag(tokens)
    # print(tokens)
    # nouns = [t[0] for t in tokens if t[1]=="NN"]
    # idx.reverse()
    # for i in idx:
    #     tokens.pop(i)
    result = [word[0] for x in tokens for word in nltk.pos_tag([x]) if word[1] == "NN"]
    # print("result",result)
    return result


def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    for i in range(len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i])

    remove_stop_words_and_punctuation(tokens)
    # print(tokens)
    tokens = keep_only_nouns(tokens)
    return tokens


def get_tfidf_matrix(dataset):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(input='content', use_idf=True, lowercase=False,
                                 analyzer='word', ngram_range=(1, 1),
                                 stop_words=None)

    tfidf_matrix = vectorizer.fit_transform(dataset)
    terms = vectorizer.get_feature_names()
    # terms = vectorizer.get_feature_names()
    # print(tfidf_matrix[(0, 10)])
    return tfidf_matrix,terms


def get_most_common_tokens(tf_idf_matrix, terms, idx):
    # from scipy.sparse import  find
    import numpy as np

    # print("********")
    tf_idf_matrix_array = tf_idf_matrix.toarray()
    # print(len(tf_idf_matrix_array))
    args = np.argsort( tf_idf_matrix_array[idx])
    # print(args[-1],tf_idf_matrix_array[0][args[-1]])
    # print(args[-2],tf_idf_matrix_array[0][args[-2]])
    # print(args[-3],tf_idf_matrix_array[0][args[-3]])
    # print(args[-4],tf_idf_matrix_array[0][args[-4]])
    # print(args[-5],tf_idf_matrix_array[0][args[-5]])
    # print(args[-1:-6:-1])
    # print([tf_idf_matrix_array[idx][i] for i in args[-1:-6:-1] ])
    result = [(terms[i],tf_idf_matrix_array[idx][i]) for i in args[::-1] ]
    # print(result)
    # print(find(tf_idf_matrix))
    # print(find(tf_idf_matrix)[0])
    # print(find(tf_idf_matrix)[1])
    # print(find(tf_idf_matrix)[2])

    # terms = filter(lambda entry: entry[0] == idx, tf_idf_matrix)
    # terms = tf_idf_matrix[idx]
    # print("terms", list(terms))
    # most = Counter(terms).most_common()
    # # print(most)
    d = {}
    for word in result:
        if len(d) <= 5:
            d.setdefault(word[1], []).append(word[0])
            # print(word[0],word[1], ord(word[0][0]))
    return [w for _, l in d.items() for w in sorted(l, reverse=True)][:5]
    # return result

def main():
    root = etree.parse("news.xml")
    corpuses = root.getroot()[0]
    dataset = [" ".join(preprocess_text(news[1].text)) for news in corpuses]
    tf_idf_matrix,terms = get_tfidf_matrix(dataset)

    # print(tf_idf_matrix)

    for idx, news in enumerate(corpuses):
        # etree.dump(news)
        print(news[0].text + ":")

        most_common_tokens = get_most_common_tokens(tf_idf_matrix, terms,idx)
        # print(most_common_tokens)
        print(" ".join(most_common_tokens))
        print()
    # print(dataset)


main()
