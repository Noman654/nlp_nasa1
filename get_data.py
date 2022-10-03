import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import string
import gensim
import operator
import re
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings('ignore')

RESEARCH_INDEX = None
RESEARCH_LSI_MODEL = None
RESEARCH_TFIDF_MODEL = None
DICTIONARY = None
SPACY_NLP = None
STOP_WORDS = None
DF_RESEARCH = None
PUNCTUATIONS = None

def spacy_tokenizer(sentence,spacy_nlp,stop_words,punctuations ):
 
    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    #remove unwanted lines starting from special charcters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    
    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)
    
    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    #creating token object
    tokens = spacy_nlp(sentence)
    
    #lower, strip and lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    
    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    
    #return tokens
    return tokens


def get_resources() ->None:
    global RESEARCH_INDEX,RESEARCH_LSI_MODEL,RESEARCH_TFIDF_MODEL,DICTIONARY,\
SPACY_NLP,STOP_WORDS,DF_RESEARCH,PUNCTUATIONS

    DF_RESEARCH = pd.read_csv('final_nasa_data.csv')
    SPACY_NLP = spacy.load('en_core_web_sm')

    #create list of punctuations and stopwords
    PUNCTUATIONS = string.punctuation
    STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

    #function for data cleaning and processing
    #This can be further enhanced by adding / removing reg-exps as desired.

    df = DF_RESEARCH[DF_RESEARCH['abstract'].notna()]
    df['abstrac_r'] = df['abstract']

    def concate_categ(x):
        temp_s = ""
        x = ast.literal_eval(x)
        for i in x:
            temp_s +=i
        return temp_s

    df['abstract'] = df['abstract'] + df['title'] +df['subjectCategories'].apply(concate_categ)

    print ('Cleaning and Tokenizing...')
    df['abstract_tokenized'] = df['abstract'].map(lambda x: spacy_tokenizer(x,SPACY_NLP,STOP_WORDS,PUNCTUATIONS))

    df.reset_index(inplace=True)
    research_plot = df['abstract_tokenized']
    research_plot[0:5]

    series = pd.Series(np.concatenate(research_plot)).value_counts()[:100]
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(series)

    plt.figure(figsize=(15,15), facecolor = None)
    plt.imshow(wordcloud, interpolation='bilinear')


    #creating term dictionary
    DICTIONARY = corpora.Dictionary(research_plot)


    #dictionary.filter_extremes(no_below=4, no_above=0.2)

    #list of few which which can be further removed
    stoplist = set('hello and if this can would should could tell ask stop come go')
    stop_ids = [DICTIONARY.token2id[stopword] for stopword in stoplist if stopword in DICTIONARY.token2id]
    DICTIONARY.filter_tokens(stop_ids)

    #print top 50 items from the dictionary with their unique token-id
    dict_tokens = [[[DICTIONARY[key], DICTIONARY.token2id[DICTIONARY[key]]] for key, value in DICTIONARY.items() if key <= 50]]

    corpus = [DICTIONARY.doc2bow(desc) for desc in research_plot]

    word_frequencies = [[(DICTIONARY[id], frequency) for id, frequency in line] for line in corpus[0:3]]

    RESEARCH_TFIDF_MODEL = gensim.models.TfidfModel(corpus, id2word=DICTIONARY)
    RESEARCH_LSI_MODEL = gensim.models.LsiModel(RESEARCH_TFIDF_MODEL[corpus], id2word=DICTIONARY, num_topics=300)

    gensim.corpora.MmCorpus.serialize('research_tfidf_model_mm', RESEARCH_TFIDF_MODEL[corpus])
    gensim.corpora.MmCorpus.serialize('research_lsi_model_mm',RESEARCH_LSI_MODEL[RESEARCH_TFIDF_MODEL[corpus]])

    research_tfidf_corpus = gensim.corpora.MmCorpus('research_tfidf_model_mm')
    research_lsi_corpus = gensim.corpora.MmCorpus('research_lsi_model_mm')

    RESEARCH_INDEX = MatrixSimilarity(research_lsi_corpus, num_features = research_lsi_corpus.num_terms)



def search_similar_research(search_term):

    query_bow = DICTIONARY.doc2bow(spacy_tokenizer(search_term,SPACY_NLP,STOP_WORDS,PUNCTUATIONS))
    query_tfidf = RESEARCH_TFIDF_MODEL[query_bow]
    query_lsi = RESEARCH_LSI_MODEL[query_tfidf]

    RESEARCH_INDEX.num_best = 5

    researchs_list = RESEARCH_INDEX[query_lsi]

    researchs_list.sort(key=itemgetter(1), reverse=True)
    research_names = []

    for j, research in enumerate(researchs_list):

        research_names.append (
            {
                'Relevance': round((research[1] * 100),2),
                'Title': DF_RESEARCH['title'][research[0]],
                'category': DF_RESEARCH['subjectCategories'][research[0]],
                'research Plot': DF_RESEARCH['abstract'][research[0]]
            }

        )
        if j == (RESEARCH_INDEX.num_best-1):
            break

    return pd.DataFrame(research_names, columns=['Relevance','Title' ,'category' ,'research Plot'])
