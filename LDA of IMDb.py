import os
import codecs
import json
import string
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from datetime import datetime
from gensim import corpora, models
import logging
import math
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

NUM_TOPICS = k

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class NlpPreProcess(object):
    """Preprocess the html of gdelt-data"""
    def __init__(self, stopfile):
        self.wnl = WordNetLemmatizer()
        with codecs.open(stopfile, 'r', 'utf-8') as f:
            self.stoplist = f.read().splitlines()
        print('The number of stopwords is %s' % len(self.stoplist))

    def preprocess_excel(self, excel_path):
        """Preprocess data from an Excel file"""
        df = pd.read_excel(excel_path)
        stime = datetime.now()
        num = 0
        doclist = []
        for index, row in df.iterrows():
            doc = row.iloc[0]
            if isinstance(doc, str):  
                sentences = nltk.sent_tokenize(doc.lower())
                for sentence in sentences:
                    sentence = sentence.translate(str.maketrans('', '', string.punctuation + string.digits))
                    words = nltk.word_tokenize(sentence)
                    clean_doc = [self.wnl.lemmatize(word) for word in words if len(word) >= 3 and word not in self.stoplist and wordnet.synsets(word)]
                    if clean_doc:
                        doclist.append(clean_doc)
                        num += 1
        print('Time cost is : %s' % (datetime.now() - stime))
        print('The number of valid docs is : %s' % num)
        return doclist

class GLDA(object):
    """LDA Model Training using gensim"""
    def __init__(self, stopfile=None):
        if stopfile:
            with codecs.open(stopfile, 'r', 'utf-8') as f:
                self.stopword_list = f.read().split(' ')
            print('The number of stopwords is : %s' % len(self.stopword_list))
        else:
            self.stopword_list = None

    def lda_train(self, num_topics, doclist, result_folder, dictionary_path=None, corpus_path=None, iterations=5000, passes=1, workers=3):       
        time1 = datetime.now()
        if dictionary_path:
            dictionary = corpora.Dictionary.load(dictionary_path)
        else:
            dictionary = corpora.Dictionary(doclist)
            dictionary.save(os.path.join(result_folder, 'dictionary.dictionary'))
        if corpus_path:
            corpus = corpora.MmCorpus(corpus_path)
        else:
            corpus = [dictionary.doc2bow(doc) for doc in doclist]
            corpora.MmCorpus.serialize(os.path.join(result_folder, 'corpus.mm'), corpus)
        tfidf = models.TfidfModel(corpus)
        corpusTfidf = tfidf[corpus]
        time2 = datetime.now()
        lda_multi = models.LdaMulticore(
            corpus=corpusTfidf,
            id2word=dictionary,
            num_topics=num_topics,
            iterations=iterations,
            workers=workers,
            passes=passes
        )
        lda_multi.print_topics(num_topics, 30)
        print('LDA training time cost is : %s, all time cost is : %s ' % (datetime.now() - time2, datetime.now() - time1))
        lda_multi.save(os.path.join(result_folder, 'lda_tfidf_%s_%s.model' % (num_topics, iterations)))
        topic_id_file = codecs.open(os.path.join(result_folder, 'topic.json'), 'w', 'utf-8')
        for i in range(len(corpusTfidf)):
            topic_id = lda_multi[corpusTfidf[i]][0][0]
            topic_id_file.write(str(topic_id) + ' ')
        topic_id_file.close()
        return lda_multi, corpusTfidf, dictionary

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    """Calculate the perplexity of an LDA model"""
    print('Info of this LDA model:')
    print('Number of testset: %s; Size of dictionary: %s; Number of topics: %s' % (len(testset), size_dictionary, num_topics))
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []  # Store the probability of topic-word
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    
    doc_topics_list = []  # Store the doc-topic tuples
    for doc in testset:
        doc_topics_list.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0  # Probability of the document
        doc = testset[i]
        doc_word_num = 0  # Number of words in the document
        for word_id, num in doc:
            prob_word = 0.0  # Probability of the word 
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # Calculate p(w) : p(w) = sum_z(p(z)*p(w|z))
                prob_topic = doc_topics_list[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id].get(word, 0)
                prob_word += prob_topic * prob_topic_word
            if prob_word <= 0:
                prob_word = 1
            prob_doc += num * math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    
    prep = math.exp(-prob_doc_sum / testset_word_num)  # Perplexity = exp(-sum(p(d))/sum(Nd))
    print("The perplexity of this LDA model is: %s" % prep)
    return prep

if __name__ == '__main__':
    stopword_filepath = 
    excel_path = 
    result_folder = 
    os.makedirs(result_folder, exist_ok=True)
    
    # Preprocess Excel data
    nlp_preprocess = NlpPreProcess(stopword_filepath)
    doclist = nlp_preprocess.preprocess_excel(excel_path)
    
    # Train LDA model
    passes = 3
    iterations = 6000
    workers = 8
    lda = GLDA(stopword_filepath)
    lda_model, corpusTfidf, dictionary = lda.lda_train(
        NUM_TOPICS,
        doclist,
        result_folder,
        dictionary_path=None,
        corpus_path=None,
        iterations=iterations,
        passes=passes,
        workers=workers
    )

    # Calculate perplexity
    testset = [corpusTfidf[i * 300] for i in range(len(corpusTfidf) // 300)]
    prep = perplexity(lda_model, testset, dictionary, len(dictionary.keys()), NUM_TOPICS)

    # Visualize topics
    vis_data = gensimvis.prepare(lda_model, corpusTfidf, dictionary)
    pyLDAvis.save_html(vis_data, os.path.join(result_folder, 'lda_vis.html'))
    pyLDAvis.show(vis_data)

    # Save topic keywords to Excel
    topic_keywords = []
    for topic_id in range(NUM_TOPICS):
        topic = lda_model.show_topic(topic_id, topn=30)
        for word, prob in topic:
            topic_keywords.append([topic_id, word, prob])
    
    df_topic_keywords = pd.DataFrame(topic_keywords, columns=['Topic ID', 'Keyword', 'Frequency'])
    df_topic_keywords.to_excel(os.path.join(result_folder, 'topic_keywords.xlsx'), index=False)
