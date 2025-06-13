import os
import codecs
import pandas as pd
import sentencepiece as spm
from datetime import datetime
from gensim import corpora, models
import logging
import math
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


NUM_TOPICS = k
PASSES = 3
ITERATIONS = 6000
WORKERS = 8

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class NlpPreProcess(object):
    """Preprocess the html of gdelt-data"""
    def __init__(self, stopfile, spm_model):
        self.sp = spm.SentencePieceProcessor(model_file=spm_model)
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
                
                for stopword in self.stoplist:
                    doc = doc.replace(stopword, '')

                words = self.sp.encode_as_pieces(doc)  

                
                clean_doc = [word for word in words if len(word) >= 2 and word not in self.stoplist]
                
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

    def lda_train(self, num_topics, doclist, result_folder, dictionary_path=None, corpus_path=None, iterations=6000, passes=3, workers=8):       
        time1 = datetime.now()
        if not doclist:
            raise ValueError("The document list is empty after preprocessing. Please check your data and preprocessing steps.")
        
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
    
    if not testset:
        print("Testset is empty. Exiting perplexity calculation.")
        return float('inf')
    
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
    
    if testset_word_num == 0:
        print("No words in testset. Exiting perplexity calculation.")
        return float('inf')
    
    prep = math.exp(-prob_doc_sum / testset_word_num)  # Perplexity = exp(-sum(p(d))/sum(Nd))
    print("The perplexity of this LDA model is: %s" % prep)
    return prep

if __name__ == '__main__':
    stopword_filepath = 
    excel_path = 
    result_folder = 
    spm_model = '
    os.makedirs(result_folder, exist_ok=True)
    
    # Preprocess Excel data
    nlp_preprocess = NlpPreProcess(stopword_filepath, spm_model)
    doclist = nlp_preprocess.preprocess_excel(excel_path)
    
    
    if not doclist:
        raise ValueError("The document list is empty after preprocessing. Please check your data and preprocessing steps.")
    
    # Train LDA model
    lda = GLDA(stopword_filepath)
    lda_model, corpusTfidf, dictionary = lda.lda_train(
        NUM_TOPICS,
        doclist,
        result_folder,
        dictionary_path=None,
        corpus_path=None,
        iterations=ITERATIONS,
        passes=PASSES,
        workers=WORKERS
    )

    # Calculate perplexity
    testset = [corpusTfidf[i * 300] for i in range(len(corpusTfidf) // 300)]
    if not testset:
        print("Testset is empty. Please check your corpus and sampling method.")
    else:
        prep = perplexity(lda_model, testset, dictionary, len(dictionary.keys()), NUM_TOPICS)

    # Visualize topics
    vis_data = gensimvis.prepare(lda_model, corpusTfidf, dictionary, mds='mmds', sort_topics=False)
    html_path = os.path.join(result_folder, 'lda_vis.html')
    pyLDAvis.save_html(vis_data, html_path)

    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    html_content = html_content.replace(
        'https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.js',
        'D:\\pythonk\\lda\\pyLDAvis\\ldavis.v1.0.0.js'
    )
    html_content = html_content.replace(
        'https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css',
        'D:\\pythonk\\lda\\pyLDAvis\\ldavis.v1.0.0.css'
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    
    pyLDAvis.display(vis_data)
