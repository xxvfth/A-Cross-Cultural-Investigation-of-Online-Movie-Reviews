import pandas as pd
from gsdmm import MovieGroupProcess


data_path = 
df = pd.read_excel(data_path, header=None) 
df['content'] = df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
df = df[df['content'].str.strip() != '']  
docs = df['content'].apply(lambda x: x.split()).tolist()  


vocab = set(word for doc in docs for word in doc)
vocab_size = len(vocab)



mgp = MovieGroupProcess(K=100, alpha=0.1, beta=0.1, n_iters=20)


y = mgp.fit(docs, vocab_size)



non_empty_clusters = sum(1 for cluster in range(mgp.K) if mgp.cluster_doc_count[cluster] > 0)
print(f"all {non_empty_clusters} group\n")



output_file_path = 

with open(output_file_path, 'w', encoding='utf-8') as f:
    for cluster in range(mgp.K):
        if mgp.cluster_doc_count[cluster] > 0:  
            total_count = sum(mgp.cluster_word_distribution[cluster].values())
            sorted_words = sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda item: item[1], reverse=True)
            top_words = sorted_words[:20]  
            cluster_result = f"Cluster {cluster} words: " + ", ".join([f"{word}: {count / total_count:.4f}" for word, count in top_words])
            f.write(cluster_result + "\n")
            print(cluster_result)
            print(f"Cluster {cluster} documents: {mgp.cluster_doc_count[cluster]}\n")

print(f"save: {output_file_path}")