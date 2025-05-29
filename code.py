import pandas as pd
import re
import nltk
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.matcher import PhraseMatcher
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from colbert.infra import Indexer, Searcher
from nltk.tokenize import word_tokenize
from num2words import num2words
from sklearn.decomposition import TruncatedSVD
import fasttext
import fasttext.util
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Test Bleu
## discrete
nlp = spacy.load("en_core_web_sm")
custom_greetings = {'hello', 'hi', 'hey', 'greetings', 'hola', 'bonjour', 'salut', 'ciao', 'hallo', 'namaste', 'nihao', 'konnichiwa', 'ohayo'}
synonyms = {
    "ai": "artificial intelligence",
    "a.i.": "artificial intelligence",
    "ml": "machine learning",
    "m.l.": "machine learning",
    "dl": "deep learning",
    "deep-learning": "deep learning",
    "nlp": "natural language processing",
    "usa": "united states",
    "u.s.a.": "united states",
    "us": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "covid": "coronavirus",
    "covid-19": "coronavirus",
    "big data": "data science",
    "gpu": "graphics processing unit",
    "cpu": "central processing unit",
    "vpn": "virtual private network",
    "ip": "internet protocol",
    "dns": "domain name system",
    "tcp": "transmission control protocol",
    "hiv": "human immunodeficiency virus",
    "aids": "acquired immunodeficiency syndrome",
    "sars": "severe acute respiratory syndrome"
}

train_df = pd.read_csv("/home/3304575/.vscode/assignment 2/train_responses.csv")
dev_df = pd.read_csv("/home/3304575/.vscode/assignment 2/dev_responses.csv")
test_df = pd.read_csv("/home/3304575/.vscode/assignment 2/test_prompts.csv")
train_df = train_df[['conversation_id','user_prompt', 'model_response']].dropna()
dev_df = dev_df[['conversation_id','user_prompt', 'model_response']].dropna()
test_df = test_df[['conversation_id', 'user_prompt']]   
def discrete_preprocess_text(text, remove_stopwords=True, use_ner=True): 
    ### 1. cleaning
    text = str(text).lower()
    text = re.sub(r'[“”‘’´`]', "'", text)  # 统一引号为标准单引号
    text = re.sub(r'\s+', ' ', text).strip()       # remove space
    text = re.sub(r'[–—]', '-', text)       # 统一长破折号为短横线
    text = re.sub(r'[…]+', '...', text)    # 统一省略号
    text = re.sub(r'[^\w\s\'\-:=_+*/()]', '', text)
    text = re.sub(r"\b(who|what|where|when|why|how) s\b", r"\1 is", text, flags=re.IGNORECASE)

    words = text.split()
    words = [synonyms.get(word, word) for word in words]  # **按词替换**
    text = " ".join(words)

    greeting_pattern = r'^\s*([a-zA-Z]+)[,!;:]*\s*'
    first_word = re.match(greeting_pattern, text, re.IGNORECASE)
    if first_word and first_word.group(1).lower() in custom_greetings:
        text = re.sub(greeting_pattern, '', text)

    ### 2 token, lemma, stopwords, pos 
    doc = nlp(text)
    tokens, lemmas, pos_tags, content = [], [], [], []
    for sentence in doc.sents:
        for token in sentence:
            '''
            tokens.append(token.text)
            lemmas.append(token.lemma_)
            pos_tags.append(token.pos_)
            important_words = {"who", "why", "what", "how", "when", "where"}
            negations = {"not", "never", "no", "n't"}
            MATH_SYMBOLS = {"+", "-", "*", "/",'=','_'}
            if token.text in important_words or token.text in negations or token.text in MATH_SYMBOLS:
                content.append(token.text)  # 直接保留疑问词 & 否定词
            elif not token.is_stop and token.pos_ in {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'}:
                content.append(token.lemma_)  # common tfidf
            '''
            content.append(token.text) # smooth version
    ### 固定搭配的n-gram延迟到表示层
 
    return " ".join(content)

train_discrete = train_df.copy()
dev_discrete = dev_df.copy()
test_discrete = test_df.copy()
train_discrete['processed_prompt'] = train_discrete['user_prompt'].apply(discrete_preprocess_text)
dev_discrete['processed_prompt'] = dev_discrete['user_prompt'].apply(discrete_preprocess_text)
test_discrete['processed_prompt'] = test_discrete['user_prompt'].apply(discrete_preprocess_text)

from rank_bm25 import BM25Okapi
train_discrete = pd.read_csv("train_discrete.csv")
dev_discrete = pd.read_csv("dev_discrete.csv")
test_discrete = pd.read_csv("test_discrete.csv")
train_texts = [word_tokenize(text) for text in train_discrete["processed_prompt"].tolist()]
dev_texts = [word_tokenize(text) for text in dev_discrete["processed_prompt"].tolist()]

# ====== BM25 计算 ======
bm25 = BM25Okapi(train_texts, k1=1.5, b=0.75)

# ====== TF-IDF 变体计算 ======
train_sentences = [" ".join(text) for text in train_texts]
dev_sentences = [" ".join(text) for text in dev_texts]

# 1. 标准 TF-IDF
vectorizer_tfidf = TfidfVectorizer(use_idf=True, smooth_idf=False)
train_tfidf = vectorizer_tfidf.fit_transform(train_sentences)
dev_tfidf = vectorizer_tfidf.transform(dev_sentences)

# 2. 平滑 IDF 版本
vectorizer_smooth_tfidf = TfidfVectorizer(use_idf=True, smooth_idf=True)
train_smooth_tfidf = vectorizer_smooth_tfidf.fit_transform(train_sentences)
dev_smooth_tfidf = vectorizer_smooth_tfidf.transform(dev_sentences)

# 3. 对数 TF 版本
vectorizer_log_tf = TfidfVectorizer(use_idf=False, sublinear_tf=True)
train_log_tf = vectorizer_log_tf.fit_transform(train_sentences)
dev_log_tf = vectorizer_log_tf.transform(dev_sentences)

# ====== 计算最优匹配 ======
retrieved_responses = []
best_match_indices = []

for i, dev_query in enumerate(dev_texts):
    # 计算 BM25 相似度
    bm25_scores = bm25.get_scores(dev_query)
    
    # 计算 TF-IDF 变体的相似度
    tfidf_scores = cosine_similarity(dev_tfidf[i], train_tfidf)[0]
    smooth_tfidf_scores = cosine_similarity(dev_smooth_tfidf[i], train_smooth_tfidf)[0]
    log_tf_scores = cosine_similarity(dev_log_tf[i], train_log_tf)[0]
    
    # 综合多个相似度得分，选择最大值的索引
    all_scores = np.array([bm25_scores, tfidf_scores, smooth_tfidf_scores, log_tf_scores])
    best_match_idx = np.argmax(all_scores.max(axis=0))  # 选择最优匹配索引
    
    best_match_indices.append(best_match_idx)
    retrieved_responses.append(train_discrete.iloc[best_match_idx]["model_response"])

dev_discrete["retrieved_conversation_id"] = train_discrete["conversation_id"].iloc[best_match_indices].values
dev_discrete["retrieved_response_id"] = best_match_indices  # 直接存索引作为 response_id
dev_discrete["retrieved_response"] = retrieved_responses

sampled_dev = dev_discrete.sample(n=1000, random_state=42) if len(dev_discrete) > 1000 else dev_discrete
smoothing = SmoothingFunction()
sampled_dev["bleu_score"] = sampled_dev.apply(
    lambda x: sentence_bleu(
        [x["model_response"].split()],  
        x["retrieved_response"].split(),
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing.method3,
    ),
    axis=1
)

bleu_mean = sampled_dev["bleu_score"].mean()
print(f"Optimized Hybrid Retrieval BLEU Score: {bleu_mean:.4f}")

## dense static
train_df = pd.read_csv("train_responses.csv")
dev_df = pd.read_csv("dev_responses.csv")
test_df = pd.read_csv("test_prompts.csv")

train_df = train_df[['conversation_id', 'user_prompt', 'model_response']].dropna()
dev_df = dev_df[['conversation_id', 'user_prompt', 'model_response']].dropna()
test_df = test_df[['conversation_id', 'user_prompt']]
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[“”‘’´`]', "'", text)  # 统一引号为标准单引号
    text = re.sub(r'\s+', ' ', text).strip()       # remove space
    text = re.sub(r'[–—]', '-', text)       # 统一长破折号为短横线
    text = re.sub(r'[…]+', '...', text)    # 统一省略号
    text = re.sub(r'[^\w\s\'\-:=_+*/()?]', '', text)
    text = re.sub(r"\b(who|what|where|when|why|how) s\b", r"\1 is", text, flags=re.IGNORECASE)

    words = text.split()
    words = [synonyms.get(word, word) for word in words]  # **按词替换**
    text = " ".join(words)

    greeting_pattern = r'^\s*([a-zA-Z]+)[,!;:]*\s*'
    first_word = re.match(greeting_pattern, text, re.IGNORECASE)
    if first_word and first_word.group(1).lower() in custom_greetings:
        text = re.sub(greeting_pattern, '', text)
    
    return text

train_df['processed_prompt'] = train_df['user_prompt'].apply(preprocess_text)
dev_df['processed_prompt'] = dev_df['user_prompt'].apply(preprocess_text)
test_df['processed_prompt'] = test_df['user_prompt'].apply(preprocess_text)

train_df = pd.read_csv("train_discrete.csv")
dev_df = pd.read_csv("dev_discrete.csv")
test_df = pd.read_csv("test_discrete.csv")
# ====== FastText 载入 & 计算句向量 ======
fasttext.util.download_model('en', if_exists='ignore')
ft_model = fasttext.load_model("cc.en.300.bin")

train_vectors_ft = np.array([ft_model.get_sentence_vector(sent) for sent in train_df["processed_prompt"]])
dev_vectors_ft = np.array([ft_model.get_sentence_vector(sent) for sent in dev_df["processed_prompt"]])

# ====== Doc2Vec 训练 & 生成向量 ======
train_texts = [word_tokenize(text) for text in train_df["processed_prompt"].tolist()]
train_tagged = [TaggedDocument(words=text, tags=[i]) for i, text in enumerate(train_texts)]

doc2vec_model = Doc2Vec(vector_size=300, window=8, min_count=2, workers=4, epochs=100, dm=1, dbow_words=1)  # **PV-DM**
doc2vec_model.build_vocab(train_tagged)
doc2vec_model.train(train_tagged, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

train_vectors_d2v = np.array([doc2vec_model.dv[i] for i in range(len(train_texts))])
dev_vectors_d2v = np.array([doc2vec_model.infer_vector(word_tokenize(text)) for text in dev_df["processed_prompt"]])

# ====== 计算 FastText & Doc2Vec 相似度 ======
train_vectors_ft_norm = normalize(train_vectors_ft)
dev_vectors_ft_norm = normalize(dev_vectors_ft)
similarity_matrix_ft = cosine_similarity(dev_vectors_ft_norm, train_vectors_ft_norm)

train_vectors_d2v_norm = normalize(train_vectors_d2v)
dev_vectors_d2v_norm = normalize(dev_vectors_d2v)
similarity_matrix_d2v = cosine_similarity(dev_vectors_d2v_norm, train_vectors_d2v_norm)

# 计算 FastText 和 Doc2Vec 的最大相似度值
max_similarity_ft = np.max(similarity_matrix_ft, axis=1)  # (4998,)
max_similarity_d2v = np.max(similarity_matrix_d2v, axis=1)  # (4998,)

# 计算索引
best_match_indices = np.where(max_similarity_ft > max_similarity_d2v,
                              np.argmax(similarity_matrix_ft, axis=1),
                              np.argmax(similarity_matrix_d2v, axis=1))


retrieved_responses = train_df["model_response"].iloc[best_match_indices].values
dev_df["retrieved_conversation_id"] = train_df["conversation_id"].iloc[best_match_indices].values
dev_df["retrieved_response_id"] = best_match_indices
dev_df["retrieved_response"] = retrieved_responses

# ====== 计算 BLEU 评分 ======
sampled_dev = dev_df.sample(n=1000, random_state=42) if len(dev_df) > 1000 else dev_df
smoothing = SmoothingFunction()
sampled_dev["bleu_score"] = sampled_dev.apply(
    lambda x: sentence_bleu(
        [x["model_response"].split()],  
        x["retrieved_response"].split(),
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing.method3,
    ),
    axis=1
)

bleu_mean = sampled_dev["bleu_score"].mean()
print(f"Optimized Max-Similarity BLEU Score: {bleu_mean:.4f}")


## open
nlp = spacy.load("en_core_web_trf")

def bert_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[“”‘’´`]', "'", text)  # 统一引号为标准单引号
    text = re.sub(r'\s+', ' ', text).strip()       # remove space
    text = re.sub(r'[–—]', '-', text)       # 统一长破折号为短横线
    text = re.sub(r'[…]+', '...', text)    # 统一省略号
    text = re.sub(r'[^\w\s\'\-:=_+*/()?]', '', text)
    text = re.sub(r"\b(who|what|where|when|why|how) s\b", r"\1 is", text, flags=re.IGNORECASE)

    words = text.split()
    words = [synonyms.get(word, word) for word in words]  # **按词替换**
    text = " ".join(words)

    greeting_pattern = r'^\s*([a-zA-Z]+)[,!;:]*\s*'
    first_word = re.match(greeting_pattern, text, re.IGNORECASE)
    if first_word and first_word.group(1).lower() in custom_greetings:
        text = re.sub(greeting_pattern, '', text)
    
    doc = nlp(text)
    keep_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'NUM'}
    tokens = [token.lemma_ for token in doc if token.pos_ in keep_pos and not token.is_stop]
    
    return " ".join(tokens)

train_df = pd.read_csv("train_responses.csv")
dev_df = pd.read_csv("dev_responses.csv")

train_df["processed_prompt"] = train_df["user_prompt"].apply(bert_preprocess)
dev_df["processed_prompt"] = dev_df["user_prompt"].apply(bert_preprocess)

train_df = pd.read_csv("train_discrete.csv")
dev_df = pd.read_csv("dev_discrete.csv")

from sentence_transformers import SentenceTransformer

# 更换模型
model = SentenceTransformer("BAAI/bge-large-en")  
train_embeddings = model.encode(train_df["processed_prompt"].tolist(), normalize_embeddings=True)
dev_embeddings = model.encode(dev_df["processed_prompt"].tolist(), normalize_embeddings=True)

import faiss

d = train_embeddings.shape[1]  # 维度
index = faiss.IndexHNSWFlat(d, 32)  # HNSW 近似最近邻
index.hnsw.efConstruction = 200
index.add(train_embeddings)
index.hnsw.efSearch = 128  # 搜索时的近邻数量

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

reranker_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates):
    inputs = reranker_tokenizer([query] * len(candidates), candidates, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze().numpy()
    return candidates[np.argmax(scores)]

retrieved_responses = []
for i, query in enumerate(dev_df["processed_prompt"].tolist()):
    _, candidate_indices = index.search(dev_embeddings[i].reshape(1, -1), 10)  # 取 Top-10
    candidate_responses = train_df["model_response"].iloc[candidate_indices[0]].tolist()
    best_response = rerank(query, candidate_responses)
    retrieved_responses.append(best_response)

dev_df["retrieved_response"] = retrieved_responses

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothing = SmoothingFunction()
sampled_dev = dev_df.sample(n=1000, random_state=42) if len(dev_df) > 1000 else dev_df
sampled_dev["bleu_score"] = [
    sentence_bleu([ref.split()], hyp.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing.method3)
    for ref, hyp in zip(sampled_dev["model_response"], sampled_dev["retrieved_response"])
]

bleu_mean = sampled_dev["bleu_score"].mean()
print(f"Optimized BLEU Score with Cross-Encoder: {bleu_mean:.4f}")

# Match Pridections
## discrete 
train_dev_discrete = pd.concat([train_discrete, dev_discrete], ignore_index=True)

# 分词处理
train_dev_texts = [word_tokenize(text) for text in train_dev_discrete['processed_prompt'].tolist()]
test_texts = [word_tokenize(text) for text in test_discrete["processed_prompt"].tolist()]

# ====== BM25 计算 ======
bm25 = BM25Okapi(train_dev_texts, k1=1.5, b=0.75)

# ====== TF-IDF 变体计算 ======
train_dev_sentences = [" ".join(text) for text in train_dev_texts]
test_sentences = [" ".join(text) for text in test_texts]

# 1. 标准 TF-IDF
vectorizer_tfidf = TfidfVectorizer(use_idf=True, smooth_idf=False)
train_dev_tfidf = vectorizer_tfidf.fit_transform(train_dev_sentences)
test_tfidf = vectorizer_tfidf.transform(test_sentences)

# 2. 平滑 IDF 版本
vectorizer_smooth_tfidf = TfidfVectorizer(use_idf=True, smooth_idf=True)
train_dev_smooth_tfidf = vectorizer_smooth_tfidf.fit_transform(train_dev_sentences)
test_smooth_tfidf = vectorizer_smooth_tfidf.transform(test_sentences)

# 3. 对数 TF 版本
vectorizer_log_tf = TfidfVectorizer(use_idf=False, sublinear_tf=True)
train_dev_log_tf = vectorizer_log_tf.fit_transform(train_dev_sentences)
test_log_tf = vectorizer_log_tf.transform(test_sentences)

# ====== 计算最优匹配 ======
best_match_indices = []

for i, test_query in enumerate(test_texts):
    # 计算 BM25 相似度
    bm25_scores = bm25.get_scores(test_query)
    
    # 计算 TF-IDF 变体的相似度
    tfidf_scores = cosine_similarity(test_tfidf[i], train_dev_tfidf)[0]
    smooth_tfidf_scores = cosine_similarity(test_smooth_tfidf[i], train_dev_smooth_tfidf)[0]
    log_tf_scores = cosine_similarity(test_log_tf[i], train_dev_log_tf)[0]
    
    # 综合多个相似度得分，选择最大值的索引
    all_scores = np.array([bm25_scores, tfidf_scores, smooth_tfidf_scores, log_tf_scores])
    best_match_idx = np.argmax(all_scores.max(axis=0))  # 选择最优匹配索引
    
    best_match_indices.append(best_match_idx)

test_discrete["response_id"] = train_dev_discrete["conversation_id"].iloc[best_match_indices].values
test_discrete[["conversation_id", "response_id"]].to_csv("track_1_test.csv", index=False)

## dense static
train_df = pd.read_csv("train_discrete.csv")
dev_df = pd.read_csv("dev_discrete.csv")
test_df = pd.read_csv("test_discrete.csv")

# 合并 Train + Dev
train_dev_df = pd.concat([train_df, dev_df], ignore_index=True)

# ====== 2. 加载 FastText 模型 ======
fasttext.util.download_model('en', if_exists='ignore')  
ft_model = fasttext.load_model("cc.en.300.bin")

# 计算 FastText 句子向量
train_dev_vectors_ft = np.array([ft_model.get_sentence_vector(sent) for sent in train_dev_df["processed_prompt"]])
test_vectors_ft = np.array([ft_model.get_sentence_vector(sent) for sent in test_df["processed_prompt"]])

# ====== 3. 训练 Doc2Vec 模型 ======
train_dev_texts = [word_tokenize(text) for text in train_dev_df["processed_prompt"].tolist()]
train_dev_tagged = [TaggedDocument(words=text, tags=[i]) for i, text in enumerate(train_dev_texts)]

doc2vec_model = Doc2Vec(vector_size=300, window=8, min_count=3, workers=4, epochs=50, dm=0)  # PV-DBOW
doc2vec_model.build_vocab(train_dev_tagged)
doc2vec_model.train(train_dev_tagged, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# 计算 Doc2Vec 句子向量
train_dev_vectors_d2v = np.array([doc2vec_model.dv[i] for i in range(len(train_dev_texts))])
test_vectors_d2v = np.array([doc2vec_model.infer_vector(word_tokenize(text)) for text in test_df["processed_prompt"]])

# ====== 4. 计算相似度 ======
train_dev_vectors_ft_norm = normalize(train_dev_vectors_ft)
test_vectors_ft_norm = normalize(test_vectors_ft)

train_dev_vectors_d2v_norm = normalize(train_dev_vectors_d2v)
test_vectors_d2v_norm = normalize(test_vectors_d2v)

similarity_matrix_ft = cosine_similarity(test_vectors_ft_norm, train_dev_vectors_ft_norm)
similarity_matrix_d2v = cosine_similarity(test_vectors_d2v_norm, train_dev_vectors_d2v_norm)

# ====== 5. 选择最优匹配 ======
max_similarity_ft = np.max(similarity_matrix_ft, axis=1)
max_similarity_d2v = np.max(similarity_matrix_d2v, axis=1)

best_match_indices = np.where(max_similarity_ft > max_similarity_d2v,
                              np.argmax(similarity_matrix_ft, axis=1),
                              np.argmax(similarity_matrix_d2v, axis=1))

test_df["response_id"] = train_dev_df["conversation_id"].iloc[best_match_indices].values
test_df[["conversation_id", "response_id"]].to_csv("track_2_test.csv", index=False)

## open
import faiss
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
train_dev_discrete = pd.concat([train_discrete, dev_discrete], ignore_index=True)
test_discrete = pd.read_csv("test_discrete.csv")

# 加载强力语义匹配模型
model = SentenceTransformer("BAAI/bge-large-en")

# 计算 `train+dev` 和 `test` 的 embedding
train_dev_embeddings = model.encode(train_dev_discrete["processed_prompt"].tolist(), normalize_embeddings=True)
test_embeddings = model.encode(test_discrete["processed_prompt"].tolist(), normalize_embeddings=True)

# 建立 FAISS HNSW 近邻索引
d = train_dev_embeddings.shape[1]  # 向量维度
index = faiss.IndexHNSWFlat(d, 32)  # HNSW 近似最近邻
index.hnsw.efConstruction = 200
index.add(train_dev_embeddings)
index.hnsw.efSearch = 128  # 搜索时的近邻数量

# 加载 Cross-Encoder（用于 rerank）
reranker_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates):
    inputs = reranker_tokenizer([query] * len(candidates), candidates, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze().numpy()
    return candidates[np.argmax(scores)]

best_match_indices = []

for i, query in enumerate(test_discrete["processed_prompt"].tolist()):
    _, candidate_indices = index.search(test_embeddings[i].reshape(1, -1), 10)  # 取 Top-10
    candidate_responses = train_dev_discrete["model_response"].iloc[candidate_indices[0]].tolist()
    best_response = rerank(query, candidate_responses)

    # 获取最终的 `response_id`
    best_match_idx = train_dev_discrete[train_dev_discrete["model_response"] == best_response].index[0]
    best_match_indices.append(best_match_idx)

test_discrete["response_id"] = train_dev_discrete["conversation_id"].iloc[best_match_indices].values
test_discrete[["conversation_id", "response_id"]].to_csv("track_3_test.csv", index=False)