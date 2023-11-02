import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.stem import WordNetLemmatizer
import umap
import hdbscan
import openai



def create_clusters(df, emb_model):
    embeddings = emb_model.encode(df['combined_text'])

    # Dimensionality reduction
    fit = umap.UMAP(n_neighbors=3, n_components=3, min_dist=0.05, random_state=1, n_jobs=1)
    u = fit.fit_transform(embeddings)

    # HDBSCAN - clusters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
    clusterer.fit(u)

    df['cluster_id'] = clusterer.labels_


def preprocess(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Lowercase and remove punctuation
    text = re.compile(r'[^a-zA-Z ]+').sub('', text).lower()
    
    # Tokenization
    tokens = nltk.tokenize.word_tokenize(text)
    
    # Stopword Filtering
    tokens = [token for token in tokens if token not in stopwords]
    
    # Lemmatize the words
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized_tokens


def calculate_class_tf_idf(df):
    clusters = {label: {'tokens': []} for label in set(df['cluster_id'].values)}

    for index, row in df.iterrows():
        label = row['cluster_id'] 
        tokens = row['preprocessed_text']
        
        # Update the 'tokens' list for the corresponding class in the classes dictionary
        clusters[label]['tokens'].extend(tokens)
    
    # Create the vocabularies (both overall and cluster).
    vocab = set()
    for c in clusters.keys():
        vocab = vocab.union(set(clusters[c]['tokens']))
        clusters[c]['vocab'] = set(clusters[c]['tokens'])

    # Get word frequency per cluster (TF).
    tf = np.zeros((len(clusters.keys()), len(vocab)))

    for c, _cluster in enumerate(clusters.keys()):
        for t, term in enumerate(vocab):
            tf[c, t] = clusters[_cluster]['tokens'].count(term)

    # Now calculate IDF (which tells us how common a term is). 
    # Rare terms signify greater relevance than common terms (and will output a greater IDF score).
    idf = np.zeros((1, len(vocab)))

    # calculate average number of words per cluster
    A = tf.sum() / tf.shape[0]

    for t, term in enumerate(vocab):
        # frequency of term t across all clusters
        f_t = tf[:,t].sum()
        # calculate IDF
        idf_score = np.log(1 + (A / f_t))
        idf[0, t] = idf_score

    c_tf_idf = tf*idf

    return c_tf_idf, vocab, clusters


# A function to sample 5 random sentences from each cluster
def sample_sentences(group):
    return group.sample(5, random_state=42)['combined_text'].tolist()  # Set a random_state for reproducibility


def generate_data_for_prompt_template(df, c_tf_idf, vocab, clusters, topn=10):
    # Get the top N most common words per class
    top_idx = np.argpartition(c_tf_idf, -topn)[:, -topn:]

    # Now we create a disctionary with cluster label as a key and top 10 most important words 
    # per class from c-TF-IDF step and 5 random sentences from each cluster as values.   
    vlist = list(vocab)
    data = {}

    for c, _cluster in enumerate(clusters.keys()):
        topn_idx = top_idx[c, :]
        topn_terms = [vlist[idx] for idx in topn_idx]
        
        # Get the sampled sentences for this cluster
        sampled_sentences = sample_sentences(df[df['cluster_id'] == _cluster])
        
        data[_cluster] = {
            'topn_terms': topn_terms,
            'sampled_sentences': sampled_sentences
        }

    return data


def get_topic_and_description_per_cluster(data_per_cluster):
    prompt = """I have a topic that is described by the following keywords: {keywords} 
                In this topic, the following documents are a small but representative subset of 
                all documents in the topic: {documents} 
                Based on the information above, please give a short label of the topic and a description of this topic 
                in the following format: 
                [Your Topic]: [Your Description]. 
                For example: 
                Login and Navigation Issues: This topic covers issues related to logging in, receiving invite codes, 
                adding phone numbers, navigation problems within the app.  
                Description should not exceed 20 words and should provide a concise summary."""

    # Set up your OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')
    openai.api_key = api_key

    # Empty list to store the data
    output_data = []

    for _cluster, cluster_data in data_per_cluster.items():
        if _cluster != -1:
            keywords = ', '.join(cluster_data['topn_terms'])
            documents = ', '.join(cluster_data['sampled_sentences'])

            prompt_for_cluster = prompt.format(keywords=keywords, documents=documents)

            # Use OpenAI API to generate a short label and description
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",  
                prompt=prompt_for_cluster,
                temperature=0.3,
                max_tokens=45
            )

            generated_text = response.choices[0].text.strip()
            
            # Assuming generated_text is in the format "topic: <description>"
            topic, description = generated_text.split(':')

            # Store the results in output_data
            output_data.append({'cluster_id': _cluster, 'group name': topic.strip(), 'group description': description.strip()})
            
    cluster_descriptions_df = pd.DataFrame(output_data)
    return cluster_descriptions_df


def add_description_for_outliers(data_per_cluster, cluster_descriptions_df):
    if -1 in data_per_cluster:
        # Create a new DataFrame with the specified values
        outlier_row = pd.DataFrame({'cluster_id': ['-1.0'], 
                                    'group name': ['Outlier'], 
                                    'group description': ['Each problem should be analysed separately.']})

        # Concatenate the new DataFrame with the existing cluster_name_description DataFrame
        cluster_descriptions_df = pd.concat([cluster_descriptions_df, outlier_row],ignore_index=True)
        return cluster_descriptions_df




dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize FastAPI
app = FastAPI()

# Initialize embedding model and NLTK 
embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define the endpoint
@app.post("/requests")
def process_requests(file: UploadFile = File(...)):
    try:
        print("Processing request...")

        # Save the uploaded file
        with open('./data/input.csv', 'wb') as f:
            f.write(file.file.read())

        # Load the CSV file
        df = pd.read_csv('./data/input.csv', sep='|')
        df.rename(columns={"reasson": "reason"}, inplace=True)
        df['combined_text'] = df['name'].str.cat(': ' + df['reason'])
        
        create_clusters(df, embed_model)

        df['preprocessed_text'] = df['combined_text'].apply(preprocess)

        # Identifying top 10 most important words per cluster using c-TF-IDF, along with 5 random documents 
        # from each cluster for the prompt to generative model.
        c_tf_idf, vocab, clusters = calculate_class_tf_idf(df)
        data_per_cluster = generate_data_for_prompt_template(df, c_tf_idf, vocab, clusters, topn=10)


        # Generate a topic and description for each cluster using the custom prompt.
        cluster_descriptions_df = get_topic_and_description_per_cluster(data_per_cluster)

        # If the class of outliers (-1) was identified by HDBSCAN, add the topic and description to the table.
        cluster_descriptions_df = add_description_for_outliers(data_per_cluster, cluster_descriptions_df)

        cluster_descriptions_df['cluster_id'] = cluster_descriptions_df['cluster_id'].astype(float).round().astype('int64')
        merged_df = pd.merge(df, cluster_descriptions_df, on='cluster_id', how='left')
        merged_df = merged_df[['id', 'name', 'reason', 'cluster_id', 'group name', 'group description']]
    
        # Save the output CSV
        output_path = './data/output.csv'
        merged_df.to_csv(output_path, index=False)

        return { 'output_path': output_path }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
            




