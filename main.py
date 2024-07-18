# app.py
import streamlit as st
from transformers import BertModel

from WebCrawler import WebCrawler
from DataChunker import DataChunker
from VectorDatabase import VectorDatabase
from HybridRetrieval import HybridRetrieval
from QuestionAnswering import QuestionAnswering


# Streamlit App
def main():
    st.title('QnA')

    st.sidebar.title('Settings')
    base_url = st.sidebar.text_input('Base URL', 'https://docs.nvidia.com/cuda/')
    depth_limit = st.sidebar.slider('Depth Limit', 1, 5, 5)
    query = st.sidebar.text_input('Query', 'Explain GPU Compatibility')

    if st.sidebar.button('Run Web Crawler'):
        st.write('Running web crawler...')
        try:
            crawler = WebCrawler(base_url, depth_limit)
            crawler.run()
            st.write(f'Web crawling completed. Crawled {len(crawler.data)} pages.')
            
            st.write('Processing data...')
            data = [text for url, text in crawler.data]
            chunker = DataChunker(data)
            
            st.write('Chunking data...')
            clusters = chunker.chunk_data()

            st.write('Generating embeddings...')
            model = BertModel.from_pretrained('bert-base-uncased')
            embeddings = [model.encode(chunk) for cluster, chunk in enumerate(clusters)]

            st.write('Storing embeddings in Milvus...')
            vector_db = VectorDatabase(collection_name="cuda_docs")
            vector_db.connect_milvus()
            vector_db.create_collection()
            metadata = [url for url, text in crawler.data]
            vector_db.insert_vectors(embeddings, metadata)
            st.write('Data stored in Milvus.')
            
            st.write('Retrieving and re-ranking data...')
            collection = vector_db.collection  # Get the collection object from VectorDatabase
            retrieval_model = HybridRetrieval(collection)
            
            st.write('Performing BM25 search...')
            bm25_scores = retrieval_model.bm25_search(query)
            
            st.write('Performing BERT-based search...')
            bert_scores = retrieval_model.bert_search(query)
            
            st.write('Combining and ranking results...')
            combined_scores = retrieval_model.combine_scores(bm25_scores, bert_scores)
            ranked_results = retrieval_model.rank_results(combined_scores, top_k=10)

            st.write('QuestionAnsweringnswers...')
            qa = QuestionAnswering()
            contexts = [result("metadata") for result in ranked_results]
            answers = [qa.get_answer(query, context) for context in contexts]
            
            st.write('Answers:')
            for i, answer in enumerate(answers):
                st.write(f"Answer {i+1}: {answer}")
        
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()