# StreamlitNLP

# CUDA Documentation Web Scraper and QA System

## Project Overview
This project implements a web scraping and question-answering (QA) system focused on NVIDIA's CUDA documentation. It includes functionalities for web crawling, data chunking, vector database creation using MILVUS, retrieval, re-ranking, and question answering using hybrid methods combining BM25 and BERT-based approaches.

## Setup Instructions
To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
     git clone https://github.com/gal-actic/StreamlitNLP
     cd StreamlitNLP

3. **Install dependencies:**
   ```bash
     pip install -r requirements.txt

Ensure Python 3.x is installed on your system.

3. **Set up MILVUS:**
- Install MILVUS by following the instructions on [MILVUS official site](https://milvus.io/docs/v2.0.0/install_milvus.md).

## Usage
Once set up, you can interact with the project through the Streamlit user interface:

1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py

Access the Streamlit app via the provided URL. Enter queries and explore the retrieved and ranked results.

## Dependencies
- `transformers==4.13.0`
- `pymilvus==2.1.0`
- `sentence-transformers==2.1.0`
- `scikit-learn==0.24.2`
- `streamlit==1.0.0`

Ensure these Python packages are installed in your environment.

## Interface
The project includes a Streamlit-based user interface for easy interaction:

- **Access the UI:**
Open a web browser and navigate to the provided URL after running `streamlit run app.py`. Input queries to generate answers based on the retrieved data.

## Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **Vaishnavi Singh**
- **Email:** galactic200624@gmail.com.
  
## Copyright
© 2024 gal-actic. All rights reserved.
