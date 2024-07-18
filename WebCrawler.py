import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz 

# Web Crawler
class WebCrawler:
    def __init__(self, base_url, depth_limit=5):
        self.base_url = base_url
        self.depth_limit = depth_limit
        self.visited_urls = set()
        self.data = []

    def fetch_url(self, url, depth):
        if depth > self.depth_limit or url in self.visited_urls:
            return
        try:
            response = requests.get(url)
            if response.status_code == 200:
                self.visited_urls.add(url)
                content_type = response.headers.get('content-type', '').lower()
                
                if 'text/html' in content_type:
                    # Parse HTML content
                    soup = BeautifulSoup(response.content, 'html.parser')
                    self.data.append((url, soup.get_text()))
                    
                    # Find links in HTML for further crawling
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        self.fetch_url(next_url, depth + 1)
                
                elif 'application/pdf' in content_type:
                    # Parse PDF content
                    doc = fitz.open(stream=response.content, filetype='pdf')
                    pdf_text = ''
                    for page_num in range(len(doc)):
                        page_text = doc[page_num].get_text()
                        pdf_text += page_text
                    self.data.append((url, pdf_text))
                
                
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    def run(self):
        self.fetch_url(self.base_url, 0)
