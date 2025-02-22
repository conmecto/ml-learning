import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urljoin, urlparse
from PIL import Image
from io import BytesIO

class WebScraper:
    def __init__(self, url):
        self.url = url
        self.base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(url))
        self.soup = None
        self.min_width = 500
        self.min_height = 500
    
    def fetch_page(self):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.url, headers=headers)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.text, 'html.parser')
            return True
        except requests.RequestException as e:
            print(f"Error fetching the webpage: {e}")
            return False

    def get_image_dimensions(self, img_url):
        try:
            response = requests.get(img_url, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img.size
        except Exception as e:
            print(f"Error checking dimensions for {img_url}: {e}")
            return None

    def is_image_large_enough(self, img_url):
        dimensions = self.get_image_dimensions(img_url)
        if dimensions:
            width, height = dimensions
            return width >= self.min_width and height >= self.min_height
        return False

    def extract_text_content(self):
        if not self.soup:
            return {}
        
        text_content = {
            'paragraphs': [p.get_text().strip() for p in self.soup.find_all('p')],
            'headers': [h.get_text().strip() for h in self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])],
            'lists': [li.get_text().strip() for li in self.soup.find_all('li')],
            'links': [{'text': a.get_text().strip(), 'url': a.get('href')} 
                      for a in self.soup.find_all('a') if a.get('href')]
        }
        
        return text_content

    def extract_large_images(self):
        if not self.soup:
            return []
        
        images = []
        for img in self.soup.find_all('img'):
            src = img.get('src')
            alt = img.get('alt', '')
            
            if src:
                if not urlparse(src).netloc:
                    src = urljoin(self.base_url, src)
                
                print(f"Checking image: {src}")
                if self.is_image_large_enough(src):
                    dimensions = self.get_image_dimensions(src)
                    images.append({
                        'url': src,
                        'alt_text': alt,
                        'width': dimensions[0] if dimensions else None,
                        'height': dimensions[1] if dimensions else None
                    })
                    print(f"Found large image: {src} ({dimensions[0]}x{dimensions[1]})")
        
        return images

    def download_images(self, folder='downloaded_images'):
        images = self.extract_large_images()
        if not images:
            print("No large images found to download.")
            return []
        
        os.makedirs(folder, exist_ok=True)
        
        downloaded_images = []
        for i, img in enumerate(images):
            try:
                response = requests.get(img['url'], stream=True)
                response.raise_for_status()
                
                file_extension = os.path.splitext(urlparse(img['url']).path)[1] or '.jpg'
                filename = f"large_image_{i+1}{file_extension}"
                filepath = os.path.join(folder, filename)
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_images.append({
                    'original_url': img['url'],
                    'local_path': filepath,
                    'alt_text': img['alt_text'],
                    'dimensions': f"{img['width']}x{img['height']}"
                })
                print(f"Downloaded: {filename} ({img['width']}x{img['height']})")
                
            except requests.RequestException as e:
                print(f"Failed to download {img['url']}: {e}")
        
        return downloaded_images

def save_to_csv(text_content, images, filename='scraped_data.csv'):
    rows = []
    
    for content_type, items in text_content.items():
        if content_type == 'links':
            for item in items:
                rows.append({
                    'Type': 'Link',
                    'Content': item['text'],
                    'URL': item['url'],
                    'Dimensions': ''
                })
        else:
            for item in items:
                rows.append({
                    'Type': content_type.capitalize(),
                    'Content': item,
                    'URL': '',
                    'Dimensions': ''
                })
    
    for img in images:
        rows.append({
            'Type': 'Large Image',
            'Content': img['alt_text'],
            'URL': img['url'],
            'Dimensions': f"{img['width']}x{img['height']}"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():    
    #url = input("Enter the URL to scrape: ")
    # url = 'https://www.boat-lifestyle.com/products/airdopes-alpha-true-wireless-earbuds'
    url = 'https://mamaearth.in/product/ubtan-face-wash-with-turmeric-saffron-for-tan-removal-150-ml'

    scraper = WebScraper(url)
    
    if scraper.fetch_page():
        text_content = scraper.extract_text_content()
        large_images = scraper.extract_large_images()
        
        print("\nContent Summary:")
        print(f"Paragraphs: {len(text_content['paragraphs'])}")
        print(f"Headers: {len(text_content['headers'])}")
        print(f"List items: {len(text_content['lists'])}")
        print(f"Links: {len(text_content['links'])}")
        print(f"Large Images (>500x500): {len(large_images)}")
        
        if input("\nSave data to CSV? (y/n): ").lower() == 'y':
            filename = input("Enter CSV filename (default: scraped_data.csv): ").strip() or 'scraped_data.csv'
            save_to_csv(text_content, large_images, filename)
        
        if input("Download large images? (y/n): ").lower() == 'y':
            folder = input("Enter download folder (default: downloaded_images): ").strip() or 'downloaded_images'
            scraper.download_images(folder)

if __name__ == "__main__":
    main()