"""
Web scraper tool module for AnyDB MCP Server.

Contains functionality for:
- Web page scraping and content extraction
- URL validation and processing
- Integration with vector database for storage and retrieval
"""

import asyncio
import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import html2text
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from filetool import VectorDatabaseManager

# Get logger
logger = logging.getLogger('mcp_server')


class WebScraperManager:
    """Manages web scraping operations with robust error handling."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._create_session()
        
        # Configure html2text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # Don't wrap lines
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy and proper headers."""
        session = requests.Session()
        
        # Set up retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set common headers to avoid being blocked
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    def _validate_url(self, url: str) -> str:
        """Validate and normalize URL."""
        if not url.strip():
            raise ValueError("URL cannot be empty")
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Validate URL format
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")
        
        # Re-parse to get the final URL after normalization
        parsed = urllib.parse.urlparse(url)
        
        # Security check - block certain schemes and local addresses
        blocked_schemes = ['file', 'ftp', 'javascript']
        if parsed.scheme.lower() in blocked_schemes:
            raise ValueError(f"URL scheme '{parsed.scheme}' is not allowed")
        
        # Block local/private IP addresses for security
        blocked_patterns = [
            r'^localhost$',
            r'^127\.',
            r'^192\.168\.',
            r'^10\.',
            r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
            r'^\[::1\]$',
            r'^\[::'
        ]
        
        for pattern in blocked_patterns:
            if re.match(pattern, parsed.netloc.lower()):
                raise ValueError(f"Local/private URLs are not allowed: {url}")
        
        return url
    
    def _extract_content(self, html: str, url: str) -> Dict[str, str]:
        """Extract and clean content from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Extract title
        title_element = soup.find('title')
        title = title_element.get_text().strip() if title_element else ''
        if not title:
            title = urllib.parse.urlparse(url).netloc
        
        # Extract meta description
        meta_desc = ''
        meta_element = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_element:
            meta_desc = meta_element.get('content', '').strip()
        
        # Convert HTML to clean text
        text_content = self.html_converter.handle(str(soup))
        
        # Clean up the text
        lines = text_content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:  # Filter out very short lines
                cleaned_lines.append(line)
        
        clean_text = '\n'.join(cleaned_lines)
        
        # Truncate if too long (vector database limits)
        max_length = 50000  # Reasonable limit for vector storage
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "\n\n[Content truncated due to length...]"
        
        return {
            'title': title,
            'description': meta_desc,
            'content': clean_text,
            'url': url,
            'scraped_at': datetime.now().isoformat()
        }
    
    async def scrape_url(self, url: str) -> Dict[str, str]:
        """Scrape content from a URL asynchronously."""
        logger.info(f"Starting to scrape URL: {url}")
        
        try:
            # Validate URL
            validated_url = self._validate_url(url)
            logger.debug(f"Validated URL: {validated_url}")
            
            # Perform the web request in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.session.get(validated_url, timeout=self.timeout)
            )
            
            # Check response status
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                raise ValueError(f"URL does not return HTML content. Content-Type: {content_type}")
            
            # Extract content
            extracted = self._extract_content(response.text, validated_url)
            
            logger.info(f"Successfully scraped {len(extracted['content'])} characters from {validated_url}")
            return extracted
            
        except requests.RequestException as e:
            error_msg = f"Failed to fetch URL {url}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error scraping URL {url}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)


class WebScraperTools:
    """High-level web scraper tool operations."""
    
    def __init__(self, vector_db_manager: VectorDatabaseManager):
        self.scraper = WebScraperManager()
        self.vector_db = vector_db_manager
    
    async def scrape_and_store(self, url: str, custom_filename: Optional[str] = None) -> Dict[str, Any]:
        """Scrape a URL and store the content in the vector database."""
        logger.info(f"Scrape and Store - URL: {url}")
        
        try:
            # Scrape the URL
            scraped_data = await self.scraper.scrape_url(url)
            
            # Generate filename if not provided
            if not custom_filename:
                parsed_url = urllib.parse.urlparse(scraped_data['url'])
                domain = parsed_url.netloc.replace('www.', '')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                custom_filename = f"webpage_{domain}_{timestamp}.txt"
            
            # Create document content with metadata
            document_content = f"""Title: {scraped_data['title']}
URL: {scraped_data['url']}
Scraped: {scraped_data['scraped_at']}
Description: {scraped_data['description']}

{scraped_data['content']}"""
            
            # Prepare metadata for vector storage
            metadata = {
                'source': 'web_scraper',
                'url': scraped_data['url'],
                'title': scraped_data['title'],
                'description': scraped_data['description'],
                'scraped_at': scraped_data['scraped_at'],
                'content_length': len(scraped_data['content'])
            }
            
            # Store in vector database
            chunks_added = await self.vector_db.add_file(
                filename=custom_filename,
                content=document_content,
                metadata=metadata
            )
            
            response_data = {
                'url': scraped_data['url'],
                'title': scraped_data['title'],
                'filename': custom_filename,
                'content_length': len(scraped_data['content']),
                'chunks_added': chunks_added,
                'message': f"Successfully scraped and stored webpage as '{custom_filename}'"
            }
            
            logger.info(f"Successfully stored scraped content with {chunks_added} chunks")
            return response_data
            
        except Exception as e:
            error_msg = f"Failed to scrape and store URL {url}: {str(e)}"
            logger.error(error_msg)
            return {
                'url': url,
                'error': error_msg,
                'message': 'Failed to scrape webpage'
            }
    
    async def query_scraped_content(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Query the vector database for scraped web content."""
        logger.info(f"Querying scraped content: {query}")
        
        try:
            # Search the vector database
            search_results = await self.vector_db.search_files(query, max_results)
            
            # Filter results to only include web scraped content
            web_results = []
            for result in search_results:
                metadata = result.get('metadata', {})
                if metadata.get('source') == 'web_scraper':
                    web_results.append({
                        'filename': result['filename'],
                        'content': result['content'],
                        'similarity_score': result['similarity_score'],
                        'url': metadata.get('url', ''),
                        'title': metadata.get('title', ''),
                        'scraped_at': metadata.get('scraped_at', '')
                    })
            
            response_data = {
                'query': query,
                'results_found': len(web_results),
                'results': web_results,
                'message': f"Found {len(web_results)} relevant web pages"
            }
            
            logger.info(f"Query returned {len(web_results)} web scraping results")
            return response_data
            
        except Exception as e:
            error_msg = f"Failed to query scraped content: {str(e)}"
            logger.error(error_msg)
            return {
                'query': query,
                'error': error_msg,
                'message': 'Failed to search scraped content'
            }
    
    async def list_scraped_pages(self) -> Dict[str, Any]:
        """List all scraped web pages stored in the vector database."""
        logger.info("Listing all scraped web pages")
        
        try:
            # Get all files from vector database
            all_files = await self.vector_db.list_files()
            
            # Filter for web scraped content
            web_pages = []
            for file_info in all_files:
                metadata = file_info.get('metadata', {})
                if metadata.get('source') == 'web_scraper':
                    web_pages.append({
                        'filename': file_info['filename'],
                        'url': metadata.get('url', ''),
                        'title': metadata.get('title', 'No title'),
                        'description': metadata.get('description', ''),
                        'scraped_at': metadata.get('scraped_at', ''),
                        'content_length': metadata.get('content_length', 0),
                        'chunk_count': file_info.get('chunk_count', 0)
                    })
            
            response_data = {
                'total_pages': len(web_pages),
                'pages': web_pages,
                'message': f"Found {len(web_pages)} scraped web pages"
            }
            
            logger.info(f"Listed {len(web_pages)} scraped web pages")
            return response_data
            
        except Exception as e:
            error_msg = f"Failed to list scraped pages: {str(e)}"
            logger.error(error_msg)
            return {
                'error': error_msg,
                'message': 'Failed to list scraped web pages'
            }
    
    async def remove_scraped_page(self, filename: str) -> Dict[str, Any]:
        """Remove a scraped web page from the vector database."""
        logger.info(f"Removing scraped page: {filename}")
        
        try:
            # Remove from vector database
            success = await self.vector_db.remove_file(filename)
            
            if success:
                response_data = {
                    'filename': filename,
                    'message': f"Successfully removed scraped page '{filename}'"
                }
                logger.info(f"Successfully removed scraped page: {filename}")
            else:
                response_data = {
                    'filename': filename,
                    'message': f"Scraped page '{filename}' not found or already removed"
                }
                logger.warning(f"Scraped page not found: {filename}")
            
            return response_data
            
        except Exception as e:
            error_msg = f"Failed to remove scraped page {filename}: {str(e)}"
            logger.error(error_msg)
            return {
                'filename': filename,
                'error': error_msg,
                'message': 'Failed to remove scraped page'
            }