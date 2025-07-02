#!/usr/bin/env python3
"""
Simple Domain Port Scanner with Web Screenshot Capture
Reads domains from text file and outputs to Excel with screenshots
"""

import asyncio
import socket
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import concurrent.futures
from playwright.async_api import async_playwright
import pandas as pd

# =============================================================================
# Logger Setup Class
# =============================================================================
class LoggerSetup:
    @staticmethod
    def setup_logging() -> logging.Logger:
        """Setup simple logging configuration"""
        # Get script directory
        script_dir = Path(__file__).parent.absolute()
        log_dir = script_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('domain_scanner')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        log_file = log_dir / f'domain_scanner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

# =============================================================================
# DNS Resolver Class
# =============================================================================
class DNSResolver:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timeout = 5
    
    def resolve_domain(self, domain: str) -> Optional[str]:
        """Resolve domain to IP address"""
        try:
            ip_address = socket.gethostbyname(domain)
            self.logger.info(f"Domain {domain} resolves to IP: {ip_address}")
            return ip_address
        except socket.gaierror as e:
            self.logger.debug(f"DNS resolution failed for {domain}: {e}")
            return None
        except Exception as e:
            self.logger.debug(f"Error resolving {domain}: {e}")
            return None
class PortScanner:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timeout = 5
    
    def check_port(self, domain: str, port: int) -> bool:
        """Check if a specific port is open on a domain"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((domain, port))
            sock.close()
            return result == 0
        except Exception as e:
            self.logger.debug(f"Error checking {domain}:{port} - {e}")
            return False
    
    def scan_domain(self, domain: str) -> Dict:
        """Scan a domain for ports 80 and 443"""
        self.logger.info(f"Scanning domain: {domain}")
        
        port_80_open = self.check_port(domain, 80)
        port_443_open = self.check_port(domain, 443)
        
        # Create URLs for screenshot capture - prioritize HTTPS, avoid duplicates
        accessible_urls = []
        if port_443_open:
            # If HTTPS is available, only capture HTTPS version
            accessible_urls.append(f"https://{domain}")
        elif port_80_open:
            # Only capture HTTP if HTTPS is not available
            accessible_urls.append(f"http://{domain}")
        
        result = {
            'domain': domain,
            'port_80': 'Yes' if port_80_open else 'No',
            'port_443': 'Yes' if port_443_open else 'No',
            'accessible_urls': accessible_urls,
            'has_open_ports': port_80_open or port_443_open
        }
        
        self.logger.info(f"Domain {domain} - Port 80: {result['port_80']}, Port 443: {result['port_443']}")
        return result
    
    async def scan_domains_parallel(self, domains: List[str]) -> List[Dict]:
        """Scan multiple domains in parallel"""
        max_workers = 20
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self.scan_domain, domain)
                for domain in domains
            ]
            results = await asyncio.gather(*tasks)
        
        return results

# =============================================================================
# Web Screenshot Capture Class
# =============================================================================
class WebCapture:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.screenshot_timeout = 30000
        self.viewport_width = 1920
        self.viewport_height = 1080
    
    async def capture_screenshot(self, url: str, output_dir: Path) -> Dict:
        """Capture screenshot of a webpage by simulating browser behavior"""
        # Extract domain from URL for cleaner logging
        domain = url.replace('http://', '').replace('https://', '').split('/')[0]
        
        self.logger.info(f"Capturing screenshot for: {domain}")
        
        # Initialize result dictionary
        result = {
            'success': False,
            'final_url': None,
            'page_title': None,
            'screenshot_path': None
        }
        
        # Browser-like URL attempts - start simple and let redirects handle the rest
        urls_to_try = [
            f"https://{domain}",      # Try HTTPS first (modern standard)
            f"https://www.{domain}",  # Try with www prefix
            f"http://{domain}",       # Try HTTP
            f"http://www.{domain}",   # Try HTTP with www
        ]
        
        for attempt_url in urls_to_try:
            try:
                async with async_playwright() as p:
                    # Launch browser with more realistic settings
                    browser = await p.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-web-security',
                        ]
                    )
                    
                    context = await browser.new_context(
                        viewport={'width': self.viewport_width, 'height': self.viewport_height},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
                    )
                    
                    page = await context.new_page()
                    
                    # Set longer timeout and wait for redirects
                    try:
                        # Navigate and wait for the page to fully load
                        response = await page.goto(
                            attempt_url, 
                            timeout=self.screenshot_timeout,
                            wait_until='domcontentloaded'
                        )
                        
                        # Wait a bit more for any JavaScript redirects
                        await asyncio.sleep(2)
                        
                        # Wait for network to be idle (all resources loaded)
                        await page.wait_for_load_state('networkidle', timeout=15000)
                        
                        # Get the final URL after all redirects
                        final_url = page.url
                        page_title = await page.title()
                        
                        self.logger.info(f"Successfully navigated: {domain}")
                        self.logger.info(f"Final URL: {final_url}")
                        self.logger.info(f"Page title: {page_title}")
                        
                        # Create screenshot filename using original domain and timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Clean domain name for filename
                        clean_domain = domain.replace('.', '_')
                        screenshot_filename = f"{clean_domain}_{timestamp}.png"
                        screenshot_path = output_dir / 'screenshots' / screenshot_filename
                        
                        # Ensure screenshots directory exists
                        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Take full page screenshot
                        await page.screenshot(
                            path=str(screenshot_path), 
                            full_page=True,
                            timeout=10000
                        )
                        
                        await browser.close()
                        
                        self.logger.info(f"Screenshot saved: {screenshot_filename}")
                        
                        # Update result with success information
                        result.update({
                            'success': True,
                            'final_url': final_url,
                            'page_title': page_title,
                            'screenshot_path': screenshot_filename
                        })
                        
                        return result
                        
                    except Exception as page_error:
                        await browser.close()
                        self.logger.debug(f"Failed to navigate {domain}: {page_error}")
                        continue
                        
            except Exception as browser_error:
                self.logger.debug(f"Browser error for {domain}: {browser_error}")
                continue
        
        # If all attempts failed
        self.logger.error(f"Could not capture screenshot for {domain}")
        return result
    
    async def capture_timeout_evidence(self, domain: str, ip_address: str, output_dir: Path) -> Dict:
        """Capture authentic browser timeout error by using non-headless browser"""
        self.logger.info(f"Capturing authentic timeout evidence for: {domain} ({ip_address})")
        
        result = {
            'success': False,
            'final_url': 'Connection Timeout',
            'page_title': f'No Response - IP: {ip_address}',
            'screenshot_path': None,
            'error_type': 'timeout'
        }
        
        urls_to_try = [f"https://{domain}", f"http://{domain}"]
        
        for attempt_url in urls_to_try:
            try:
                async with async_playwright() as p:
                    # Use non-headless browser to get authentic error pages
                    browser = await p.chromium.launch(
                        headless=False,  # Run visible browser to get real error pages
                        args=[
                            '--no-sandbox',
                            '--disable-setuid-sandbox',
                            '--disable-dev-shm-usage',
                            '--disable-web-security',
                            '--no-first-run',
                            '--disable-extensions',
                        ]
                    )
                    
                    context = await browser.new_context(
                        viewport={'width': self.viewport_width, 'height': self.viewport_height},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
                    )
                    
                    page = await context.new_page()
                    
                    # Create screenshot filename first
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clean_domain = domain.replace('.', '_')
                    screenshot_filename = f"{clean_domain}_timeout_{timestamp}.png"
                    screenshot_path = output_dir / 'screenshots' / screenshot_filename
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        self.logger.info(f"Attempting to load {attempt_url} in visible browser...")
                        
                        # Use a reasonable timeout that allows browser to show error
                        response = await page.goto(
                            attempt_url, 
                            timeout=20000,  # 20 seconds
                            wait_until='commit'  # Just wait for navigation to start
                        )
                        
                        # If we get here, check if page actually loaded or is showing error
                        await asyncio.sleep(3)  # Wait for page to fully render
                        
                        current_url = page.url
                        page_title = await page.title()
                        
                        self.logger.info(f"Page loaded - URL: {current_url}, Title: {page_title}")
                        
                        # Take screenshot regardless - could be error page or actual content
                        await page.screenshot(path=str(screenshot_path), full_page=True, timeout=10000)
                        
                        result.update({
                            'success': True,
                            'final_url': current_url,
                            'page_title': page_title,
                            'screenshot_path': screenshot_filename,
                            'error_type': 'loaded' if response and response.ok else 'error'
                        })
                        
                        self.logger.info(f"Captured browser response: {screenshot_filename}")
                        await browser.close()
                        return result
                        
                    except Exception as navigation_error:
                        # Browser timed out or failed - this should show authentic error page
                        self.logger.info(f"Navigation failed for {attempt_url}, capturing error page...")
                        
                        try:
                            # Wait a moment for browser to display its error page
                            await asyncio.sleep(5)
                            
                            # Try to get error page information
                            try:
                                current_url = page.url
                                page_title = await page.title()
                                self.logger.info(f"Error page - URL: {current_url}, Title: {page_title}")
                            except:
                                current_url = attempt_url
                                page_title = "Browser Error"
                            
                            # Take screenshot of whatever browser is showing
                            await page.screenshot(path=str(screenshot_path), full_page=True, timeout=10000)
                            
                            self.logger.info(f"Captured authentic browser error: {screenshot_filename}")
                            
                            result.update({
                                'success': True,
                                'final_url': f'TIMEOUT: {attempt_url}',
                                'page_title': page_title,
                                'screenshot_path': screenshot_filename,
                                'error_type': 'timeout'
                            })
                            
                            await browser.close()
                            return result
                            
                        except Exception as screenshot_error:
                            self.logger.error(f"Failed to capture error screenshot: {screenshot_error}")
                            await browser.close()
                            continue
                        
            except Exception as browser_error:
                self.logger.error(f"Browser setup error for {domain}: {browser_error}")
                continue
        
        self.logger.error(f"Could not capture authentic timeout evidence for {domain}")
        return result
    
    async def _create_timeout_evidence_page(self, domain: str, ip_address: str, error_message: str, screenshot_path: Path):
        """Create an HTML page documenting the timeout and capture it as evidence"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Connection Timeout Evidence - {domain}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .header {{ color: #d32f2f; border-bottom: 2px solid #d32f2f; padding-bottom: 10px; }}
                    .info {{ margin: 20px 0; }}
                    .error {{ background-color: #ffebee; padding: 15px; border-left: 4px solid #d32f2f; margin: 10px 0; }}
                    .timestamp {{ color: #666; font-size: 14px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="header">Connection Timeout Evidence</h1>
                    <div class="info">
                        <p><strong>Domain:</strong> {domain}</p>
                        <p><strong>IP Address:</strong> {ip_address}</p>
                        <p><strong>Status:</strong> Domain resolves to IP but ports 80/443 are not responding</p>
                        <p class="timestamp"><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    <div class="error">
                        <h3>Connection Details:</h3>
                        <p>Attempted to connect to both HTTP (port 80) and HTTPS (port 443) but received no response.</p>
                        <p><strong>Error:</strong> {error_message}</p>
                    </div>
                    <div class="info">
                        <p><strong>Evidence Notes:</strong></p>
                        <ul>
                            <li>DNS resolution successful - domain points to {ip_address}</li>
                            <li>Port scanning shows no HTTP/HTTPS services running</li>
                            <li>Connection attempts result in timeout</li>
                            <li>This may indicate: server down, firewall blocking, or services not running on standard ports</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={'width': self.viewport_width, 'height': self.viewport_height}
                )
                page = await context.new_page()
                
                # Set content and take screenshot
                await page.set_content(html_content)
                await page.screenshot(path=str(screenshot_path), full_page=True)
                
                await browser.close()
                
        except Exception as e:
            self.logger.error(f"Failed to create timeout evidence page: {e}")
    async def capture_multiple_screenshots(self, urls: List[str], domains_with_ips: List[Dict], output_dir: Path) -> Dict:
        """Capture screenshots for multiple URLs and timeout evidence for domains with IPs but no open ports"""
        max_concurrent = 3
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def capture_with_semaphore(url):
            async with semaphore:
                return await self.capture_screenshot(url, output_dir)
        
        async def capture_timeout_with_semaphore(domain_info):
            async with semaphore:
                return await self.capture_timeout_evidence(
                    domain_info['domain'], 
                    domain_info['ip_address'], 
                    output_dir
                )
        
        # Capture screenshots for accessible URLs
        screenshot_tasks = [capture_with_semaphore(url) for url in urls]
        screenshot_results = await asyncio.gather(*screenshot_tasks)
        
        # Capture timeout evidence for domains with IPs but no open ports
        timeout_tasks = [capture_timeout_with_semaphore(domain_info) for domain_info in domains_with_ips]
        timeout_results = await asyncio.gather(*timeout_tasks)
        
        # Create a lookup dictionary by domain
        screenshot_data = {}
        
        # Add regular screenshot results
        for i, url in enumerate(urls):
            domain = url.replace('http://', '').replace('https://', '').split('/')[0]
            screenshot_data[domain] = screenshot_results[i]
        
        # Add timeout evidence results
        for i, domain_info in enumerate(domains_with_ips):
            domain = domain_info['domain']
            screenshot_data[domain] = timeout_results[i]
        
        return screenshot_data

# =============================================================================
# Domain Input Handler Class
# =============================================================================
class DomainInputHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.input_dir = Path('input')
        self.input_dir.mkdir(exist_ok=True)
    
    def load_domains_from_file(self, filename: str) -> List[str]:
        """Load domains from a text file (one domain per line)"""
        file_path = self.input_dir / filename
        domains = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    domain = line.strip()
                    if domain and not domain.startswith('#'):
                        domains.append(domain)
            
            self.logger.info(f"Loaded {len(domains)} domains from {filename}")
            return domains
            
        except Exception as e:
            self.logger.error(f"Error loading domains from {filename}: {e}")
            return []

# =============================================================================
# Excel Export Class
# =============================================================================
class ExcelExporter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
    
    def export_to_excel(self, scan_results: List[Dict], screenshot_data: Dict, dns_results: Dict) -> str:
        """Export results to Excel file with final URLs and IP addresses"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"domain_scan_results_{timestamp}.xlsx"
        excel_path = self.output_dir / excel_filename
        
        try:
            # Create DataFrame with the enhanced columns
            df_data = []
            for result in scan_results:
                domain = result['domain']
                screenshot_info = screenshot_data.get(domain, {})
                ip_address = dns_results.get(domain, 'No DNS Resolution')
                
                # Determine status
                if result['has_open_ports']:
                    status = 'Accessible'
                elif ip_address != 'No DNS Resolution':
                    status = 'DNS Resolves - No Web Services'
                else:
                    status = 'No DNS Resolution'
                
                df_data.append({
                    'Domain Name': result['domain'],
                    'IP Address': ip_address,
                    'Port 80': result['port_80'],
                    'Port 443': result['port_443'],
                    'Status': status,
                    'Final URL': screenshot_info.get('final_url', 'N/A'),
                    'Page Title': screenshot_info.get('page_title', 'N/A'),
                    'Evidence Captured': 'Yes' if screenshot_info.get('success', False) else 'No'
                })
            
            df = pd.DataFrame(df_data)
            
            # Export to Excel
            df.to_excel(excel_path, index=False, engine='openpyxl')
            
            self.logger.info(f"Results exported to Excel: {excel_path}")
            return str(excel_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            return ""

# =============================================================================
# Main Application Class
# =============================================================================
class DomainScannerApp:
    def __init__(self):
        # Get script directory for relative paths
        self.script_dir = Path(__file__).parent.absolute()
        os.chdir(self.script_dir)
        
        # Set up directory paths
        self.output_dir = Path('output')
        
        # Setup logging
        self.logger = LoggerSetup.setup_logging()
        
        # Initialize components
        self.dns_resolver = DNSResolver(self.logger)
        self.port_scanner = PortScanner(self.logger)
        self.web_capture = WebCapture(self.logger)
        self.domain_handler = DomainInputHandler(self.logger)
        self.excel_exporter = ExcelExporter(self.logger)
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = ['input', 'output', 'logs']
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    async def run_scan(self, domains: List[str]):
        """Run the complete scanning process"""
        if not domains:
            self.logger.error("No domains provided for scanning")
            return
        
        self.logger.info(f"Starting scan for {len(domains)} domains")
        
        # Step 1: DNS Resolution
        self.logger.info("Starting DNS resolution...")
        dns_results = {}
        for domain in domains:
            ip_address = self.dns_resolver.resolve_domain(domain)
            if ip_address:
                dns_results[domain] = ip_address
        
        # Step 2: Scan ports
        self.logger.info("Starting port scanning...")
        scan_results = await self.port_scanner.scan_domains_parallel(domains)
        
        # Step 3: Collect URLs for screenshot capture and domains needing timeout evidence
        urls_to_capture = []
        domains_for_timeout_evidence = []
        
        for result in scan_results:
            domain = result['domain']
            if result['has_open_ports']:
                # Domain has open ports - capture normal screenshots
                urls_to_capture.extend(result['accessible_urls'])
            elif domain in dns_results:
                # Domain resolves but has no open ports - capture timeout evidence
                domains_for_timeout_evidence.append({
                    'domain': domain,
                    'ip_address': dns_results[domain]
                })
                self.logger.info(f"Domain {domain} resolves to {dns_results[domain]} but has no open HTTP/HTTPS ports")
        
        accessible_count = len(urls_to_capture)
        timeout_count = len(domains_for_timeout_evidence)
        
        self.logger.info(f"Found {accessible_count} accessible URLs and {timeout_count} domains needing timeout evidence")
        
        # Step 4: Capture screenshots and timeout evidence
        screenshot_data = {}
        if urls_to_capture or domains_for_timeout_evidence:
            self.logger.info("Starting evidence capture...")
            screenshot_data = await self.web_capture.capture_multiple_screenshots(
                urls_to_capture, domains_for_timeout_evidence, self.output_dir
            )
        
        # Step 5: Export to Excel
        excel_path = self.excel_exporter.export_to_excel(scan_results, screenshot_data, dns_results)
        
        self.logger.info("Scan completed successfully")
        self.logger.info(f"Excel results: {excel_path}")
        self.logger.info(f"Evidence saved in: {self.output_dir / 'screenshots'}")
    
    def run_from_file(self, filename: str = 'domains.txt'):
        """Run scan from input text file"""
        domains = self.domain_handler.load_domains_from_file(filename)
        
        if not domains:
            self.logger.error("No domains loaded from file")
            return
        
        # Run the scan
        asyncio.run(self.run_scan(domains))

# =============================================================================
# Main Function
# =============================================================================
def main():
    """Main function to run the application"""
    app = DomainScannerApp()
    
    # Run from domains.txt file in input directory
    app.run_from_file('domains.txt')

if __name__ == "__main__":
    main()