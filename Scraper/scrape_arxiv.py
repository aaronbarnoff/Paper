# This script fetches papers from the arxiv external server, using their API.
#     More on their API: https://info.arxiv.org/help/api/index.html and https://info.arxiv.org/help/api/user-manual.html
# Basic error checking and delays are implemented to avoid violating their flow control and DDoS protection.
# The script creates an arxivPapers folder and creates a .txt file for each record listing the metadata.
#      The arXiv metadata captured includes: ID, title, authors, creation date, categories, abstract, pdf Links

import feedparser
import urllib
import os
import time
import logging
import re

def removeLatex(text): # remove latex now rather than later
    # Remove LaTeX commands
    text = re.sub(r'\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^\}]*\})?', '', text)
    # Remove inline math
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\\((.*?)\\\)', '', text)
    # Remove display math
    text = re.sub(r'\\\[(.*?)\\\]', '', text)
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)
    return text

start = 0000       # Which result to start at, e.g. 0 or 10001
amount =  2000  # How many to retrieve per request; they limit this to 2000
end = 10000     # Which result to end at, e.g. 10000 or 20000

# List of all the computer science categories
csCats = "AI AR CC CE CG CL CR CV CY DB DC DL DM DS ET" 
"FL GL GR GT HC IR IT LG LO MA MM MS NA NE NI OH"
"OS PF PL RO SC SD SE SI SY"

# Create the query to search all cs categories e.g. "cat:cs.AI OR cat:.SE OR ..."
catQuery = ''.join(['cat:cs.' + cat + '+OR+' for cat in csCats.split(" ")]).removesuffix('+OR+')

# Implement some basic logging, especially to ensure the server requests are correct
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_filename = 'arxiv_fetch.log'
logging.basicConfig(
    level=logging.INFO,      # Set the minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log messages        
)

papersFolder = "arxivPapers"
os.makedirs(papersFolder, exist_ok=True)

# These settings are to help avoid triggering any kind of DDoS protection (from 429 or 503 "retry-after" requests)
request_delay = 3   # They specify a 3 second delay minimum
backoff_factor = 2  # Basic exponential backoff 
max_retries = 3     # For non 503 errors

base_url = 'http://export.arxiv.org/api/query?' # base URL for arXiv API

while start < end:
    query = 'search_query={}&start={}&max_results={}'.format(catQuery, start, amount)
    full_url = base_url + query
    logging.info(f"Requesting URL: {full_url}")
    try:
        for attempt in range(max_retries):
            response = urllib.request.urlopen(full_url)

            # Check status codes
            if response.getcode() == 200:  # Request acknowledged.
                break  
            elif response.getcode() in [429, 503]:  # This is to avoid their server interpreting our requests as a DDoS attack.
                retry_after = int(response.headers.get("Retry-After", request_delay))
                logging.warning(f"{response.getcode()} Too Many Requests. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
            elif 500 <= response.getcode() < 600:  # Basic exponential backoff to avoid spamming them.
                logging.warning(f"Server error {response.getcode()}. Retrying...")
                time.sleep(request_delay * (backoff_factor ** attempt))
            else:
                logging.error(f"Unexpected status code {response.getcode()}. Stopping requests.")
                break
        else:
            logging.error("Max retries reached. Stopping requests.")
            break

        # Parse the response using feedparser
        feed = feedparser.parse(response)
        logging.info(f"Fetched {len(feed.entries)} papers.")

        # Print out feed information
        if start == 0:
            logging.info('Feed title: {}'.format(feed.feed.title))
            logging.info('Feed last updated: {}'.format(feed.feed.updated))
            logging.info('Total results: {}'.format(feed.feed.opensearch_totalresults))
            logging.info('Items per page: {}'.format(feed.feed.opensearch_itemsperpage))
        logging.info('Start index: {}'.format(feed.feed.opensearch_startindex))

        # Run through each entry and write out information
        for entry in feed.entries:
            author_string = entry.author
            
            try:
                author_string += ' ({})'.format(entry.arxiv_affiliation)  # author's affiliations
            except AttributeError:
                pass

            # Make file in index for this paper
            fileName = os.path.join(papersFolder, f"arxiv_{entry.id.split('/abs/')[-1].replace('/', '_')}.txt")
            if os.path.exists(fileName):
                logging.info('Paper already exists.')
                continue

            with open(f'{fileName}', "w", encoding="utf-8") as file:
                # Record metadata to file in a way that is easy to index
                id = entry.id.replace("\n", ' ')  # Remove newlines from the entries, helpful for indexing
                published = entry.published.replace('\n', ' ')
                updated = entry.updated.replace('\n', ' ')
                title = re.sub(r'\s+', ' ', entry.title.replace('\n', ' '))

                file.write(f"Arxiv_ID:{removeLatex(id)}\n")
                file.write(f"Published:{removeLatex(published)}\n")
                file.write(f"Updated:{removeLatex(updated)}\n")
                file.write(f"Title:{removeLatex(title)}\n")

                try:
                    author_names = ', '.join(author.name.replace('\n', ' ') for author in entry.authors)
                    file.write(f"Authors:{removeLatex(author_names)}\n")
                except AttributeError:
                    file.write(f"Authors:None\n")

                try:
                    doi = entry.arxiv_doi.replace('\n', ' ')
                except AttributeError:
                    doi = 'None'
                file.write(f"DOI:{doi}\n")

                # Get the links to the abs page and PDF for this e-print
                for link in entry.links:
                    linkR = link.href.replace('\n', ' ')
                    if link.rel == 'alternate':
                        file.write(f"Abstract_Link:{removeLatex(linkR)}\n")
                    elif link.title == 'pdf':
                        file.write(f"PDF_Link:{removeLatex(linkR)}\n")     

                try:
                    journal_ref = entry.arxiv_journal_ref.replace('\n', ' ')  # Journal reference to arxiv paper
                except AttributeError:
                    journal_ref = 'None'
                file.write(f"Journal_Ref:{removeLatex(journal_ref)}\n")

                try:
                    comment = entry.arxiv_comment.replace('\n', ' ')  # Author's comment
                except AttributeError:
                    comment = 'None'
                file.write(f"Comments:{removeLatex(comment)}\n")
        
                pCat = entry.tags[0]['term'].replace('\n', ' ')
                file.write(f"Primary_Category:{removeLatex(pCat)}\n")
                all_categories = [removeLatex(t['term'].replace('\n', ' ')) for t in entry.tags]
                file.write(f"All_Categories:{', '.join(all_categories)}\n")

                abs = entry.summary.replace('\n', ' ')
                file.write(f"Abstract:{removeLatex(abs)}\n")

        # Increment start after successfully processing entries
        start += amount

        logging.info(f"Progress: {min(start, end)}/{end} papers.")
        time.sleep(request_delay)

    except urllib.error.HTTPError as e:
        logging.error(f"HTTP Error: {e.code}")
    except urllib.error.URLError as e:
        logging.error(f"URL Error: {e.reason}")
