# job-keywords-extractor (jke)
This project ranks the top keywords or expressions listed on job requirements according to a job title.

## Requirements
* Install dependencies in [requirements.txt](requirements.txt) in a virtual environment.

## Quickstart
* Run from root:
    ```bash
    python -m jke -t "Job Title"
    ```
* For options and descriptions run: 
    ```bash
    python -m jke --help
    ```

## Algorithm
* Scrape job requirements from [jobs.workable.com](https://jobs.workable.com/) using the class [Scraper](jke/scraper.py);
* Run the TF-IDF algorithm using the class [TFIDF](jke/tfidf.py);
* Rank the top `n` keywords or expressions.
