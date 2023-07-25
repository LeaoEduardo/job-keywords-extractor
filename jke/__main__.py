import click

from jke.scraper import Scraper
from jke.tfidf import TFIDFPreprocessor

@click.command
@click.option("--job-title", "-t", type=str, required=True)
@click.option("--location", "-l", type=str, default="United States")
@click.option("--remote", "-r", type=str, default="true")
@click.option("--min-jobs", "-j", type=int, default=100)
@click.option("--n-rank", type=int, default=25)
@click.option("--ngram", type=tuple, default=(2,3))
def main(job_title, location, remote, min_jobs, n_rank, ngram):
    sc = Scraper(job_title, location, remote)
    
    html_descriptions = sc.scrape(min_jobs=min_jobs)  # List of HTML descriptions

    preprocessor = TFIDFPreprocessor(ngram=ngram)
    preprocessed_data = preprocessor.preprocess_descriptions(html_descriptions)
    tfidf_matrix = preprocessor.fit_transform_tfidf(preprocessed_data)
    top_words = preprocessor.rank_top_words(tfidf_matrix, n=n_rank)

    print(f"Top {n_rank} highest rating words with their respective weights:")
    for word, weight in top_words:
        print(f"{word}: {weight:.4f}")

# Example usage of the class
if __name__ == '__main__':
    main()