import click

from jke.scraper import Scraper
from jke.tfidf import TFIDF

@click.command
@click.option("--job-title", "-t", type=str, required=True)
@click.option("--location", "-l", type=str, default="United States", help="Defaults to United States. Have in mind that non english job descriptions may affect the result")
@click.option("--remote", "-r", type=str, default="true", help="Remote only. 'true' or 'false'")
@click.option("--min-jobs", "-j", type=int, default=100, help="Minimum amount of jobs to run the algorithm")
@click.option("--n-rank", type=int, default=25, help="Amount of words or expressions returned")
@click.option("--ngram", type=tuple, default=(2,3), help="Minimum and maximum of words in returned expressions")
def main(job_title, location, remote, min_jobs, n_rank, ngram):
    sc = Scraper(job_title, location, remote)
    
    html_descriptions = sc.scrape(min_jobs=min_jobs)  # List of HTML descriptions

    preprocessor = TFIDF(ngram=ngram)
    preprocessed_data = preprocessor.preprocess_descriptions(html_descriptions)
    tfidf_matrix = preprocessor.fit_transform_tfidf(preprocessed_data)
    top_words = preprocessor.rank_top_words(tfidf_matrix, n=n_rank)

    print(f"Top {n_rank} highest rating words with their respective weights:")
    for word, weight in top_words:
        print(f"{word}: {weight:.4f}")

# Example usage of the class
if __name__ == '__main__':
    main()