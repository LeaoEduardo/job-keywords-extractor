import requests

class Scraper():

    ROOT_URL = "https://jobs.workable.com/api/v1/jobs?query={}&location={}&remote={}"

    def __init__(self, job_title: str, location: str, remote: str="true") -> None:
        self.query_url = self.ROOT_URL.format(job_title, location, remote)

    def scrape(self, min_jobs: int=100) -> list:
        url = self.query_url
        requirements = []
        while len(requirements) < min_jobs:
            response = requests.get(url).json()
            for job in response["jobs"]:
                requirements.append(job["requirementsSection"])
            try:
                next_page_token = response["nextPageToken"]
            except KeyError:
                break
            url = self.query_url + f"&page={next_page_token}"
        return requirements