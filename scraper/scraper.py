import requests

class Scraper():

    ROOT_URL = "https://jobs.workable.com/api/v1/jobs?query={}&location={}&remote={}"
    descriptions = []
    min_jobs = 100

    def __init__(self, job_title: str, location: str, remote: str="true") -> None:
        self.query_url = self.ROOT_URL.format(job_title, location, remote)

    def scrape(self):
        url = self.query_url
        while len(self.descriptions) < self.min_jobs:
            response = requests.get(url).json()
            for job in response["jobs"]:
                self.descriptions.append(job["description"])
            try:
                next_page_token = response["nextPageToken"]
            except KeyError:
                break
            url = self.query_url + f"&page={next_page_token}"

args = {
    "job_title": "Data Scientist",
    "location": "United States",
}
sc = Scraper(**args)
sc.scrape()
print(len(sc.descriptions))
print(sc.descriptions[0])