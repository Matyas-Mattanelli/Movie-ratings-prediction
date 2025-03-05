import auxiliary as aux
import pickle

# Specify the link as a function of page number
link = lambda page: f'https://www.csfd.cz/zebricky/vlastni-vyber/?page={page}&filter=rlW0rKOyVwbjYPWipzyanJ4vBz51oTjfVzqyoaWyVwcoKFjvrJIupy9zpz9gVwbkBQp4YPW5MJSlK3EiVwblZQVmYPWuL3EipvV6J10fVzEcpzIwqT9lVwcoKK0'

# Set up the driver
driver = aux.WebScraper()

# Loop through all pages and get the movie links
movie_links = []
for page in range(1, 1000):
    driver.get(link(page)) # Connect to the link of the corresponding page
    a_tags = aux.find_elements(driver, by='class', val='film-title-name') # Get the tags containing movie title and link
    if a_tags is not None:
        links = [a.get_attribute('href') for a in a_tags]
        movie_links.extend(links)
        print(f'Page {page} done.', end='\r')
    else:
        print(f'Stopped on page {page}. Number of obtained links: {len(movie_links)}')
        break

# Store the links
with open('movie_links.pkl', 'wb') as handle:
    pickle.dump(movie_links, handle)