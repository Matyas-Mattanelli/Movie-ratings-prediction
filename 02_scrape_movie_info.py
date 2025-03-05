import pickle
import auxiliary as aux
from selenium import webdriver
import os

# Load movie links
with open('movie_links.pkl', 'rb') as handle:
    movie_links = pickle.load(handle)

# Set up the driver
driver = aux.WebScraper()

# Define a function scraping info for the given link
def scrape_movie_info(link: str, driver: webdriver.Chrome):
    """
    Function scraping information about a movie based on a link
    """ 
    # Connect to the website
    driver.get(link)

    # Find the title
    title = aux.find_elements(driver, by='tag', val='h1', multi=False)
    title = aux.get_text(title)

    # Find the genres
    genres = aux.find_elements(driver, by='class', val='genres', multi=False)
    genres = aux.get_text(genres)

    # Find year, origin, and duration
    origin = aux.find_elements(driver, by='class', val='origin', multi=False)
    try:
        country, year, duration = aux.get_text(origin).split(', ', 2)
    except ValueError:
        origin_split = aux.get_text(origin).split(', ')
        if len(origin_split) == 2:
            year, duration = origin_split
            if not year.strip().isnumeric():
                raise Exception(f'Year is not numeric: {year}')
            if 'min' not in duration:
                raise Exception(f'Wrong duration: {duration}')
            country = ''
        else:
            raise Exception(f'Unexpected number of splits for origin: {origin}')

    # Find the director
    director = aux.find_elements(driver, by='class', val='creators', multi=False)
    if director is not None:
        director = aux.find_elements(director, by='tag', val='a', multi=False)
        if director is not None:
            director = aux.get_text(director)

    # Find the rating
    rating = aux.find_elements(driver, by='class', val='film-rating-average', multi=False)
    rating = aux.get_text(rating)

    # Find number of ratings and number of fans
    counters = aux.find_elements(driver, by='class', val='counter')
    no_of_ratings, no_of_fans = aux.get_text(counters[0]), aux.get_text(counters[1])

    # Find the preview
    preview = aux.find_elements(driver, by='class', val='plot-preview', multi=False, wait=3)
    if preview is None:
        preview = aux.find_elements(driver, by='class', val='plot-full', multi=False, wait=3)
        if preview is None:
            preview = aux.find_elements(driver, by='class', val='plot-full.hidden', multi=False, wait=3)
    if preview is not None:
        preview = aux.get_text(preview)
    else:
        preview = ''

    # Return a tuple of the results
    return title, genres, country.strip(), year.strip(), duration.strip(), director, rating, no_of_ratings, no_of_fans, preview

# Load file with the results scraped so far
if not os.path.isfile('raw_data.pkl'): # If the file does not exist, initialize it along with the results file
    raw_data = {i:None for i in range(len(movie_links))}
else: # If it exists, load it
    with open('raw_data.pkl', 'rb') as handle:
        raw_data = pickle.load(handle)

# Loop through the movie links and scrape them
skipped = 0 # Initialize a skip counter
for idx, movie_link in enumerate(movie_links):
    
    # Check if the data has not been scraped already
    if raw_data[idx] is None: 
        try:
            res = scrape_movie_info(movie_link, driver)
            raw_data[idx] = res
        except KeyboardInterrupt:
            print('Process interupted. Storing the data obtained so far.')
            with open('raw_data.pkl', 'wb') as out:
                pickle.dump(raw_data, out)
            break
        except:
            print(f'{movie_link} skipped')
            skipped += 1

    # Print progress
    print(f'Movie {idx + 1}/{len(movie_links)} done. Total skipped: {skipped}', end='\r')

    # Store the data every 50 iterations so as not to lose progress
    # if (idx + 1) % 50 == 0:
    #     with open('raw_data.pkl', 'wb') as out:
    #         pickle.dump(raw_data, out)

else:
    # Store the final data if the looped finished
    with open('raw_data.pkl', 'wb') as out:
        pickle.dump(raw_data, out)