from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time

# Function initiating a selenium driver
def WebScraper(link: str | None = None, sleep: int | None = None, headless: bool = True) -> webdriver.Chrome:
    """
    Function initiating a selenium driver. Optionally, connects to a website
    """
    # Set up the driver
    options = webdriver.ChromeOptions() # Define the options
    options.add_experimental_option("excludeSwitches", ['enable-logging'])
    options.add_argument("--log-level=3")
    if headless:
        options.add_argument('--headless') # Prevent Chrome from popping up
    driver = webdriver.Chrome(options = options) # Initialize the web driver

    # Connect to the website
    if link:
        driver.get(link)
        if sleep:
            time.sleep(sleep) # Wait for the page to load

    return driver

# Function wrapper for finding elements using selenium accounting for compound class names
def find_elements(driver: webdriver.Chrome, by: str,  val: str, multi: bool = True, wait: int = 10):
    """
    Function wrapper for finding element/elements which waits for the elements to appear for a specified time period
    """
    # Specify the appropriate functions
    find_func = EC.presence_of_all_elements_located if multi else EC.presence_of_element_located
    match by:
        case 'class':
            by = By.CLASS_NAME
        case 'tag':
            by = By.TAG_NAME
        case _:
            raise Exception('Invalid by value specified')
    
    # Find the element/elements
    try:
        res = WebDriverWait(driver, wait).until(find_func((by, val)))
    except TimeoutException: # If the element was not found in the given time, return None
        res = None
    
    return res

# Function to get text from a Selenium element
def get_text(element, strip: bool = True):
    """
    Function extracting the text from a Selenium element
    """

    # Return the longest (most complete) text returned by the methods
    res = max([element.text, element.get_attribute("innerText"), element.get_attribute("textContent")], key=len)
    
    # Remove trailing spaces
    if strip:
        res = res.strip()
    
    return res