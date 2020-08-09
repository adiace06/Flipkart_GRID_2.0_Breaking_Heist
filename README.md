# **Flipkart_GRID_2.0_Breaking_Heist**
Team Breaking_Heist's solution to Flipkart Fashion Trend Prediction challenge for GRID 2.0

# Concept Video has been shared for easy explaination of how the code works.
Please View this jen-vgjz-anj (2020-08-09 at 10_17 GMT-7) (2).zip

An attempt to bring in Artificial Intelligence for assistance in fashion industry. Our project aims at traversing multiple websites to predict the current as well as upcoming trend envisaged by fashion enthusiasts.

## WEB SCRAPER 
#### scraper.py
scraper.py contains code to scrape [Amazon Indian Website](https://www.amazon.in/), flipkart, zara and other ecommerce websites. 
The user will be prompted to fill in the the type of product he/she shall desire to scrape details of.
The user will them be asked to give a certain number of pages he/she wants to scrape.

The scraper shall return a csv file with filename "Query.csv" in the folder where the code has been executed from. This csv file contains all the details of the product.
In the same folder Images will start getting scraped and all the images of the products will be saved according to a particular serial number which will be mentioned in the csv created. Therefore a structured dataset is created.

#### scrapper/scrapper
scrapper/scrapper contains spiders built on [Scrapy](https://scrapy.org/).
This scraper extracts information from [This Website](https://fashionvignette.blogspot.com/)
To execute this scraper 
- One has to install [Scrapy](https://docs.scrapy.org/en/latest/intro/install.html). 
- One needs to change FILES_STORE="The complete directory of folder where one wants to install the images" in settings.py
- Change the working directory to scrapper
- Type the command "scrapy crawl pyimagesearch-blog-spider -o output.json"
```
scrapy crawl pyimagesearch-blog-spider -o output.json
```
output can be changed to whatever filename user wants. This JSON file contains various decriptions about images scrapped.
