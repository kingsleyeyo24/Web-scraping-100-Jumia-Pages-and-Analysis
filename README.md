# Web Scraping 100 Jumia Pages (and Analysis)

### Project Overview

This project focuses on web scraping product data from 100 pages of Jumia, a leading e-commerce platform. The goal is to extract, clean, and analyze data on computing devices, 
phones, and accessories to gain insights into pricing, discounts, ratings, and consumer trends.

### Data Source

The data was obtained from Jumia Nigeria (**[jumia.com.ng](https://www.jumia.com.ng/)**), a popular e-commerce platform. 
The data was scraped from multiple product listings across 100 pages, covering various categories such as computing devices, phones, and accessories.

### Tools

- Python (Jupyter Notebook)

### Imports

I started by importing essential libraries for **web scraping (`requests`, `BeautifulSoup`), data processing (`pandas`, `numpy`), storage (`csv`), 
visualization (`matplotlib`, `seaborn`), and execution control (`time`, `warnings`, `IPython.display`)** to extract, analyze, and visualize data efficiently. 

![Image](https://github.com/user-attachments/assets/5d88b654-a24e-4b7c-b06e-620e65d7d504)

###  Sending Requests & Parsing HTML

To mimic a real browser, I first needed to find my browser's User-Agent by searching for it.

![Image](https://github.com/user-attachments/assets/7a1d990c-979b-4ce8-9f0a-ab7c6dcc73fc)

I sent a request to verify if I am allowed to extract data from this website and to check for any restrictions or blocks.

Status Code:
- 200 means the request was successful, and the webpage is accessible.
- 403 Forbidden means The server understood the request but refuses to authorize it

![Image](https://github.com/user-attachments/assets/0d71696e-e5f0-4513-ae62-0bd6960cc9df)
We received a 200 status code, which means we're good to go! 

