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

### Web scrape

I first created an empty list to store scraped product details.

![Image](https://github.com/user-attachments/assets/0aa3ccf3-3ea5-405b-ad6f-47c0223ca99a)

I then looped through pages 1 to 50, displaying the current page being scraped. Since the Computing category had 50 pages and the Phones & Tablets category had another 50, I had to repeat this process for both categories.

![Image](https://github.com/user-attachments/assets/7421871b-6c3e-4d50-a45a-1abdec98b7f4)

Next I formated the base URL with the current page number, then sent an HTTP request to fetch the webpage’s content while using headers to mimic a real browser.

![Image](https://github.com/user-attachments/assets/41a5aeb0-9e3b-4c4e-8af5-0f083357c3b6)

To handle failed request (if Jumia blocks the request), it prints a warning and skips that page.

![Image](https://github.com/user-attachments/assets/968c4015-5a16-4d21-8e2e-9042b5ed1f03)

Next, I used BeautifulSoup to take the raw webpage code and organizes it in a way that makes it easier to find and extract specific information, like product names and prices. 

![Image](https://github.com/user-attachments/assets/a368bca6-58ac-4903-a407-5faa9c6bd054)

I scanned the entire webpage and collected all product listings by looking for <article> tags that have the class "prd _fb col c-prd".
Each product on Jumia's website is wrapped inside an <article> tag with this specific class. By using soup.find_all(), I gathered all these product sections into a list called items.

![Image](https://github.com/user-attachments/assets/8f651173-3fcc-4c63-bc68-1f7c89f46654)

Next, I had to go through each product on the webpage and extracts important details like:
Product name (title), Current price, Old price (if available, for discounts), Discount percentage, Customer rating (stars) and Number of reviews.
To avoid errors, I added a default value like "No Title" or "0 reviews".

![Image](https://github.com/user-attachments/assets/d16e1b6b-29c0-4eb7-9f7e-b9a0741042e0)

Then I code to check if the product(cards) has any special badges (like "Best Seller" or "Free Shipping") and stores them as a single text string.

![Image](https://github.com/user-attachments/assets/9d5b0c3e-ee0b-4b15-a495-bef764d78893)

Lastly, this line saves all the extracted product details into our products list.

![Image](https://github.com/user-attachments/assets/9ed84b42-c6bf-452f-a34d-1057d50e73c6)

If we scrape too fast, Jumia might block our access. Hence, I added this code for 2 seconds delay before moving to the next page.

![Image](https://github.com/user-attachments/assets/0659d0a9-9c3e-4049-aa1c-fc8e102ec552)
