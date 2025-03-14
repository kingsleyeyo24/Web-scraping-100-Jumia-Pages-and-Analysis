# this allows our code to send HTTP requests to websites
import requests

# Extracting data from web pages.
from bs4 import BeautifulSoup

# for data manipulation and analysis
import pandas as pd

# Writing to a CSV File
import csv

# Pausing execution to avoid being blocked
import time

# for better display
from IPython.display import display

# for vizualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Numerical computing
import numpy as np

# Blocks FutureWarnings from being displayed in output
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


url = "https://www.jumia.com.ng/phones-tablets/"

# Set headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

# Send the request and parse the HTML
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

print("Status Code:", response.status_code)

# Base URL for Phones & Tablets category (pagination included)
base_url = "https://www.jumia.com.ng/computing/?page={page}#catalog-listing"

# Headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# List to store all product data
products = []

# Loop through pages 1 to 50
for page in range(1, 51):
    print(f"Scraping Page {page}...")  # Status update
    
    url = base_url.format(page=page)
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch Page {page}, skipping...")
        continue

    soup = BeautifulSoup(response.content, "html.parser")

    # Find all product cards
    items = soup.find_all("article", class_="prd _fb col c-prd")

    for item in items:
        title = item.find("h3", class_="name").text.strip() if item.find("h3", class_="name") else "No Title"
        price = item.find("div", class_="prc").text.strip() if item.find("div", class_="prc") else "No Price"
        old_price = item.find("div", class_="old").text.strip() if item.find("div", class_="old") else "No discount"
        discount = item.find("div", class_="bdg _dsct").text.strip() if item.find("div", class_="bdg _dsct") else "0%"
        rating = item.find("div", class_="stars _s").text.strip() if item.find("div", class_="stars _s") else "No rating"
        reviews = item.find("div", class_="rev").text.strip() if item.find("div", class_="rev") else "0 reviews"

        # Extract all badges 
        badges = item.find_all("span", class_="bdg")
        badge_texts = ", ".join([badge.text.strip() for badge in badges]) if badges else "No badges"

        # Append product data
        products.append([title, price, old_price, discount, rating, reviews, badge_texts])

    time.sleep(2)  # Pause to avoid being blocked

# Convert to DataFrame
comp_df = pd.DataFrame(products, columns=["Title", "Price", "Old Price", "Discount", "Rating", "Reviews", "Badges"])

# Save to CSV
df.to_csv("jumia_products_computing.csv", index=False)

phones_df = pd.read_csv("jumia_products_phones&tablet.csv")
phones_df.head()

comp_df["Category"] = "Computing"
comp_df.tail()

phones_df.loc[:, "Category"] = "Phones & Tablets"
phones_df.head()

def wrangle(comp_df, phones_df):
    # Merged DataFrame
    merged_df = pd.concat([comp_df, phones_df], ignore_index=True)

    # Clean Rating column
    merged_df["Rating"] = merged_df["Rating"].astype(str).str.replace(" out of 5", "", regex=False)
    merged_df["Rating"] = pd.to_numeric(merged_df["Rating"], errors="coerce").fillna(0)

    # Extract numeric values from Reviews column and create a new column
    merged_df["Review Count"] = merged_df["Reviews"].str.extract(r"\((\d+)\)").astype(float)

    # Keep only the rating values in Reviews column
    merged_df["Reviews"] = merged_df["Reviews"].str.extract(r"([\d.]+)").astype(float)

    # Drop Reviews column (data is same as Rating)
    merged_df.drop(columns=["Reviews"], inplace=True)

    # Drop discount column
    merged_df.drop(columns=["Discount"], inplace=True)

    # Convert Ratings to float
    merged_df["Rating"] = merged_df["Rating"].astype(float)

    # Remove currency symbol and commas 
    merged_df["Price"] = merged_df["Price"].str.replace(r"[₦,]", "", regex=True)
    
    # Handle 7488 - 7500 by extracting the first number
    merged_df["Price"] = merged_df["Price"].str.extract(r"(\d+)").astype(float)

    # Remove currency symbol and commas
    merged_df["Old Price"] = merged_df["Old Price"].str.replace(r"[₦,]", "", regex=True)
    
    # Convert to numeric, setting non-numeric values to NaN, then replace NaN with 0
    merged_df["Old Price"] = pd.to_numeric(merged_df["Old Price"], errors="coerce").fillna(0)

    # Calculate for discount
    merged_df["Discount %"] = ((merged_df["Old Price"] - merged_df["Price"]) / merged_df["Old Price"]) * 100

    # Convert -inf to 0
    merged_df["Discount %"] = merged_df["Discount %"].replace([-np.inf, np.inf], 0)

    # Fill NaN with 0
    merged_df["Review Count"] = merged_df["Review Count"].fillna(0)
    
    return merged_df 

# Columns to exclude
exclude_cols = ["Title", "Price", "Old Price"]

# Loop through each column and display unique values
for col in merged_df.columns:
    if col not in exclude_cols:
        print(f"Unique values in '{col}':")
        print(merged_df[col].unique(), "\n")

merged_df.info()

# Display the top 10 most reviewed products
print(top_reviewed[["Title", "Review Count", "Rating", "Price", "Discount %"]])

# Sort products by Review Count in descending order
top_reviewed = merged_df.sort_values(by="Review Count", ascending=False).head(10)

# Visualization
plt.figure(figsize=(10, 5))
plt.barh(top_reviewed["Title"], top_reviewed["Review Count"], color="skyblue")
plt.xlabel("Review Count")
plt.ylabel("Product Title")
plt.title("Top 10 Most Reviewed Products")
plt.gca().invert_yaxis() 
plt.show();

# Display the top 10 most reviewed products
display(top_discounted[["Title", "Review Count", "Rating", "Price", "Old Price", "Discount %"]])

# Sort products by Discount in descending order
top_discounted = merged_df.sort_values(by="Discount %", ascending=False).head(10)

# Visualization
plt.figure(figsize=(10, 7))
plt.barh(top_discounted["Title"], top_discounted["Discount %"], color="skyblue")
plt.xlabel("Discount Count")
plt.ylabel("Product Title")
plt.title("Top 10 Most Discounted Products")
plt.gca().invert_yaxis()  

plt.show();

merged_df["Discount %"].describe().round(2)

merged_df["Price"].describe().round(2)

# Sort products by Price in descending order
most_expensive = merged_df.sort_values(by="Price", ascending=False).head(5)

# Visualization
plt.figure(figsize=(10, 7))
plt.barh(most_expensive["Title"], most_expensive["Price"], color="skyblue")
plt.xlabel("Price")
plt.ylabel("Product Title")
plt.title("Top 5 Most Expensive Products")
plt.gca().invert_yaxis()  
ax = plt.gca()  
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.show()

# Price vs. Rating
plt.figure(figsize=(10, 6))
plt.scatter(merged_df["Price"], merged_df["Rating"], alpha=0.5, color="blue")
plt.xlabel("Price")
plt.ylabel("Rating")
plt.title("Price vs. Rating Analysis")
plt.grid(True)
ax = plt.gca() 
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.show()

correlation = merged_df["Price"].corr(merged_df["Rating"])
print(f"Correlation between Price and Rating: {correlation}")

correlation = merged_df["Discount %"].corr(merged_df["Review Count"])
print(f"Correlation between Discount and Reviews: {correlation}")

# Discount vs. Popularity (Number of Reviews)
plt.figure(figsize=(10, 6))
plt.scatter(merged_df["Discount %"], merged_df["Review Count"], alpha=0.5, color="blue")
plt.xlabel("Discount (%)")
plt.ylabel("Number of Reviews")
plt.title("Discount vs. Popularity (Number of Reviews)")
plt.grid(True)
ax = plt.gca() 
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.show()

# Count of occurrences of each badge type
badge_counts = merged_df["Badges"].value_counts()

# Plot bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=badge_counts.index, y=badge_counts.values, hue=badge_counts.index, palette="viridis", legend=False)

plt.xlabel("Badge Type")
plt.ylabel("Count")
plt.title("Distribution of Badges")
plt.xticks(rotation=45) 

plt.show()

# Ratings Distribution by Badge Type Visualization
merged_df.groupby("Badges")["Rating"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.boxplot(x="Badges", y="Rating", data=merged_df)
plt.xticks(rotation=45)
plt.title("Ratings Distribution by Badge Type")

plt.show()

badge_dummies = pd.get_dummies(merged_df["Badges"])  
correlation = badge_dummies.corrwith(merged_df["Rating"])
print(correlation)

# Average Discount Percentage by Category Visualization
top_discount_categories = merged_df.groupby("Category")["Discount %"].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_discount_categories.index, y=top_discount_categories.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Average Discount Percentage by Category")
plt.xlabel("Category")
plt.ylabel("Average Discount (%)")

plt.show()

# Filter rows where 'Title' contains "Laptop"
laptops = merged_df[
    merged_df["Title"].str.contains(r"Laptop.*RAM|RAM.*Laptop", case=False, na=False) &
    ~merged_df["Title"].str.contains(r"Skin|Cover|Sticker|Stand", case=False, na=False)
]

display(laptops.head(10))

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full text in cells

display(laptops) 

print(laptops.shape)  

display(cheapest_laptops)
print(cheapest_laptops.shape)  

# Filter laptop items and select the 3 cheapest
cheapest_laptops = laptops.nsmallest(3, 'Price')

# Set figure size
plt.figure(figsize=(10, 6))

# Create barplot
sns.barplot(x=cheapest_laptops["Title"], y=cheapest_laptops["Price"], palette="viridis")

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha="right")

# Add title and labels
plt.title("Top 3 Cheapest Laptops", fontsize=14)
plt.xlabel("Laptop", fontsize=12)
plt.ylabel("Price (₦)", fontsize=12)

# Show plot
plt.show()

display(laptops["Rating"].value_counts())

# Get top-rated unique laptops
most_rated_laptops = laptops.nlargest(10, 'Rating').drop_duplicates(subset="Title").head(3)

# Set figure size to prevent excessive width
fig, ax = plt.subplots(figsize=(8, 5))

# Shorten long laptop names for better display
most_rated_laptops["Short_Title"] = most_rated_laptops["Title"].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)

# Barplot for Price
sns.barplot(x=most_rated_laptops["Short_Title"], y=most_rated_laptops["Price"], ax=ax, palette="viridis", alpha=0.8)

# Scatterplot for Ratings 
ax.scatter(most_rated_laptops["Short_Title"], most_rated_laptops["Rating"] * 50000, 
           color="red", label="Rating (scaled)", marker="o", s=100)

# Rotate x-axis labels for better readability
plt.xticks(rotation=30, ha="right")

# Add titles and labels
plt.title("Top 3 Rated Laptops: Price & Rating")
plt.xlabel("Laptop")
plt.ylabel("Price (₦)")

# Add legend
plt.legend()

# Adjust layout to fit labels
plt.tight_layout()

# Show plot
plt.show()

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full text in cells

display(merged_df) 

print(expensive_vs_cheap.shape)

# Select the top 5 most expensive and least expensive phones
most_expensive = phones.nlargest(3, 'Price')
least_expensive = phones.nsmallest(3, 'Price')

# Combine both into one DataFrame
expensive_vs_cheap = pd.concat([most_expensive, least_expensive])

# Set figure size
plt.figure(figsize=(12, 7)) 

sns.barplot(
    y=expensive_vs_cheap["Title"], 
    x=expensive_vs_cheap["Price"], 
    palette="coolwarm", 
    order=expensive_vs_cheap.sort_values(by="Price", ascending=False)["Title"]
)

# Add title and labels
plt.title("Most and Least Expensive Phones")
plt.xlabel("Price (₦)")
plt.ylabel("Phone")

ax = plt.gca()  
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()  # Adjust layout to prevent label cutoffs

plt.show()

# Get top-rated unique phones
most_rated_phones = phones.nlargest(3, 'Rating').drop_duplicates(subset="Title").head(3)

# Set figure size to prevent excessive width
fig, ax = plt.subplots(figsize=(8, 5))

# Shorten long phone names for better display
most_rated_phones["Short_Title"] = most_rated_phones["Title"].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)

# Barplot for Price
sns.barplot(x=most_rated_phones["Short_Title"], y=most_rated_phones["Price"], ax=ax, palette="viridis", alpha=0.8)

# Scatterplot for Ratings (Scaled for better visibility)
ax.scatter(most_rated_phones["Short_Title"], most_rated_phones["Rating"] * 50000, 
           color="red", label="Rating (scaled)", marker="o", s=100)

# Rotate x-axis labels for better readability
plt.xticks(rotation=30, ha="right")

# Add titles and labels
plt.title("Top 3 Rated Phones: Price & Rating", fontsize=12)
plt.xlabel("Phone", fontsize=10)
plt.ylabel("Price (₦)", fontsize=10)

# Add legend
plt.legend()

# Adjust layout to fit labels
plt.tight_layout()

# Show plot
plt.show()
