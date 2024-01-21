import pandas as pd
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import numpy as np

# 1 - Use the requests module to download the html for URL.
start_url = "https://ca.trustpilot.com/review/kabo.co"
response = requests.get(start_url)


# 2 - Extract the total number of reviews
soup = BeautifulSoup(response.text, "html.parser")
item = soup.find(
    name="p",
    attrs={"class": "typography_body-l__KUYFJ typography_appearance-default__AAY17"},
)
N = int(item.contents[0].replace(",", ""))
# print(N)


# 3 - Iterate over the review pages.
pages = [start_url]

while True:
    response = requests.get(start_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        next_page_link = soup.find("a", string="Next page").get("href")

        if next_page_link:
            start_url = "https://ca.trustpilot.com" + next_page_link
            pages.append(start_url)
        else:
            # print('No more pages to iterate.')
            break
    else:
        print("Failed to retrieve page. Status code:", response.status_code)
        break


# 4 - From each page extract the reviews.
co_name_element = soup.find("div", id="business-unit-title")
co_name_element = co_name_element.find(
    "span", class_="title_displayName__TtDDM"
).text.strip()

company_name = []
dates = []
ratings = []
review_bodies = []

for i in range(0, len(pages)):
    response = requests.get(pages[i])
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        card_wrappers = soup.find_all(
            "div", class_=lambda x: x and "styles_cardWrapper" in x
        )

        for card_wrapper in card_wrappers:

            company_name.append(co_name_element)

            time_element = card_wrapper.find("time").get("datetime")
            dates.append(time_element)

            rating_element = card_wrapper.find(
                "div", {"data-service-review-rating": True}
            )
            if rating_element:
                rating = rating_element["data-service-review-rating"]
                ratings.append(rating)
            else:
                print("Rating not found in the HTML.")

            review_text_element = card_wrapper.find(
                attrs={"data-service-review-text-typography": "true"}
            )
            if review_text_element:
                review_body = review_text_element.text
                review_bodies.append(review_body)
            else:
                review_bodies.append("Review text not found")

    else:
        print("This page returned a status code of", response.status_code)
        break

# 5 - From each review, store the following to the CSV file:
data = {
    "companyName": company_name,
    "datePublished": dates,
    "ratingValue": ratings,
    "reviewBody": review_bodies,
}

df = pd.DataFrame(data)
# df.head()


# 6 - The final CSV file should have at least 500 rows and four columns
# limit the number of reviews to 500
df = df.head(500)

# save the DataFrame as a CSV file
df.to_csv(str(company_name[0]) + ".csv", index=False)
