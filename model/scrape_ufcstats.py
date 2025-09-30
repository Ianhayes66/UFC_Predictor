import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "http://ufcstats.com/statistics/fighters?char={}&page=all"

def scrape_fighters():
    all_data = []
    for letter in list("abcdefghijklmnopqrstuvwxyz"):
        url = BASE_URL.format(letter)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table.b-statistics__table tbody tr")
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.select("td")]
            if not cols or len(cols) < 7:
                continue
            name, height, weight, reach, stance, wins, losses, draws = (
                cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7]
            )
            all_data.append({
                "name": name,
                "height": height,
                "weight": weight,
                "reach": reach,
                "stance": stance,
                "wins": wins,
                "losses": losses,
                "draws": draws
            })
    df = pd.DataFrame(all_data)
    df.to_csv("data/ufc_fighters.csv", index=False)
    print(f"Saved {len(df)} fighters to data/ufc_fighters.csv")

if __name__ == "__main__":
    scrape_fighters()
