import pandas as pd
import time
import requests
import re
from typing import Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from src.utils.config import load_config


def scrape_yandex_reviews(place_id: str, headless: bool = True, scroll_delay: float = 2.0, max_scroll_attempts: int = 10, chromedriver_path: Optional[str] = None) -> pd.DataFrame:
    """
    Scrape reviews from Yandex Maps using Selenium.
    Args:
        place_id: Yandex Maps place ID (e.g., peredovyye_platezhnyye_resheniya/103262022758).
        headless: Run browser in headless mode.
        scroll_delay: Seconds to wait between scrolls.
        max_scroll_attempts: Maximum scroll attempts to load reviews.
        chromedriver_path: Path to pre-installed ChromeDriver; if None, use ChromeDriverManager.
    Returns:
        DataFrame with columns: author, date, text, rating, source, sentiment.
    """
    driver = None
    try:
        opts = Options()
        if headless:
            opts.add_argument("--headless")
        opts.add_argument("--window-size=1600,1000")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        if chromedriver_path:
            service = Service(chromedriver_path)
        else:
            service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=opts)
        url = f"https://yandex.ru/maps/org/{place_id}/reviews"
        driver.get(url)
        time.sleep(3)  # Wait for initial JS load

        # Scroll to load all reviews
        attempts = 0
        prev_height = driver.execute_script("return document.body.scrollHeight")
        while attempts < max_scroll_attempts:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_delay)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == prev_height:
                break
            prev_height = new_height
            attempts += 1

        # Click spoiler buttons to reveal full text
        spoiler_buttons = driver.find_elements(By.CSS_SELECTOR, "span.spoiler-view__button")
        for btn in spoiler_buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.1)
            except:
                pass

        reviews = []
        infos = driver.find_elements(By.CSS_SELECTOR, "div.business-review-view__info")
        print(f"Yandex: Found {len(infos)} info blocks")
        for info in infos:
            try:
                # Author
                try:
                    author = info.find_element(By.CSS_SELECTOR, "div.business-review-view__author-container").text.splitlines()[0].strip()
                except:
                    author = ""

                # Date
                try:
                    date = info.find_element(By.CSS_SELECTOR, "span.business-review-view__date > span").text.strip()
                except:
                    date = ""

                # Text
                try:
                    raw = info.find_element(By.XPATH, ".//div[@itemprop='reviewBody']").text
                    text = re.sub(r"\s+", " ", raw).strip()
                except:
                    text = ""

                # Rating
                try:
                    rating_meta = info.find_element(By.CSS_SELECTOR, "meta[itemprop='ratingValue']")
                    rating = int(float(rating_meta.get_attribute("content")))
                except:
                    rating = -1

                reviews.append({
                    "author": author,
                    "date": date,
                    "text": text,
                    "rating": rating,
                    "source": "yandex",
                    "sentiment": ""  # Placeholder, updated by sentiment_generator.py
                })
            except Exception as e:
                print(f"Yandex: Error processing review: {str(e)}")
                continue

        df = pd.DataFrame(reviews)
        print(f"Yandex: Scraped {len(df)} reviews")
        return df

    except Exception as e:
        print(f"Yandex: Error scraping reviews: {str(e)}")
        return pd.DataFrame()
    finally:
        try:
            if driver is not None:
                driver.quit()
        except:
            pass


def scrape_2gis_reviews(business_id: str, api_key: str, locale: str = "ru_RU", page_size: int = 50) -> pd.DataFrame:
    """
    Fetch reviews from 2GIS API.
    Args:
        business_id: 2GIS business ID.
        api_key: 2GIS API key.
        locale: Language locale (default: ru_RU).
        page_size: Number of reviews per API call.
    Returns:
        DataFrame with columns: author, date, text, rating, source, sentiment.
    """
    try:
        url = f"https://public-api.reviews.2gis.com/2.0/branches/{business_id}/reviews"
        offset = 0
        reviews = []

        while True:
            resp = requests.get(
                url,
                params={
                    "key": api_key,
                    "locale": locale,
                    "limit": page_size,
                    "offset": offset
                },
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json().get("reviews", [])
            if not data:
                break

            for it in data:
                try:
                    author = (it.get("user") or {}).get("name", "").strip()
                    date = it.get("created_at", "")[:10] if it.get("created_at") else ""
                    text = (it.get("text") or "").strip()
                    rating = it.get("rating", -1)

                    reviews.append({
                        "author": author,
                        "date": date,
                        "text": text,
                        "rating": rating,
                        "source": "2gis",
                        "sentiment": ""  # Placeholder, updated by sentiment_generator.py
                    })
                except Exception as e:
                    print(f"2GIS: Error processing review: {str(e)}")
                    continue

            print(f"2GIS: Loaded {len(data)} reviews with offset={offset}")
            offset += page_size

        df = pd.DataFrame(reviews)
        print(f"2GIS: Fetched {len(df)} reviews")
        return df

    except Exception as e:
        print(f"2GIS: Error fetching reviews: {str(e)}")
        return pd.DataFrame()


def get_google_reviews() -> pd.DataFrame:
    """
    Return hardcoded Google reviews.
    Returns:
        DataFrame with columns: author, date, text, rating, source, sentiment.
    """
    google_reviews = [
        {
            "author": "Василь Куруц",
            "date": "2025",
            "text": "Хочу выразить благодарность компании ППР и лично менеджеру Наталье Савичевой за отличную работу и удобство предоставляемых услуг. Мы ценим вашу работу и высокий профессионализм . Отдельно хотим отметить удобство использования каршеринга  - это очень удобно и выгодно для нас!",
            "rating": 5,
            "source": "google",
            "sentiment": ""
        },
        {
            "author": "Максим Адеев",
            "date": "2019",
            "text": "Благодарен компании за очень удобные сервисы, личный кабинет, возможность выгрузки данных в наши системы и менеджеру за поддержку в любое время суток. Хорошо, что вы все время предлагаете что то новое для работы с топливными картами. Клиент ориентированность на высшем уровне!",
            "rating": 5,
            "source": "google",
            "sentiment": ""
        },
        {
            "author": "Zhanna Davoyan",
            "date": "2022",
            "text": "пользуемся в компании картами ППР года три. удобно то, что можно оплачивать все, что связано с автопарком. водителям удобно заправляться, потому почти везде принимают карты и еще шиномонтаж и мойки тоже по карте можно оплатить.",
            "rating": 5,
            "source": "google",
            "sentiment": ""
        },
        {
            "author": "Jean Anatole",
            "date": "2019",
            "text": "Однажды внезапно заблокировали все аккаунты нашей компании для входа в ЛК. Техподдержка сообщила, что заблокировал кто-то из нас, и предложил быстро разблокировать аккаунты, попросил прислать им заявление по электронной почте. Заявление отправил, через два дня нас так и не разблокировали, снова звоню в техподдержку, говорят, что заявления нет, разговор длился 20 минут, в итоге заявление все таки нашли и доступ разблокировали. Написал заявление с просьбой выяснить причину по которой вообще произошла блокировка, через день перезвонили и сказали что блокировок не было, и еще через день перезвонили и сказали, что к сожалению логи хранятся 4 дня и причины блокировки выяснить уже не удаётся. Вот так вот ) UPD: Отправил данные как попросили в комментарии, перезвонили через 5 дней, сказали что ситуация не изменилась и логи хранятся только 4 дня. Просто они ко всем отзывам оставляют свой стандартный коммент.",
            "rating": 1,
            "source": "google",
            "sentiment": ""
        },
        {
            "author": "Евгений Кононенков",
            "date": "2019",
            "text": "Лидер на рынке топливных карт! Самый лучший сервис и самые инновационные подходы к бизнесу!",
            "rating": 5,
            "source": "google",
            "sentiment": ""
        },
        {
            "author": "Павел Андреев",
            "date": "2022",
            "text": "Есть все, что нужно, но дружелюбность интерфейса оставляет желать лучшего.",
            "rating": 4,
            "source": "google",
            "sentiment": ""
        },
        {
            "author": "Роман М",
            "date": "2022",
            "text": "Отличный офис",
            "rating": 5,
            "source": "google",
            "sentiment": ""
        },
        {
            "author": "Artem Parasochka",
            "date": "2018",
            "text": "Всё круто!",
            "rating": 5,
            "source": "google",
            "sentiment": ""
        }
    ]
    df = pd.DataFrame(google_reviews)
    print(f"Google: Loaded {len(df)} hardcoded reviews")
    return df


def scrape_data() -> pd.DataFrame:
    """
    Scrape reviews from Yandex Maps, 2GIS, and include hardcoded Google reviews.
    Returns:
        Unified DataFrame with columns: text, sentiment, author, date, rating, source.
    """
    try:
        config = load_config()
        yandex_place_id = config["scraper"].get("yandex_place_id", "")
        business_id = config["scraper"].get("2gis_business_id", "")
        api_key = config["scraper"].get("2gis_api_key", "")
        chromedriver_path = config["scraper"].get("chromedriver_path", None)

        dfs = []
        if yandex_place_id:
            df_yandex = scrape_yandex_reviews(yandex_place_id, headless=True, chromedriver_path=chromedriver_path)
            if not df_yandex.empty:
                dfs.append(df_yandex)

        if business_id and api_key:
            df_2gis = scrape_2gis_reviews(business_id, api_key)
            if not df_2gis.empty:
                dfs.append(df_2gis)

        df_google = get_google_reviews()
        if not df_google.empty:
            dfs.append(df_google)

        if not dfs:
            print("No reviews scraped from any source")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        # Ensure required columns and types
        df = df[["text", "sentiment", "author", "date", "rating", "source"]]
        df["text"] = df["text"].astype(str)
        df["sentiment"] = df["sentiment"].astype(str)
        print(f"Total reviews scraped: {len(df)}")
        return df

    except Exception as e:
        print(f"Error in scrape_data: {str(e)}")
        return pd.DataFrame()