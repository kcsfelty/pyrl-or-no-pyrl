"""Scrape Deal or No Deal Fandom pages via MediaWiki API."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import requests

API_BASE = "https://deal.fandom.com/api.php"
USER_AGENT = "pyrl-or-no-pyrl/0.1 (fandom-scraper)"

CATEGORY_FILTER = re.compile(r"(episode|episodes|game|games)", re.IGNORECASE)
OFFER_PATTERN = re.compile(r"offer[^\d]*([\$£])\s*([\d,]+)", re.IGNORECASE)
ORDINAL_PATTERN = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|final)\b",
    re.IGNORECASE,
)


ORDINAL_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "final": "Final",
}


def api_get(params: Dict[str, str], timeout: int = 30) -> dict:
    params = dict(params)
    params.setdefault("format", "json")
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(API_BASE, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def allcategories(prefix: str) -> List[str]:
    categories = []
    cont = None
    while True:
        params = {
            "action": "query",
            "list": "allcategories",
            "acprefix": prefix,
            "aclimit": "max",
        }
        if cont:
            params["accontinue"] = cont
        data = api_get(params)
        for item in data.get("query", {}).get("allcategories", []):
            categories.append(item["*"])
        cont = data.get("continue", {}).get("accontinue")
        if not cont:
            break
    return categories


def categorymembers(category: str) -> List[str]:
    titles = []
    cont = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": "0",
            "cmlimit": "max",
        }
        if cont:
            params["cmcontinue"] = cont
        data = api_get(params)
        for item in data.get("query", {}).get("categorymembers", []):
            titles.append(item["title"])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return titles


def fetch_wikitext(title: str) -> str:
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
    }
    data = api_get(params)
    wikitext = data.get("parse", {}).get("wikitext", {}).get("*")
    if wikitext:
        return wikitext
    # Fallback to revisions API
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvprop": "content",
        "rvslots": "main",
    }
    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        revs = page.get("revisions", [])
        if revs:
            return revs[0].get("slots", {}).get("main", {}).get("*") or ""
    return ""


def parse_offers(text: str) -> List[dict]:
    offers = []
    for match in OFFER_PATTERN.finditer(text):
        currency = match.group(1)
        amount = int(match.group(2).replace(",", ""))
        window = text[max(0, match.start() - 40) : match.end() + 40]
        ordinal_match = ORDINAL_PATTERN.search(window)
        ordinal = ordinal_match.group(1).lower() if ordinal_match else None
        round_value = ORDINAL_MAP.get(ordinal) if ordinal else None
        offers.append(
            {
                "round": round_value,
                "currency": currency,
                "amount": amount,
                "context": window.strip(),
            }
        )
    return offers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="data/fandom_games_raw.json")
    parser.add_argument("--out-csv", default="data/fandom_games_raw.csv")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    prefixes = ["Deal", "Deal_or_No_Deal", "Deal or No Deal"]
    categories = []
    for prefix in prefixes:
        categories.extend(allcategories(prefix))

    categories = sorted(set(c for c in categories if CATEGORY_FILTER.search(c)))

    titles = []
    for category in categories:
        titles.extend(categorymembers(category))

    titles = sorted(set(titles))
    if args.max_pages and args.max_pages > 0:
        titles = titles[: args.max_pages]

    results = []
    for idx, title in enumerate(titles, 1):
        wikitext = fetch_wikitext(title)
        offers = parse_offers(wikitext)
        results.append(
            {
                "title": title,
                "offers": offers,
                "wikitext": wikitext,
            }
        )
        if args.delay:
            time.sleep(args.delay)

    # Write JSON
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Write CSV (flatten offers)
    with open(args.out_csv, "w") as f:
        f.write("title,round,currency,amount,context\n")
        for entry in results:
            for offer in entry["offers"]:
                ctx = offer["context"].replace("\n", " ").replace(",", " ")
                f.write(
                    f"{entry['title']},{offer['round']},{offer['currency']},{offer['amount']},{ctx}\n"
                )

    print("Categories:", len(categories))
    print("Titles:", len(titles))
    print("Pages scraped:", len(results))


if __name__ == "__main__":
    main()
