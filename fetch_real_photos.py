"""
Download real photographs (negatives for the evaluation) from Wikimedia Commons "Quality images"
categories, one per CIFAR-10 class. Only files uploaded before 2020 are used, so none can come
from a modern image generator. Credits/licences are written to evaluation/images/real/CREDITS.csv.

  python fetch_real_photos.py            # 1 photo per class -> 10 photos
  python fetch_real_photos.py --per-class 3
"""
import argparse
import csv
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "evaluation" / "images" / "real"
UA = {"User-Agent": "image-detector-eval/1.0 (academic evaluation script)"}
API = "https://commons.wikimedia.org/w/api.php?"
CATEGORIES = {
    "airplane": ["Quality images of airplanes"],
    "automobile": ["Quality images of automobiles"],
    "bird": ["Quality images of birds"],
    "cat": ["Quality images of cats in Japan", "Quality images of cats in Portugal",
            "Quality images of cats in Belgium", "Quality images of cats in Ukraine"],
    "deer": ["Quality images of Cervidae"],
    "dog": ["Quality images of dogs"],
    "frog": ["Quality images of Anura"],
    "horse": ["Quality images of horses"],
    "ship": ["Quality images of ships"],
    "truck": ["Quality images of trucks"],
}


def fetch(url, tries=6):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 10 * (i + 1)
            print(f"  rate-limited, waiting {wait}s")
            time.sleep(wait)


def api(params):
    return json.loads(fetch(API + urllib.parse.urlencode({**params, "format": "json"})))


def list_titles(category, depth=1):
    """File titles in the category (+ one level of subcategories). Cheap: no image info."""
    titles, cont = [], {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                 "cmtype": "file|subcat", "cmlimit": 500, **cont})
        for m in d.get("query", {}).get("categorymembers", []):
            if m["ns"] == 14 and depth > 0:
                titles += list_titles(m["title"].removeprefix("Category:"), depth - 1)
            elif m["ns"] == 6 and m["title"].lower().endswith((".jpg", ".jpeg")):
                titles.append(m["title"])
        if "continue" not in d:
            return titles
        cont = d["continue"]
        time.sleep(0.5)


def image_info(titles):
    """Upload date, size, licence and a 1024-px thumbnail URL for up to 10 titles."""
    d = api({"action": "query", "titles": "|".join(titles), "prop": "imageinfo",
             "iiprop": "url|size|mime|timestamp|extmetadata", "iiurlwidth": 1024})
    return [{"title": p["title"], **p["imageinfo"][0]}
            for p in d.get("query", {}).get("pages", {}).values() if "imageinfo" in p]


def clean(html):
    return re.sub(r"<[^>]+>", "", html or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    credits = []
    n = 0
    for cls, cats in CATEGORIES.items():
        titles = []
        for c in cats:
            try:
                titles += list_titles(c)
            except urllib.error.HTTPError as e:
                print(f"  {c}: {e}")
        titles = sorted(set(titles))
        rng.shuffle(titles)
        chosen = []
        for i in range(0, min(len(titles), 60), 10):  # inspect candidates 10 at a time
            for f in image_info(titles[i: i + 10]):
                if (f.get("mime") == "image/jpeg" and f.get("timestamp", "9999") < "2020-01-01"
                        and f.get("width", 0) >= 1024 and f.get("height", 0) >= 600):
                    chosen.append(f)
            if len(chosen) >= args.per_class:
                break
            time.sleep(1)
        if not chosen:
            print(f"{cls}: no suitable photo found")
            continue
        for f in chosen[: args.per_class]:
            path = OUT / f"{n:02d}_{cls}.jpg"
            n += 1
            path.write_bytes(fetch(f["thumburl"]))
            md = f.get("extmetadata", {})
            credits.append({
                "file": path.name, "class": cls, "commons_title": f["title"],
                "author": clean(md.get("Artist", {}).get("value")),
                "license": md.get("LicenseShortName", {}).get("value", ""),
                "uploaded": f["timestamp"][:10], "source": f["descriptionurl"],
            })
            print(f"{path.name}: {f['title']} ({credits[-1]['license']}, uploaded {f['timestamp'][:10]})")
            time.sleep(1)
    with open(OUT / "CREDITS.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(credits[0]))
        w.writeheader()
        w.writerows(credits)
    print(f"\n{len(credits)} photos -> {OUT}")


if __name__ == "__main__":
    main()
