#!/usr/bin/env python3
import json
import re

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nse-predictor/0.1)"}


def probe(url: str) -> None:
    print("=" * 60)
    print(url)
    try:
        r = requests.get(url, timeout=60, headers=HEADERS, allow_redirects=True)
    except Exception as exc:  # noqa: BLE001
        print("ERR", exc)
        return
    print("status", r.status_code, "ctype", r.headers.get("Content-Type"))
    text = r.text
    print("snippet", text[:300].replace("\n", " "))
    links = re.findall(r"https?://[^\"'\\s]+(?:zip|csv|download|s3)[^\"'\\s]*", text, flags=re.I)
    print("links", links[:15])
    files = list(dict.fromkeys(re.findall(r"/files/[a-f0-9\-]+", text)))
    print("files", files[:15])
    # Next.js JSON blob?
    m = re.search(r"__NEXT_DATA__\" type=\"application/json\">(.*?)</script>", text)
    if m:
        data = json.loads(m.group(1))
        print("next_keys", list(data.keys())[:10])
        print(json.dumps(data, indent=2)[:2000])


if __name__ == "__main__":
    for u in [
        "https://data.mendeley.com/datasets/5hk4zw32f5/1",
        "https://data.mendeley.com/api/datasets/5hk4zw32f5",
        "https://data.mendeley.com/public-api/datasets/5hk4zw32f5",
        "https://data.mendeley.com/datasets/5hk4zw32f5/1/files/",
    ]:
        probe(u)
