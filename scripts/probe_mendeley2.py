#!/usr/bin/env python3
import json

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nse-predictor/0.1)"}
r = requests.get(
    "https://data.mendeley.com/public-api/datasets/5hk4zw32f5",
    timeout=60,
    headers=HEADERS,
)
data = r.json()
print(json.dumps(data, indent=2)[:5000])
files = data.get("files") or data.get("versions", [{}])[0].get("files") or []
print("files type", type(files), "len", len(files) if hasattr(files, "__len__") else None)
if isinstance(files, list) and files:
    print("first file", json.dumps(files[0], indent=2)[:1000])
    fid = files[0].get("id") or files[0].get("uuid")
    for tmpl in [
        f"https://data.mendeley.com/public-files/datasets/5hk4zw32f5/files/{fid}/file_downloaded",
        f"https://data.mendeley.com/public-api/datasets/5hk4zw32f5/files/{fid}",
        f"https://data.mendeley.com/public-api/datasets/5hk4zw32f5/files/{fid}/download",
        f"https://data.mendeley.com/datasets/5hk4zw32f5/1/files/{fid}",
    ]:
        rr = requests.get(tmpl, timeout=60, headers=HEADERS, allow_redirects=True, stream=True)
        print(tmpl, "->", rr.status_code, rr.headers.get("Content-Type"), rr.headers.get("Content-Length"))
        rr.close()
