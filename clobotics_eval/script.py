# clobo_dd_client.py
import requests, hmac, base64, hashlib, time, uuid, json
from urllib.parse import urlparse, unquote
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env if present
OPENAPI_APP_ID="ed581d52-79c0-11f0-85e8-00155d6f14e0"
OPENAPI_APP_SECRET=""
OPENAPI_HOST = "https://adapterapi-test.clobotics.com"
# OPENAPI_APP_ID = os.environ["OPENAPI_APP_ID"]
# OPENAPI_APP_SECRET = os.environ["OPENAPI_APP_SECRET"]
EXCLUDE_CONTENT_MD5_URLS = {"/op/upload"}  # keep as-is; md5 not required for these

def _signature(string_to_sign: str) -> str:
    mac = hmac.new(OPENAPI_APP_SECRET.encode("utf-8"),
                   string_to_sign.encode("utf-8"),
                   digestmod=hashlib.sha256)
    return "cbs:" + OPENAPI_APP_ID + ":" + base64.b64encode(mac.digest()).decode("utf-8")

def _canon_resource(url: str) -> str:
    u = urlparse(url)
    path = u.path
    if not u.query:
        return path
    parts = []
    for kv in sorted(u.query.split("&")):
        k, v = kv.split("=", 1)
        parts.append(k + "=" + unquote(v))
    return path + "?" + "&".join(parts)

def send_request(method: str, path: str, body=None, data=None, files=None):
    url = OPENAPI_HOST + path
    ts = str(int(time.time()))
    nonce = uuid.uuid4().hex
    canon = _canon_resource(url)

    headers = {
        "Timestamp": ts,
        "Nonce": nonce,
    }

    # POST/PUT with JSON body that REQUIRES Content-MD5 in the signature
    if method in ("POST", "PUT") and path not in EXCLUDE_CONTENT_MD5_URLS and files is None and data is None:
        # IMPORTANT: serialize ONCE; MD5 these bytes; send EXACTLY these bytes
        body_str = json.dumps(body if body is not None else {}, separators=(', ', ': '), ensure_ascii=False)
        body_bytes = body_str.encode("utf-8")
        content_md5 = base64.b64encode(hashlib.md5(body_bytes).digest()).decode("utf-8")

        string_to_sign = f"{method}\n{content_md5}\n{ts}\n{nonce}\n{canon}"
        headers["Authorization"] = _signature(string_to_sign)
        headers["Content-Type"] = "application/json"

        return requests.request(method, url=url, headers=headers, data=body_bytes)

    # All other cases (GET/DELETE, or POST/PUT to excluded paths, or multipart/form/form-data)
    else:
        string_to_sign = f"{method}\n{ts}\n{nonce}\n{canon}"
        headers["Authorization"] = _signature(string_to_sign)

        # If it's POST/PUT and caller passed files/data, just forward as-is (typical for /op/upload)
        if method in ("POST", "PUT"):
            if files is not None or data is not None:
                return requests.request(method, url=url, headers=headers, data=data, files=files)
            # If caller provided a JSON body but this path is excluded from MD5, still send JSON bytes
            if body is not None:
                body_str = json.dumps(body, separators=(', ', ': '), ensure_ascii=False)
                headers["Content-Type"] = "application/json"
                return requests.request(method, url=url, headers=headers, data=body_str.encode("utf-8"))
            # Otherwise empty POST/PUT
            return requests.request(method, url=url, headers=headers)

        # GET/DELETE/etc.
        return requests.request(method, url=url, headers=headers)


if __name__ == "__main__":
    # 1) Create an async task with your HEIF image URL
    create_path = "/doordash/photo_inference"
    # HEIF doesn't work
    # image_url = "https://doordash-ground-truth-ingestion.cdn4dd.com/jets_shelf_scan/1800640/dx_task-352a88d8-a94e-43ba-ac38-722c7252c60d-4-2025-03-17T16:44:53Z-full_shelf.heif"
    image_url = "https://doordash-ground-truth-ingestion.cdn4dd.com/jets_shelf_scan/1111728/ops_tool--18-2025-05-21T18:13:28Z-full_shelf.heif"

    # Stable UUID derived from the image URL
    request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, image_url))

    body = {
        "request_id": request_id,
        "image_url": image_url,
    }
    resp = send_request("POST", create_path, body=body)
    print("CREATE:", resp.status_code, resp.text)
    resp.raise_for_status()
    task_id = resp.json().get("data", {}).get("task_id") or resp.json().get("task_id")
    assert task_id, "No task_id in response"

    # 2) Poll result
    import time as _t
    result_path = f"/doordash/result/{task_id}"
    for _ in range(60):
        r = send_request("GET", result_path)
        if r.status_code != 200:
            print("RESULT:", r.status_code, r.text)
            _t.sleep(5)
            continue
        j = r.json()
        status = (j.get("data") or {}).get("status") or j.get("status")
        print("STATUS:", status)
        if status in ("SUCCESS", "FAIL"):
            with open("response.json", "w") as f:
                f.write(r.text)
            print("Saved response.json")
            break
        _t.sleep(5)
