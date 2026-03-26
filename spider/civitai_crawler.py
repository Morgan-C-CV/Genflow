import asyncio
import json
import os
import requests
import urllib.parse
from playwright.async_api import async_playwright

def get_image_ids_from_api(model_id, model_version_id, limit=30):
    input_data = {
        "json": {
            "period": "AllTime",
            "periodMode": "published",
            "sort": "Newest",
            "withMeta": False,
            "modelVersionId": model_version_id,
            "modelId": model_id,
            "hidden": False,
            "limit": 50,
            "browsingLevel": 1,
            "cursor": None
        },
        "meta": {
            "values": {
                "cursor": ["undefined"]
            }
        }
    }
    
    encoded_input = urllib.parse.quote(json.dumps(input_data))
    url = f"https://civitai.green/api/trpc/image.getImagesAsPostsInfinite?input={encoded_input}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://civitai.green/models/{model_id}",
    }
    
    print(f"Fetching gallery from API...")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"API request failed with status {response.status_code}")
        return []
        
    data = response.json()
    items = data.get("result", {}).get("data", {}).get("json", {}).get("items", [])
    
    image_ids = []
    for item in items:
        images = item.get("images", [])
        for img in images:
            image_ids.append(img.get("id"))
            if len(image_ids) >= limit:
                return image_ids
                
    return image_ids

async def get_metadata(page, image_id):
    url = f"https://civitai.green/images/{image_id}"
    print(f"Loading {url}...")
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(8)
    
    metadata = {"id": image_id, "url": url}
    
    try:
        # Inject clipboard mock
        await page.evaluate("""
            window.lastCopiedText = null;
            navigator.clipboard.writeText = (text) => {
                window.lastCopiedText = text;
                return Promise.resolve();
            };
        """)
        
        # Try to click "COPY ALL" button (most reliable for user's requested format)
        copy_all = page.get_by_text("COPY ALL").first
        if not await copy_all.is_visible():
            copy_all = await page.query_selector("button:has-text('COPY ALL')")
            
        if copy_all:
             is_vis = await copy_all.is_visible() if hasattr(copy_all, "is_visible") else True
             if is_vis:
                 await copy_all.click()
                 await asyncio.sleep(1)
                 copied_text = await page.evaluate("window.lastCopiedText")
                 if copied_text:
                     metadata["full_metadata_string"] = copied_text
                     # Also try to extract prompt/neg prompt for convenience in JSON
                     lines = copied_text.split("\n")
                     if len(lines) > 0:
                         metadata["prompt"] = lines[0]
                     if len(lines) > 1 and "Negative prompt:" in lines[1]:
                         metadata["negative_prompt"] = lines[1].replace("Negative prompt:", "").strip()

        # Fallback/Additional individual fields
        if "prompt" not in metadata:
            prompt_xpath = "//p[.//text()='Prompt']/following-sibling::div"
            prompt_el = await page.query_selector("xpath=" + prompt_xpath)
            if prompt_el:
                metadata["prompt"] = (await prompt_el.inner_text()).replace("Show more", "").strip()

        # Extract individual badges for specific fields
        badges = await page.query_selector_all(".mantine-Badge-root")
        for badge in badges:
            text = await badge.inner_text()
            if ":" in text:
                parts = text.split(":", 1)
                key = parts[0].strip().lower().replace(" ", "_")
                val = parts[1].strip()
                metadata[key] = val

        # Main Image URL
        image_el = await page.query_selector("img[class*='EdgeImage_image']")
        if not image_el:
            image_el = await page.query_selector("div.relative.flex.size-full.items-center.justify-center > img")
            
        if image_el:
            metadata["image_url"] = await image_el.get_attribute("src")
            
    except Exception as e:
        print(f"Error extracting metadata for {image_id}: {e}")
        
    return metadata

async def main():
    model_id = 411088
    model_version_id = 458257
    output_dir = "civitai_gallery"
    os.makedirs(output_dir, exist_ok=True)
    
    image_ids = get_image_ids_from_api(model_id, model_version_id, limit=30)
    print(f"Found {len(image_ids)} image IDs.")
    
    if not image_ids:
        print("No images found. Exiting.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = await context.new_page()
        
        results = []
        for i, img_id in enumerate(image_ids):
            print(f"[{i+1}/{len(image_ids)}] Processing image {img_id}...")
            meta = await get_metadata(page, img_id)
            results.append(meta)
            
            # Download image
            if "image_url" in meta:
                try:
                    img_url = meta["image_url"]
                    # Usually thumbnails are served, check if we can get original
                    # If URL contains width=, we might want to remove it for high res
                    img_data = requests.get(img_url).content
                    img_name = f"image_{img_id}.jpg"
                    with open(os.path.join(output_dir, img_name), "wb") as f:
                        f.write(img_data)
                    meta["local_path"] = img_name
                except Exception as e:
                    print(f"Failed to download image {img_id}: {e}")
            
            # Save progress
            with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        await browser.close()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
