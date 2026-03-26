import asyncio
import json
import os
import requests
import re
import sys
from playwright.async_api import async_playwright

async def get_metadata(page, image_id):
    url = f"https://civitai.green/images/{image_id}"
    print(f"Loading {url}...")
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(8)
        
        metadata = {"id": image_id, "url": url}
        
        # Inject clipboard mock
        await page.evaluate("""
            window.lastCopiedText = null;
            navigator.clipboard.writeText = (text) => {
                window.lastCopiedText = text;
                return Promise.resolve();
            };
        """)
        
        # Try to click "COPY ALL" button
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
                     lines = copied_text.split("\n")
                     if len(lines) > 0:
                         metadata["prompt"] = lines[0]
                     if len(lines) > 1 and "Negative prompt:" in lines[1]:
                         metadata["negative_prompt"] = lines[1].replace("Negative prompt:", "").strip()

        # Extract individual badges
        badges = await page.query_selector_all(".mantine-Badge-root")
        for badge in badges:
            text = await badge.inner_text()
            if ":" in text:
                parts = text.split(":", 1)
                key = parts[0].strip().lower().replace(" ", "_")
                val = parts[1].strip()
                metadata[key] = val

        # Extract "Resources used"
        try:
            resources = {"loras": []}
            # Wait a bit for dynamic content
            await asyncio.sleep(2)
            
            # Find the header using locator for better reliability 
            res_header_loc = page.get_by_text("Resources used")
            if await res_header_loc.count() > 0:
                res_header = res_header_loc.first
                await res_header.scroll_into_view_if_needed()
                
                # Check for "Show X more" and click
                show_more_loc = page.get_by_text(re.compile(r"Show \d+ more"))
                if await show_more_loc.count() > 0:
                    await show_more_loc.first.click()
                    await asyncio.sleep(1)

                # Get the list items
                items = await page.query_selector_all("xpath=//p[contains(text(), 'Resources used')]/following-sibling::ul/li")
                if not items:
                     parent = await page.query_selector("xpath=//p[contains(text(), 'Resources used')]/..")
                     if parent:
                         items = await parent.query_selector_all("ul li")
                
                if not items:
                    items = await page.query_selector_all("xpath=//li[.//a[contains(@href, '/models/')]]")
                
                for i, item in enumerate(items):
                    li_text = (await item.text_content() or "").strip()
                    li_text_lower = li_text.lower()
                    
                    res_type = ""
                    if "checkpoint" in li_text_lower:
                        res_type = "Checkpoint"
                    elif "lora" in li_text_lower:
                        res_type = "LoRA"
                    
                    if res_type:
                        links = await item.query_selector_all("a")
                        name = "Unknown"
                        version = ""
                        if len(links) >= 1:
                            name_p = await links[0].query_selector("p")
                            if name_p: name = await name_p.inner_text()
                        if len(links) >= 2:
                            version_p = await links[1].query_selector("p")
                            if version_p: version = await version_p.inner_text()
                        
                        weight = ""
                        if res_type == "LoRA":
                            match = re.search(r"(?:lora|checkpoint)\s*(\d+\.?\d*)", li_text_lower)
                            if match:
                                weight = match.group(1)
                            else:
                                match = re.search(r"(?<![\d\.])(\d\.\d+|\d)(?![\d\.])", li_text)
                                if match: weight = match.group(1)
                        
                        if res_type == "Checkpoint":
                            resources["base_model"] = {"name": name, "version": version}
                        elif res_type == "LoRA":
                            if not any(l["name"] == name for l in resources["loras"]):
                                resources["loras"].append({"name": name, "version": version, "weight": weight})
            
            if resources.get("base_model") or resources["loras"]:
                metadata["resources"] = resources
        except Exception as re_e:
            print(f"Error extracting resources for {image_id}: {re_e}")
            
            if resources.get("base_model") or resources["loras"]:
                metadata["resources"] = resources
        except Exception as re_e:
            print(f"Error extracting resources for {image_id}: {re_e}")

        # Main Image URL
        image_el = await page.query_selector("img[class*='EdgeImage_image']")
        if not image_el:
            image_el = await page.query_selector("div.relative.flex.size-full.items-center.justify-center > img")
            
        if image_el:
            metadata["image_url"] = await image_el.get_attribute("src")
            
        return metadata
    except Exception as e:
        print(f"Error for image {image_id}: {e}")
        return {"id": image_id, "url": url, "error": str(e)}

async def main(list_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata_file = os.path.join(output_dir, "metadata.json")
    
    # Load existing results if any (for resumption)
    results = []
    processed_ids = set()
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_ids = {str(item["id"]) for item in results if "error" not in item}
                print(f"Resuming from {len(processed_ids)} already processed images.")
        except Exception as e:
            print(f"Error loading existing metadata: {e}")

    # Read the list of URLs
    with open(list_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
        
    image_ids = []
    for url in urls:
        # Extract ID from https://civitai.green/images/125428448
        match = re.search(r"/images/(\d+)", url)
        if match:
            image_ids.append(match.group(1))
            
    print(f"Total image IDs in list: {len(image_ids)}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = await context.new_page()
        
        for i, img_id in enumerate(image_ids):
            if img_id in processed_ids:
                continue
                
            print(f"[{i+1}/{len(image_ids)}] Processing {img_id}...")
            meta = await get_metadata(page, img_id)
            results.append(meta)
            
            # Download image
            if "image_url" in meta:
                try:
                    img_url = meta["image_url"]
                    img_data = requests.get(img_url).content
                    img_name = f"image_{img_id}.jpg"
                    with open(os.path.join(output_dir, img_name), "wb") as f:
                        f.write(img_data)
                    meta["local_path"] = img_name
                except Exception as e:
                    print(f"Failed to download image {img_id}: {e}")
            
            # Save progress every image
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        await browser.close()
    print("All tasks completed!")

if __name__ == "__main__":
    import sys
    LIST_PATH = "/Users/mgccvmacair/Myproject/Academic/Genflow/spider/gallerylist_v1.txt"
    OUTPUT_DIR = "/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery"
    
    if len(sys.argv) > 1:
        LIST_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
        
    asyncio.run(main(LIST_PATH, OUTPUT_DIR))
