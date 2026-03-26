import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating...")
        await page.goto("https://civitai.green/models/411088/detailed-perfection-style-hands-feet-face-body-all-in-one-xl-f1d-sd15-pony-illu-zit-zib?modelVersionId=458257", timeout=60000)
        print(f"Title: {await page.title()}")
        
        # Scroll down
        print("Scrolling to load gallery...")
        for i in range(15):
            await page.keyboard.press("PageDown")
            await page.mouse.wheel(0, 2000)
            await asyncio.sleep(3)
            
            # Get links
            links = await page.eval_on_selector_all(
                "a[href^='/images/']", 
                "nodes => nodes.map(n => n.href)"
            )
            print(f"Scroll {i+1}: Found {len(links)} links.")
            if len(links) > 3:
                print("Gallery items found!")
                break
        
        if len(links) <= 3:
            print("Failed to load gallery items.")
        else:
            print(f"Total links found: {len(links)}")
            print(f"Sample items: {links[3:6]}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
