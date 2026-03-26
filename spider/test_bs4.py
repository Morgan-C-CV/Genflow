from bs4 import BeautifulSoup
import re

with open("resources_page.html", "r") as f:
    content = f.read()

soup = BeautifulSoup(content, 'html.parser')

# Test Header
header = soup.find('p', string=re.compile('Resources used'))
print(f"Header found: {header is not None}")

if header:
    # Test List items
    # Try following sibling ul
    ul = header.find_next_sibling('ul')
    if not ul:
        # Try parent then find ul
        ul = header.parent.find('ul')
        
    if ul:
        items = ul.find_all('li', recursive=False)
        print(f"Items found: {len(items)}")

        for i, item in enumerate(items):
            # Model name
            name_el = item.find('a').find('p') if item.find('a') else None
            name = name_el.get_text() if name_el else "Unknown"
            print(f"Item {i} name: {name}")
            
            # Version
            # Second a
            a_tags = item.find_all('a')
            version = a_tags[1].find('p').get_text() if len(a_tags) > 1 and a_tags[1].find('p') else ""
            print(f"Item {i} version: {version}")
            
            # Badges
            badges = item.find_all(class_='mantine-Badge-root')
            badge_texts = [b.get_text() for b in badges]
            print(f"Item {i} badges: {badge_texts}")
