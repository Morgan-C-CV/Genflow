from lxml import html

with open("resources_page.html", "r") as f:
    content = f.read()

tree = html.fromstring(content)

# Test Header
header = tree.xpath("//p[contains(text(), 'Resources used')]")
print(f"Header found: {len(header) > 0}")

# Test List items
items = tree.xpath("//p[contains(text(), 'Resources used')]/following-sibling::ul/li")
if not items:
    items = tree.xpath("//p[contains(text(), 'Resources used')]/..//ul/li")
print(f"Items found: {len(items)}")

for i, item in enumerate(items):
    name = item.xpath(".//a//p/text()")
    print(f"Item {i} name: {name}")
    
    # Version
    version = item.xpath(".//a[2]//p/text()")
    print(f"Item {i} version: {version}")
    
    # Badges
    badges = item.xpath(".//div[contains(@class, 'mantine-Badge-root')]//span/text()")
    print(f"Item {i} badges: {badges}")
