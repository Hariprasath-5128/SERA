import re

STOP_SECTIONS = [
    "Additional Information & Resources",
    "References",
    "Scientific Articles on PubMed",
    "Disclaimers",
    "Return to top",
]

REMOVE_PATTERNS = [
    r"Skip navigation",
    r"A\s*\.gov\s*website belongs to an official government.*?the \.gov website\.",
    r"Share sensitive information only on official,\s*secure websites\.",
    r"You Are Here:\s*Home",
    r"Search MedlinePlus",
    r"GO\b",
    r"About MedlinePlus.*",
    r"National Library of Medicine.*",
]

def clean_text(text: str) -> str:
    """Removes website junk and stops at references."""
    
    # 1. Stop when reaching reference/bottom sections
    for section in STOP_SECTIONS:
        idx = text.find(section)
        if idx != -1:
            text = text[:idx]

    # 2. Normalize tabs to spaces
    text = text.replace('\t', ' ')
    
    # 3. Remove header junk
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # 4. Normalize whitespace
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE) # strip trailing spaces per line
    
    # 5. Fix dangling punctuation (e.g. "heart muscle\n," -> "heart muscle,")
    text = re.sub(r"\n\s*([,\.;:!?])", r"\1", text)
    
    text = re.sub(r"\n{3,}", "\n\n", text) # collapse 3+ newlines into 2

    return text.strip()
