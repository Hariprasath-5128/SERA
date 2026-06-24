import re
from typing import List, Tuple

KNOWN_SECTIONS = [
    "Description",
    "Frequency",
    "Causes",
    "Inheritance",
    "Symptoms",
    "Treatment",
]

def parse_sections(text: str) -> List[Tuple[str, str]]:
    """
    Identifies predefined medical sections within the cleaned text.
    Returns a list of tuples containing (section_name, section_content).
    """
    split_points = []
    
    for sec in KNOWN_SECTIONS:
        # Match section headers appearing on their own line
        for match in re.finditer(rf'(?:^|\n\s*)({sec})(?:\s*\n|$)', text, re.IGNORECASE):
            split_points.append((match.start(), match.end(), sec))
            
    if not split_points:
        return [("General", text.strip())]
        
    # Sort by where the section appears in the text
    split_points.sort(key=lambda x: x[0])
    
    sections = []
    
    # Extract the text chunks between each header (ignoring the intro text)
    for i, (start, end, sec_name) in enumerate(split_points):
        text_start = end
        text_end = split_points[i+1][0] if i + 1 < len(split_points) else len(text)
        
        content = text[text_start:text_end].strip()
        if content:
            sections.append((sec_name, content))
            
    return sections
