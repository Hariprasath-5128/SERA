import re
from bs4 import BeautifulSoup
import trafilatura
from xml.etree import ElementTree as ET

def fix_lists(soup):
    """Converts lists into paragraph tags so Trafilatura doesn't drop them."""
    for list_tag in soup.find_all(['ul', 'ol']):
        for li in list_tag.find_all('li', recursive=False):
            p = soup.new_tag('p')
            p.string = "- " + li.get_text(separator=' ', strip=True)
            list_tag.insert_before(p)
        list_tag.decompose()

def extract_structured_content(html, domain, url=None, fallback=False):
    if not html:
        return {}
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # ---------------------------------------------------------
    # DOMAIN SPECIFIC PRE-PROCESSING (HTML LEVEL)
    # ---------------------------------------------------------
    
    if 'cancer.gov' in domain:
        # Fix missing sub-bullets by converting lists to standard paragraph blocks
        fix_lists(soup)
        
    elif 'ninds.nih.gov' in domain:
        # Remove breadcrumbs
        for breadcrumb in soup.find_all(string=re.compile(r'You Are Here:')):
            if breadcrumb and getattr(breadcrumb, 'parent', None):
                breadcrumb.parent.decompose()
                
        # Fix missing body headers (Trafilatura strips them because they are duplicated in the TOC)
        for h2 in soup.find_all('h2'):
            text = h2.get_text(strip=True)
            if not text or text.lower().startswith('table of contents'):
                h2.decompose()
                continue
            
            p = soup.new_tag('p')
            strong = soup.new_tag('strong')
            strong.string = f"__NINDS_HEADER__{text}"
            p.append(strong)
            h2.replace_with(p)
            
        for div in soup.find_all('div', class_='datadisplay'):
            div.unwrap()
                
        # Remove addresses
        address_regex = re.compile(
            r'National Institutes of Health, DHHS|'
            r'55 Kenosia Avenue|'
            r'1275 Mamaroneck Avenue|'
            r'P\.?O\.? Box \d+|'
            r'Bethesda, MD|'
            r'Suite \d+|'
            r'\d{5}-\d{4}', re.IGNORECASE
        )
        for addr in soup.find_all(string=address_regex):
            if not addr or not getattr(addr, 'parent', None):
                continue
            parent = addr.parent
            if parent:
                # Remove parent and all siblings after it
                for sibling in list(parent.next_siblings):
                    if hasattr(sibling, 'decompose'):
                        sibling.decompose()
                parent.decompose()

    elif 'niddk.nih.gov' in domain:
        # Remove [Top]
        for top_link in soup.find_all('a', string=lambda text: text and 'Top' in text.strip() and len(text.strip()) <= 5):
            top_link.decompose()
            
    # ---------------------------------------------------------
    bs4_sections = {}
    try:
        # ---------------------------------------------------------
        # DOMAIN SPECIFIC PRE-PROCESSING FOR RAREDISEASES
        # ---------------------------------------------------------
        if domain == 'rarediseases.info.nih.gov':
            main_content = soup.find('div', id='MainContent') or soup.body
            
            panels_found = soup.find_all('div', class_=lambda c: c and 'panel' in c.split())
            if panels_found:
                # Pre-clean known boilerplate
                for tag in main_content.find_all(['nav', 'aside', 'footer', 'header']):
                    tag.decompose()
                for div in main_content.find_all('div', class_=lambda c: c and any(x in c.lower() for x in ['sidebar', 'menu', 'share', 'social', 'breadcrumb', 'nav', 'print', 'disease-title', 'location'])):
                    div.decompose()
                for ol in main_content.find_all('ol', class_=lambda c: c and 'breadcrumb' in c.lower()):
                    ol.decompose()
                for div in main_content.find_all('div', id=lambda i: i and any(x in i.lower() for x in ['sidebar', 'menu', 'share', 'social'])):
                    div.decompose()
                    
                # Extract panels
                for panel in main_content.find_all('div', class_=lambda c: c and 'panel' in c.split()):
                    h2 = panel.find('h2', class_=lambda c: c and 'panel-title' in c.split())
                    if not h2: 
                        panel.decompose()
                        continue
                    
                    text = h2.get_text(strip=True)
                    if not text: 
                        panel.decompose()
                        continue
                    
                    if len(text) > 4 and len(text) % 2 == 0 and text[:len(text)//2] == text[len(text)//2:]:
                        text = text[:len(text)//2]
                        
                    content_div = panel.find('div', class_=lambda c: c and 'panel-collapse' in c.split())
                    if content_div:
                        for a in content_div.find_all('a', class_='rsbtn_play'):
                            a.decompose()
                        
                        # Strip "Listen" from the beginning
                        content = content_div.get_text(separator='\n', strip=True)
                        content = re.sub(r'^Listen\s*', '', content.strip())
                        if content:
                            bs4_sections[f"[{text.upper()}]"] = content
                    
                    # Decompose the panel so Trafilatura doesn't see it (it will only see the description text)
                    panel.decompose()
                    
                # Description is everything left
                for a in main_content.find_all('a', class_='rsbtn_play'):
                    a.decompose()
                desc = main_content.get_text(separator='\n', strip=True)
                desc = re.sub(r'^Listen\s*', '', desc.strip())
                if desc:
                    bs4_sections['[DESCRIPTION]'] = desc
            
            # ALWAYS extract symptoms explicitly from desktop layout
            list_items = main_content.find_all('div', class_=lambda c: c and 'list-item' in c.split())
            if list_items:
                symptom_blocks = []
                for item in list_items:
                    h4 = item.find('h4')
                    if h4:
                        term_text = h4.get_text(strip=True).upper()
                        
                        text_lines = []
                        col5 = item.find('div', class_=lambda c: c and 'col-5' in c.split())
                        if col5:
                            for p in col5.find_all('p'):
                                t = p.get_text(separator=' ', strip=True)
                                if t.lower().startswith('synonym:'):
                                    t = t.replace('Synonym:', 'Synonyms:')
                                if t.lower().startswith('synonyms:'):
                                    t = t.replace('Synonyms:', 'Synonyms: ')
                                    t = re.sub(r'(Synonyms:\s*)+', 'Synonyms: ', t)
                                if t:
                                    text_lines.append(t)
                        
                        if text_lines:
                            text_content = '\n'.join(text_lines)
                            word_count = len(text_content.split())
                            if word_count >= 5:
                                symptom_blocks.append(f"{term_text}:\n{text_content}")
                                
                    item.decompose()
                
                if symptom_blocks:
                    bs4_sections['SYMPTOMS'] = '\n\n'.join(symptom_blocks)
                    
            # Also decompose mobile symptoms view to prevent Trafilatura from parsing it
            symptoms_div = main_content.find('div', id='symptoms')
            if symptoms_div:
                symptoms_div.decompose()
                
            # Fallback to Trafilatura if no panels (symptoms are already removed)
            if not panels_found:
                # Fallback to Trafilatura if no panels
                for h2 in soup.find_all('h2'):
                    text = h2.get_text(strip=True)
                    if not text: continue
                    # Fix duplicate header text like 'SummarySummary'
                    if len(text) > 4 and len(text) % 2 == 0 and text[:len(text)//2] == text[len(text)//2:]:
                        text = text[:len(text)//2]
                    
                    p = soup.new_tag('p')
                    strong = soup.new_tag('strong')
                    strong.string = f"__RARE_HEADER__[{text.upper()}]"
                    p.append(strong)
                    h2.replace_with(p)
                    
                for div in soup.find_all('div'):
                    div.unwrap()

    except Exception as e:
        print(f"Error preparing HTML: {e}")
        return ""

    # ---------------------------------------------------------
    # TRAFILATURA EXTRACTION
    # ---------------------------------------------------------
    
    # Convert back to string for trafilatura
    clean_html = str(soup)
    
    # We use trafilatura's XML output to get sections
    xml_output = trafilatura.extract(
        clean_html, 
        output_format="xml", 
        include_comments=False, 
        include_tables=False,
        favor_recall=fallback
    )
    
    if not xml_output:
        # Fallback to just getting all text from soup if trafilatura fails
        text = soup.get_text(separator='\n', strip=True)
        return {"[CONTENT]": text}
        
    sections = {}
    current_topic = "[DESCRIPTION]"
    current_text = []
    
    try:
        root = ET.fromstring(xml_output)
    except ET.ParseError:
        return {"[CONTENT]": BeautifulSoup(xml_output, 'html.parser').get_text(separator='\n', strip=True)}
        
    def process_element(elem):
        nonlocal current_topic, current_text
        if elem.tag == 'head':
            # Save previous topic
            if current_text:
                joined_text = "\n".join(current_text).strip()
                if joined_text:
                    if current_topic not in sections:
                        sections[current_topic] = ""
                    sections[current_topic] += joined_text + "\n"
            
            header_text = "".join(elem.itertext()).strip().upper()
            current_topic = f"[{header_text}]"
            current_text = []
        elif elem.tag == 'p':
            text = "".join(elem.itertext()).strip()
            if text:
                if text.startswith("__NINDS_HEADER__") or text.startswith("__RARE_HEADER__"):
                    if current_text:
                        joined_text = "\n".join(current_text).strip()
                        if joined_text:
                            if current_topic not in sections:
                                sections[current_topic] = ""
                            sections[current_topic] += joined_text + "\n"
                    
                    header_text = text.replace("__NINDS_HEADER__", "").replace("__RARE_HEADER__", "").strip().upper()
                    if not header_text.startswith("["):
                        current_topic = f"[{header_text}]"
                    else:
                        current_topic = header_text
                    current_text = []
                else:
                    current_text.append(text)
        elif elem.tag == 'item':
            text = "".join(elem.itertext()).strip()
            if text:
                # Add bullet point
                current_text.append("- " + text)
        else:
            for child in elem:
                process_element(child)

    process_element(root)
    
    # Save the last topic
    if current_text:
        joined_text = "\n".join(current_text).strip()
        if joined_text:
            if current_topic not in sections:
                sections[current_topic] = ""
            sections[current_topic] += joined_text + "\n"
            
    # ---------------------------------------------------------
    # DOMAIN SPECIFIC POST-PROCESSING (DICTIONARY LEVEL)
    # ---------------------------------------------------------
    final_sections = {}
    
    for topic, content in sections.items():
        topic_clean = topic.strip()
        
        if 'REFERENCES' in topic_clean.upper():
            continue
            
        content_lines = content.split('\n')
        filtered_lines = [line for line in content_lines if 'For more information' not in line]
        content = '\n'.join(filtered_lines).strip()
        
        if not content:
            continue
        
        if 'ghr.nlm.nih.gov' in domain:
            if 'REFERENCES' in topic_clean:
                break # Stop processing, exclude references and everything below
                
        elif 'cancer.gov' in domain:
            if '[TO LEARN' in topic_clean:
                break
                
            cancer_noise_topics = ['[ABOUT PDQ]', '[PURPOSE OF THIS SUMMARY]', '[REVIEWERS AND UPDATES]', '[CLINICAL TRIAL INFORMATION]', '[PERMISSION TO USE THIS SUMMARY]', '[DISCLAIMER]', '[CONTACT US]']
            if any(nt in topic_clean for nt in cancer_noise_topics):
                continue
                
            if len(content.split()) < 50 and ('visit our' in content.lower() or 'available at' in content.lower() or 'is available from the nci' in content.lower()):
                continue
                
            content_lines = content.split('\n')
            filtered_lines = []
            for line in content_lines:
                clean_line = line.strip()
                if clean_line.startswith('Learn more about') or \
                   clean_line.startswith('Use our clinical trial search') or \
                   clean_line.startswith('You can use the clinical trial search') or \
                   clean_line.startswith('General information about clinical trials') or \
                   'Information about clinical trials is available from the NCI' in clean_line:
                    continue
                filtered_lines.append(line)
            content = '\n'.join(filtered_lines).strip()
            
            content = re.sub(r'To learn more, visit Questions to Ask Your Doctor [Aa]bout Treatment\.?', '', content).strip()
            
            if not content:
                continue
                
        elif 'niddk.nih.gov' in domain:
            if 'CLINICAL TRAITS' in topic_clean or 'CLINICAL TRIALS' in topic_clean or 'REFERENCES' in topic_clean:
                break
            if topic_clean == '[]' or topic_clean == '[TOP]':
                continue
                
            content_lines = content.split('\n')
            filtered_lines = []
            for line in content_lines:
                clean_line = line.strip()
                if clean_line.isdigit():
                    continue
                filtered_lines.append(line)
            content = '\n'.join(filtered_lines)
            
            content = content.replace('[Top]', '').replace('[]', '').replace('[TOP]', '').strip()
            if not content:
                continue
                
        elif 'ninds.nih.gov' in domain:
            if 'WHAT RESEARCH IS BEING DONE' in topic_clean:
                break # Stop processing
                
        elif 'rarediseases.info.nih.gov' in domain:
            noise_topics = ['FIND A SPECIALIST', 'RESEARCH', 'ORGANIZATIONS', 'LEARN MORE', 'NEWS & EVENTS', 'GARD ANSWERS', 'REFERENCES']
            skip = False
            for nt in noise_topics:
                if nt in topic_clean:
                    skip = True
                    break
            if skip or 'SHOWING' in topic_clean:
                continue
            
            # Remove the 'Listen' screen reader artifact from the start of the content
            content = re.sub(r'^Listen\s*', '', content.strip())
                
        elif 'nlm.nih.gov' in domain:
            if 'METHODOLOGY' in topic_clean:
                break
            
            if 'INTERACTIONS' in topic_clean:
                lines = content.split('\n')
                new_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('- ') and i + 1 < len(lines) and lines[i+1].strip().startswith('- '):
                        if len(line) < 100 and not line.endswith('.'):
                            merged = line + ": " + lines[i+1].strip()[2:]
                            new_lines.append(merged)
                            i += 2
                            continue
                    new_lines.append(line)
                    i += 1
                content = '\n'.join(new_lines)
                
        elif 'cdc.gov' in domain:
            if 'CONTACT US' in topic_clean or 'HOW CAN I LEARN MORE ABOUT THIS' in topic_clean:
                continue
                
            content = re.sub(r'(?i)This symbol means you are leaving the CDC\.gov Web site.*?Notification and Disclaimer policy\.', '', content, flags=re.DOTALL)
            content = re.sub(r'(?i)File Formats: All viewers, players, and plug-ins used on this site can be downloaded from the file formats page.*?(?:etc\.\))', '', content, flags=re.DOTALL)
            content = re.sub(r'(?i)Content last reviewed.*', '', content)
            content = re.sub(r'(?i)Page last reviewed.*', '', content)
            content = re.sub(r'(?i)Content Source:.*', '', content)
            
        content = content.strip()
        if not content:
            continue
            
        final_sections[topic_clean] = content
        
    # ---------------------------------------------------------
    # GLOBAL POST-PROCESSING: TOPIC LENGTH OPTIMIZATION
    # ---------------------------------------------------------
    
    # 1. Remove topics with < 10 words
    filtered_sections = {}
    if 'ninds.nih.gov' not in domain:
        for k, v in final_sections.items():
            if len(v.split()) >= 10:
                filtered_sections[k] = v
    else:
        filtered_sections = final_sections.copy()
            
    # 2. Merge topics with <= 40 words into adjacent topics with lower word count
    if 'ninds.nih.gov' not in domain:
        keys = list(filtered_sections.keys())
        i = 0
        while i < len(keys):
            k = keys[i]
            content = filtered_sections[k]
            word_count = len(content.split())
            
            if word_count <= 40:
                prev_k = keys[i-1] if i > 0 else None
                next_k = keys[i+1] if i < len(keys) - 1 else None
                
                if not prev_k and not next_k:
                    i += 1
                    continue
                    
                prev_count = len(filtered_sections[prev_k].split()) if prev_k else float('inf')
                next_count = len(filtered_sections[next_k].split()) if next_k else float('inf')
                
                if prev_count <= next_count:
                    filtered_sections[prev_k] = filtered_sections[prev_k].strip() + "\n\n" + k + "\n" + content.strip()
                    del filtered_sections[k]
                    keys.pop(i)
                else:
                    filtered_sections[next_k] = k + "\n" + content.strip() + "\n\n" + filtered_sections[next_k].strip()
                    del filtered_sections[k]
                    keys.pop(i)
            else:
                i += 1
            
    if 'rarediseases.info.nih.gov' in domain:
        new_sections = {}
        
        for topic, content in filtered_sections.items():
            if 'NAVIGATING HEALTH CARE DECISIONS' in topic:
                break
                
            if topic in ['[SYMPTOMS]', 'SYMPTOMS']:
                content = re.sub(r'(?i)\d+\s+Symptoms', '', content).strip()
                
            new_sections[topic] = content
            
        filtered_sections = new_sections
        
    # Merge bs4_sections AT THE VERY END, so they aren't truncated by 'NAVIGATING HEALTH CARE DECISIONS'
    for topic, content in bs4_sections.items():
        topic_clean = topic.strip()
        if 'rarediseases.info.nih.gov' in domain:
            noise_topics = ['FIND A SPECIALIST', 'RESEARCH', 'ORGANIZATIONS', 'LEARN MORE', 'NEWS & EVENTS', 'GARD ANSWERS', 'REFERENCES']
            if any(nt in topic_clean for nt in noise_topics) or 'SHOWING' in topic_clean:
                continue
                
            # Check for modern 2024 redirect noise and table pagination
            if 'Patient organizations can help patients' in content or 'Clinical studies are part of clinical research' in content:
                continue
            if 'Showing of' in content or 'Showing 1 to' in content:
                continue
                
            content = re.sub(r'^Listen\s*', '', content.strip())
            
            if topic_clean == '[DESCRIPTION]':
                # Apply regex cleanups for 2023+ layout boilerplate
                content = re.sub(r'(?i)skip to main content.*?rare diseases', '', content, flags=re.DOTALL)
                content = re.sub(r'Patient organizations can help.*?Questions about the information or resources it provides\.', '', content, flags=re.DOTALL)
                content = re.sub(r'Clinical studies are part of.*?terms/synonyms to improve results\.', '', content, flags=re.DOTALL)
                content = re.sub(r'Thank you for visiting the GARD website.*?Feedback Form', '', content, flags=re.DOTALL)
                content = re.sub(r'Thank you for visiting the new GARD website.*?GARD website\.', '', content, flags=re.DOTALL)
                content = re.sub(r'Thank you for visiting the GARD website.*?GARD website\.', '', content, flags=re.DOTALL)
                content = re.sub(r'We are currently developing a new version of GARD.*?(?:site|version\.)', '', content, flags=re.DOTALL)
                content = re.sub(r'Data from Orphanet and Human Phenotype Ontology.*?estimates, and more\.', '', content, flags=re.DOTALL)
                content = re.sub(r'Take steps toward getting a diagnosis.*?medical care\.', '', content, flags=re.DOTALL)
                content = re.sub(r'\d+\s+Symptoms', '', content)
                content = re.sub(r'\d+\s+Organizations', '', content)
                content = re.sub(r'ClinicalTrials.gov, an affiliate of NIH, provides current information.*?Except:\s*Federal Holidays\)', '', content, flags=re.DOTALL)
                content = re.sub(r'GARDGenetic and Rare Diseases.*?Eastern Time', '', content, flags=re.DOTALL)
                
            lines = content.split('\n')
            clean_lines = []
            for L in lines:
                L_stripped = L.strip()
                if L_stripped in ['Title', 'See More', 'Back to top', 'A-', 'Close', 'Suggest an update', 'Read More', 'Read Less'] or 'placeholder for the horizontal scroll slider' in L_stripped:
                    continue
                if L_stripped.startswith('Other Names:') or L_stripped.startswith('Categories:') or L_stripped.endswith(';'):
                    continue
                if L_stripped.lower().startswith('learn more about'):
                    continue
                clean_lines.append(L)
            content = '\n'.join(clean_lines).strip()
            if topic_clean == '[DESCRIPTION]' and len(content) < 50 and '.' not in content:
                continue
        
        filtered_sections[topic_clean] = content.strip()
            
    # Strip brackets from all keys to avoid double brackets when the scraper script writes them out
    # Also globally strip any line starting with 'Learn more about'
    final_cleaned_sections = {}
    for k, v in filtered_sections.items():
        clean_key = k.replace('[', '').replace(']', '').strip()
        
        # Skip REFERENCES globally
        if 'REFERENCES' in clean_key.upper():
            continue
            
        # Skip EXPLORE for cdc.gov
        if 'cdc.gov' in domain and 'EXPLORE' in clean_key.upper():
            continue
            
        if clean_key:
            lines = v.split('\n')
            clean_lines = []
            for L in lines:
                L_clean = L.strip()
                if L_clean.lower().startswith('learn more about'):
                    continue
                if L_clean.startswith('- Organizations:') or L_clean.startswith('- Categories:'):
                    continue
                if 'rare-source' in L_clean.lower():
                    continue
                if 'RESOURCE(S) FOR MEDICAL PROFESSIONALS' in L_clean:
                    continue
                if 'for more information' in L_clean.lower():
                    continue
                clean_lines.append(L)
            
            cleaned_val = '\n'.join(clean_lines).strip()
            if cleaned_val:
                final_cleaned_sections[clean_key] = cleaned_val
            
    return final_cleaned_sections
