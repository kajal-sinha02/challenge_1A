import re
import json
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
from dataclasses import dataclass
from collections import Counter, defaultdict
import statistics

@dataclass
class TextBlock:
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    is_underlined: bool
    page_num: int
    bbox: Tuple[float, float, float, float]
    y_position: float
    line_height: float
    spacing_before: float
    spacing_after: float
    spans: List[Dict]  # Store original spans for detailed analysis

class SmartHeadingDetector:
    def __init__(self):
        self.heading_indicators = {
            'structural': [
                r'^\d+\.\s+[A-Z]',  # 1. Introduction
                r'^\d+\.\d+\s+[A-Z]',  # 2.1 Subsection
                r'^\d+\.\d+\.\d+\s+[A-Z]',  # 2.1.1 Sub-subsection
                r'^[A-Z][A-Z\s&-]{8,}$',  # ALL CAPS HEADERS
                r'^(Chapter|Section|Part|Appendix)\s+[A-Z0-9]',
                r'^Appendix\s+[A-Z]:?',
            ],
            'content_keywords': [
                'introduction', 'background', 'summary', 'conclusion', 'overview',
                'methodology', 'approach', 'evaluation', 'references', 'appendix',
                'table of contents', 'acknowledgments', 'abstract', 'objectives',
                'requirements', 'implementation', 'timeline', 'milestones'
            ],
            'formatting_indicators': [
                'ends_with_colon', 'all_caps', 'title_case', 'bold_text',
                'larger_font', 'isolated_line', 'centered_text'
            ]
        }
    
    def extract_text_blocks(self, pdf_path: str) -> List[TextBlock]:
        """Extract text blocks with comprehensive formatting information"""
        doc = fitz.open(pdf_path)
        blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_dict = page.get_text("dict")
            page_height = page.rect.height
            
            # Get all text blocks on the page
            text_blocks = []
            for block in page_dict["blocks"]:
                if "lines" in block:
                    for line_idx, line in enumerate(block["lines"]):
                        # Store spans for detailed analysis
                        line_spans = line["spans"]
                        
                        # Reconstruct line text and get dominant formatting
                        line_text = ""
                        font_sizes = []
                        font_names = []
                        bold_flags = []
                        underline_flags = []
                        
                        for span in line_spans:
                            line_text += span["text"]
                            font_sizes.append(span["size"])
                            font_names.append(span["font"])
                            bold_flags.append(bool(span["flags"] & 2**4))
                            underline_flags.append(bool(span["flags"] & 2**2))
                        
                        line_text = line_text.strip()
                        if len(line_text) > 2:  # Ignore very short text
                            # Use dominant formatting
                            avg_font_size = statistics.mean(font_sizes) if font_sizes else 12
                            most_common_font = Counter(font_names).most_common(1)[0][0] if font_names else ""
                            is_bold = any(bold_flags)
                            is_underlined = any(underline_flags)
                            
                            # Calculate spacing
                            bbox = line["bbox"]
                            spacing_before = 0
                            spacing_after = 0
                            
                            if line_idx > 0:
                                prev_bbox = block["lines"][line_idx-1]["bbox"]
                                spacing_before = bbox[1] - prev_bbox[3]
                            
                            if line_idx < len(block["lines"]) - 1:
                                next_bbox = block["lines"][line_idx+1]["bbox"]
                                spacing_after = next_bbox[1] - bbox[3]
                            
                            text_block = TextBlock(
                                text=line_text,
                                font_size=avg_font_size,
                                font_name=most_common_font,
                                is_bold=is_bold,
                                is_italic="Italic" in most_common_font,
                                is_underlined=is_underlined,
                                page_num=page_num,
                                bbox=bbox,
                                y_position=bbox[1],
                                line_height=bbox[3] - bbox[1],
                                spacing_before=max(0, spacing_before),
                                spacing_after=max(0, spacing_after),
                                spans=line_spans
                            )
                            text_blocks.append(text_block)
            
            blocks.extend(text_blocks)
        
        doc.close()
        return blocks
    
    def clip_heading_at_format_change(self, block: TextBlock) -> str:
        """
        Clip heading text where font size changes or underline ends.
        Returns the clipped heading text.
        """
        if not block.spans or len(block.spans) <= 1:
            return block.text.strip()
        
        # Get the formatting of the first span (assumed to be heading format)
        first_span = block.spans[0]
        reference_font_size = first_span["size"]
        reference_is_bold = bool(first_span["flags"] & 2**4)
        reference_is_underlined = bool(first_span["flags"] & 2**2)
        
        # Build text incrementally until format changes
        clipped_text = ""
        
        for span in block.spans:
            current_font_size = span["size"]
            current_is_bold = bool(span["flags"] & 2**4)
            current_is_underlined = bool(span["flags"] & 2**2)
            
            # Check for significant font size change (more than 10% difference)
            font_size_changed = abs(current_font_size - reference_font_size) > (reference_font_size * 0.1)
            
            # Check for bold formatting change
            bold_changed = current_is_bold != reference_is_bold
            
            # Check for underline formatting change
            underline_changed = current_is_underlined != reference_is_underlined
            
            # If any significant formatting change occurs, stop here
            if font_size_changed or bold_changed or underline_changed:
                break
            
            # Add this span's text to the clipped text
            clipped_text += span["text"]
        
        # Clean up the clipped text
        clipped_text = clipped_text.strip()
        
        # If we clipped nothing or everything, return original
        if not clipped_text or clipped_text == block.text.strip():
            return block.text.strip()
        
        # Additional cleanup: don't break in the middle of words
        original_words = block.text.strip().split()
        clipped_words = clipped_text.split()
        
        # If we're in the middle of breaking a word, include the full word
        if clipped_words and original_words:
            last_clipped_word = clipped_words[-1]
            corresponding_original_idx = len(clipped_words) - 1
            
            if (corresponding_original_idx < len(original_words) and 
                not original_words[corresponding_original_idx].startswith(last_clipped_word)):
                # We're in the middle of a word, find the complete word
                for i, word in enumerate(original_words):
                    if word.startswith(last_clipped_word):
                        # Include complete words up to this point
                        clipped_text = " ".join(original_words[:i+1])
                        break
        
        return clipped_text
    
    def detect_underline_end_in_text(self, block: TextBlock) -> str:
        """
        Detect where underline formatting ends and clip the heading there.
        """
        if not block.spans or not block.is_underlined:
            return block.text.strip()
        
        # Find where underline ends
        underlined_text = ""
        
        for span in block.spans:
            is_underlined = bool(span["flags"] & 2**2)
            
            if is_underlined:
                underlined_text += span["text"]
            else:
                # Underline ended, stop here
                break
        
        # Clean up and return
        underlined_text = underlined_text.strip()
        
        # If no underlined text found or all text is underlined, return original
        if not underlined_text or underlined_text == block.text.strip():
            return block.text.strip()
        
        return underlined_text
    
    def get_clipped_heading_text(self, block: TextBlock) -> str:
        """
        Main method to clip heading text based on font changes and underline endings.
        """
        original_text = block.text.strip()
        
        # First, clip at font size/bold changes
        font_clipped = self.clip_heading_at_format_change(block)
        
        # Then, clip at underline end if applicable
        underline_clipped = self.detect_underline_end_in_text(block)
        
        # Use the shortest clipped version (most conservative clipping)
        clipped_options = [
            original_text,
            font_clipped,
            underline_clipped
        ]
        
        # Filter out empty strings
        valid_options = [opt for opt in clipped_options if opt and len(opt.strip()) > 0]
        
        if not valid_options:
            return original_text
        
        # Return the shortest valid option (most aggressive but safe clipping)
        shortest = min(valid_options, key=len)
        
        # But ensure we don't clip too aggressively (keep at least 3 characters)
        if len(shortest.strip()) >= 3:
            return shortest
        else:
            return original_text
    
    def analyze_document_typography(self, blocks: List[TextBlock]) -> Dict:
        """Analyze document typography to establish baselines"""
        font_sizes = [block.font_size for block in blocks]
        line_heights = [block.line_height for block in blocks]
        spacings = [block.spacing_before for block in blocks if block.spacing_before > 0]
        
        body_font_size = statistics.median(font_sizes)
        avg_line_height = statistics.median(line_heights)
        avg_spacing = statistics.median(spacings) if spacings else 0
        
        # Identify potential heading font sizes (above body text)
        large_fonts = [fs for fs in font_sizes if fs > body_font_size * 1.1]
        heading_font_thresholds = {
            'h1': body_font_size * 1.4,
            'h2': body_font_size * 1.2,
            'h3': body_font_size * 1.1
        }
        
        return {
            'body_font_size': body_font_size,
            'avg_line_height': avg_line_height,
            'avg_spacing': avg_spacing,
            'heading_thresholds': heading_font_thresholds,
            'max_font_size': max(font_sizes),
            'min_font_size': min(font_sizes)
        }
    
    def is_line_aligned_properly(self, block: TextBlock, is_title: bool = False) -> bool:
        """Check if text is properly aligned for heading/title"""
        page_width = 595  # Standard A4 width in points
        left_margin = 50   # Typical left margin
        right_margin = 545 # Typical right margin
        
        text_left = block.bbox[0]
        text_right = block.bbox[2]
        text_center = (text_left + text_right) / 2
        page_center = page_width / 2
        
        if is_title:
            # Title must be center aligned (within reasonable tolerance)
            center_tolerance = 80  # Allow some deviation from perfect center
            return abs(text_center - page_center) < center_tolerance
        else:
            # Headings can be left-aligned or center-aligned
            is_left_aligned = abs(text_left - left_margin) < 30
            is_center_aligned = abs(text_center - page_center) < 60
            return is_left_aligned or is_center_aligned

    def has_mixed_formatting_in_line(self, block: TextBlock) -> bool:
        """Check if the line has mixed bold/regular text (indicates not a heading)"""
        if not block.spans or len(block.spans) <= 1:
            return False
        
        # Check if there's mixed bold/regular formatting
        bold_count = sum(1 for span in block.spans if span["flags"] & 2**4)
        total_spans = len(block.spans)
        
        # If some spans are bold and some aren't, it's mixed formatting
        return 0 < bold_count < total_spans

    def calculate_heading_score(self, block: TextBlock, typography: Dict, 
                              prev_block: Optional[TextBlock] = None,
                              next_block: Optional[TextBlock] = None) -> float:
        """Calculate likelihood that a text block is a heading (0-100 score)"""
        score = 0
        
        # Use clipped text for analysis
        text = self.get_clipped_heading_text(block)
        
        # Check alignment first - must be left or center aligned
        if not self.is_line_aligned_properly(block, is_title=False):
            return 0  # Not properly aligned, can't be heading
        
        # Check for mixed formatting in line (disqualifies as heading)
        if self.has_mixed_formatting_in_line(block):
            return 0  # Mixed bold/regular in same line, not a heading
        
        # Font size scoring (0-30 points) - more weight on size
        font_ratio = block.font_size / typography['body_font_size']
        if font_ratio >= 1.5:
            score += 30
        elif font_ratio >= 1.3:
            score += 25
        elif font_ratio >= 1.15:
            score += 20
        elif font_ratio >= 1.1:
            score += 15
        elif font_ratio >= 1.05:
            score += 8
        else:
            # If not bigger than body text, needs to be bold to qualify
            if not block.is_bold:
                return 0
        
        # Bold formatting requirement (0-20 points)
        if block.is_bold:
            score += 20
        elif font_ratio < 1.2:  # If not significantly larger, must be bold
            return 0
        
        # Underline bonus (0-10 points)
        if block.is_underlined:
            score += 10
        
        # Structural patterns (0-25 points)
        for pattern in self.heading_indicators['structural']:
            if re.match(pattern, text):
                score += 25
                break
        
        # Spacing analysis (0-15 points)
        spacing_score = 0
        
        # Space before heading (common pattern)
        if block.spacing_before > typography['avg_spacing'] * 1.3:
            spacing_score += 8
        
        # Space after heading
        if next_block and block.spacing_after > typography['avg_spacing'] * 1.2:
            spacing_score += 7
        
        score += min(spacing_score, 15)
        
        # Content and format indicators (0-10 points)
        format_score = 0
        
        # Ends with colon (section headers)
        if text.endswith(':'):
            format_score += 5
        
        # Short and concise (headings are typically brief)
        if 3 <= len(text.split()) <= 12:
            format_score += 3
        
        # Title case or sentence case
        if text.istitle() or (text[0].isupper() and not text.isupper()):
            format_score += 2
        
        score += min(format_score, 10)
        
        # Check that it's smaller than title font
        max_font_on_page = typography.get('max_font_size', block.font_size)
        if block.font_size >= max_font_on_page * 0.95:  # Too close to title size
            score -= 20
        
        # Ensure it's a standalone line (headings are typically isolated)
        if len(text) > 150:  # Too long for typical heading
            score -= 15
        
        # Content keywords bonus (0-5 points)
        text_lower = text.lower()
        for keyword in self.heading_indicators['content_keywords']:
            if keyword in text_lower:
                score += 5
                break
        
        return max(0, min(score, 100))
    
    def classify_heading_level(self, block: TextBlock, typography: Dict, score: float) -> str:
        """Classify heading level based on font size hierarchy and formatting"""
        font_ratio = block.font_size / typography['body_font_size']
        text = self.get_clipped_heading_text(block)
        
        # Get relative font sizes for hierarchy
        title_font_ratio = typography['max_font_size'] / typography['body_font_size']
        
        # Numbered sections get priority in hierarchy
        if re.match(r'^\d+\.\s+', text):
            return "H1"
        elif re.match(r'^\d+\.\d+\s+', text):
            return "H2"
        elif re.match(r'^\d+\.\d+\.\d+\s+', text):
            return "H3"
        
        # Appendix sections are typically H2 level
        if re.match(r'^Appendix\s+[A-Z]', text, re.IGNORECASE):
            return "H2"
        
        # Font size and formatting based hierarchy
        if font_ratio >= title_font_ratio * 0.85:
            # Too close to title size, likely H1 but could be misidentified title
            return "H1"
        elif font_ratio >= 1.2 or (font_ratio >= 1.15 and block.is_bold and score >= 75):
            return "H1"
        elif font_ratio >= 1.0 or (font_ratio >= 0.95 and block.is_bold and score >= 60):
            return "H2"
        elif font_ratio >= 0.9 or (font_ratio >= 0.85 and block.is_bold and score >= 45):
            return "H3"
        elif block.is_bold and font_ratio >= 1.02:
            return "H4"
        else:
            return "H3"  # Default for edge cases
    
    def filter_false_positives(self, candidates: List[Dict]) -> List[Dict]:
        """Remove likely false positives"""
        filtered = []
        
        for candidate in candidates:
            text = candidate['text'].strip()
            
            # Skip very common false positives
            if any(skip in text.lower() for skip in [
                'page ', 'copyright', '©', 'version', 'date:', 'figure', 'table',
                'www.', 'http', '@', '.com', '.org'
            ]):
                continue
            
            # Skip pure numbers or very short text
            if re.match(r'^\d+$', text) or len(text) < 3:
                continue
            
            # Skip if looks like a sentence (ends with period, too long)
            if text.endswith('.') and len(text) > 100:
                continue
            
            # Skip repeated characters (OCR artifacts)
            if len(set(text.replace(' ', ''))) < 3 and len(text) > 5:
                continue
            
            filtered.append(candidate)
        
        return filtered
    
    def extract_title(self, blocks: List[TextBlock]) -> str:
        """Extract document title - must be center aligned, largest font, and bold"""
        first_page_blocks = [b for b in blocks if b.page_num == 0]
        
        if not first_page_blocks:
            return ""
        
        # Find the largest font size on first page
        max_font_size = max(block.font_size for block in first_page_blocks)
        
        # Get all blocks with the largest font size that are center-aligned and bold
        title_candidates = []
        for block in first_page_blocks:
            if (abs(block.font_size - max_font_size) < 0.5 and  # Largest font
                block.is_bold and  # Must be bold
                self.is_line_aligned_properly(block, is_title=True) and  # Must be centered
                block.y_position < 400):  # In top portion
                title_candidates.append(block)
        
        if not title_candidates:
            # Fallback: look for any large, bold, centered text
            for block in first_page_blocks:
                font_ratio = block.font_size / (sum(b.font_size for b in first_page_blocks) / len(first_page_blocks))
                if (font_ratio > 1.3 and block.is_bold and 
                    self.is_line_aligned_properly(block, is_title=True) and
                    block.y_position < 350):
                    title_candidates.append(block)
        
        if not title_candidates:
            return ""
        
        # Sort title candidates by vertical position (top to bottom)
        title_candidates.sort(key=lambda x: x.y_position)
        
        # Group consecutive blocks that might be part of multi-line title
        title_lines = []
        current_group = [title_candidates[0]]
        
        for i in range(1, len(title_candidates)):
            current_block = title_candidates[i]
            prev_block = title_candidates[i-1]
            
            # Check if blocks are close vertically (part of same title)
            vertical_distance = current_block.y_position - prev_block.y_position
            
            # If close enough (within reasonable line spacing), group together
            if vertical_distance < max_font_size * 2.5:  # Within 2.5x font size distance
                current_group.append(current_block)
            else:
                # Start new group
                title_lines.append(current_group)
                current_group = [current_block]
        
        # Add the last group
        title_lines.append(current_group)
        
        # Find the best title group (prefer the first one that meets criteria)
        for group in title_lines:
            combined_text = " ".join(self.get_clipped_heading_text(block) for block in group)
            word_count = len(combined_text.split())
            
            # Title criteria:
            # 1. All blocks must be bold and center-aligned (already filtered)
            # 2. Reasonable length (2-20 words as specified)
            # 3. Not a numbered section
            # 4. Substantial content
            if (2 <= word_count <= 20 and 
                not re.match(r'^\d+\.', combined_text.strip()) and
                len(combined_text.strip()) > 5):
                
                # Combine all text from the title group into single line
                title_text = " ".join(self.get_clipped_heading_text(block) for block in group)
                # Clean up extra whitespace
                title_text = re.sub(r'\s+', ' ', title_text).strip()
                return title_text
        
        return ""
    
    def detect_headings(self, pdf_path: str, min_score: float = 25) -> Dict:
        """Main heading detection method with improved logic and clipping"""
        blocks = self.extract_text_blocks(pdf_path)
        typography = self.analyze_document_typography(blocks)
        
        # Score all blocks
        candidates = []
        for i, block in enumerate(blocks):
            prev_block = blocks[i-1] if i > 0 else None
            next_block = blocks[i+1] if i < len(blocks)-1 else None
            
            score = self.calculate_heading_score(block, typography, prev_block, next_block)
            
            if score >= min_score:
                level = self.classify_heading_level(block, typography, score)
                clipped_text = self.get_clipped_heading_text(block)
                
                candidates.append({
                    'text': clipped_text,
                    'original_text': block.text.strip(),
                    'level': level,
                    'page': block.page_num,
                    'score': score,
                    'font_size': block.font_size,
                    'is_bold': block.is_bold,
                    'is_underlined': block.is_underlined,
                    'font_ratio': block.font_size / typography['body_font_size'],
                    'was_clipped': clipped_text != block.text.strip()
                })
        
        # Remove duplicates and false positives
        seen_texts = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate['text'] not in seen_texts:
                unique_candidates.append(candidate)
                seen_texts.add(candidate['text'])
        
        filtered_candidates = self.filter_false_positives(unique_candidates)
        
        # Additional filtering: remove text that's too similar to title
        title = self.extract_title(blocks)
        if title:
            title_words = set(title.lower().split())
            final_candidates = []
            for candidate in filtered_candidates:
                candidate_words = set(candidate['text'].lower().split())
                # If more than 60% of words overlap with title, likely not a separate heading
                overlap = len(title_words & candidate_words) / len(candidate_words) if candidate_words else 0
                if overlap < 0.6:
                    final_candidates.append(candidate)
            filtered_candidates = final_candidates
        
        # Sort by page and score
        filtered_candidates.sort(key=lambda x: (x['page'], -x['score']))
        
        # Format output
        outline = []
        for candidate in filtered_candidates:
            outline.append({
                'level': candidate['level'],
                'text': candidate['text'],
                'page': candidate['page'],
            })
        
        return {
            'title': title,
            'outline': outline,
            'debug_info': {
                'total_blocks': len(blocks),
                'candidates_found': len(candidates),
                'after_filtering': len(filtered_candidates),
                'typography': typography,
                'clipped_headings': sum(1 for c in filtered_candidates if c.get('was_clipped', False))
            }
        }

def main():
    """Test the detector"""
    detector = SmartHeadingDetector()
    
    test_files = ["file01.pdf", "file02.pdf", "file03.pdf", "file04.pdf", "file05.pdf"]
    
    for pdf_file in test_files:
        try:
            print(f"\n=== Processing {pdf_file} ===")
            result = detector.detect_headings(pdf_file)
            
            output_file = pdf_file.replace('.pdf', '_smart_detected_clipped.json')
            
            # Clean output for JSON
            clean_result = {
                "title": result["title"],
                "outline": result["outline"]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_result, f, indent=2, ensure_ascii=False)
            
            print(f"Title: {result['title'][:60]}...")
            print(f"Found {len(result['outline'])} headings")
            print(f"Clipped headings: {result['debug_info']['clipped_headings']}")
            print(f"Body font size: {result['debug_info']['typography']['body_font_size']:.1f}")
            
            for heading in result['outline'][:5]:  # Show first 5
                clipped_indicator = " [CLIPPED]" if heading.get('was_clipped', False) else ""
                print(f"  {heading['level']}: {heading['text'][:50]}...{clipped_indicator}")
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

if __name__ == "__main__":
    main() 