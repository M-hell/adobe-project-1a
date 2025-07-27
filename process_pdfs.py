#!/usr/bin/env python3
"""
PDF Processing Script - Extracts structured information from PDFs
and outputs JSON files with titles, headings, and descriptions.
"""

import os
import json
import fitz  # PyMuPDF
import re
from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path
import time

class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor with the offline model."""
        print("🔄 Loading AI model...")
        
        # Import here to avoid any issues
        from sentence_transformers import SentenceTransformer
        
        # Let's debug what's actually in the models directory
        print("📁 Checking models directory...")
        if os.path.exists('/app/models'):
            print("✅ /app/models directory exists")
            for root, dirs, files in os.walk('/app/models'):
                print(f"📂 Directory: {root}")
                if files:
                    print(f"📄 Files: {files[:10]}")  # Show first 10 files
                if dirs:
                    print(f"📁 Subdirs: {dirs[:10]}")  # Show first 10 subdirs
        else:
            print("❌ /app/models directory does not exist!")
            raise Exception("Models directory not found!")
        
        # Try multiple approaches to find and load the model
        model_loaded = False
        
        print("🔍 Approach 1: Searching for model directory...")
        for root, dirs, files in os.walk('/app/models'):
            if 'config.json' in files:
                print(f"🎯 Found config.json in: {root}")
                if any('.bin' in f or '.safetensors' in f for f in files):
                    print(f"🎯 Found model files in: {root}")
                    try:
                        self.model = SentenceTransformer(root)
                        print("✅ Model loaded successfully from auto-detected path!")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"❌ Failed to load from {root}: {e}")
        
        if not model_loaded:
            print("🔍 Approach 2: Trying common paths...")
            common_paths = [
                '/app/models/sentence-transformers_all-MiniLM-L6-v2',
                '/app/models/models--sentence-transformers--all-MiniLM-L6-v2',
            ]
            for path in common_paths:
                print(f"🔍 Checking path: {path}")
                if os.path.exists(path):
                    print(f"✅ Path exists: {path}")
                    try:
                        self.model = SentenceTransformer(path)
                        print("✅ Model loaded successfully from common path!")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"❌ Failed to load from {path}: {e}")
                else:
                    print(f"❌ Path does not exist: {path}")
        
        if not model_loaded:
            print("🔍 Approach 3: Trying to load from cache...")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models')
                print("✅ Model loaded successfully from cache!")
                model_loaded = True
            except Exception as e:
                print(f"❌ Failed to load from cache: {e}")
        
        if not model_loaded:
            print("❌ Could not load the AI model!")
            print("🔧 Using fallback mode - will work without AI clustering")
            self.model = None
        else:
            print("🎉 Model initialization complete!")
    
    def extract_text_with_structure(self, pdf_path):
        """Extract text from PDF while preserving structure information."""
        doc = fitz.open(pdf_path)
        
        text_blocks = []
        all_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text blocks with font information
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        font_sizes = []
                        
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                line_text += text + " "
                                font_sizes.append(span["size"])
                        
                        if line_text.strip():
                            avg_font_size = np.mean(font_sizes) if font_sizes else 12
                            text_blocks.append({
                                'text': line_text.strip(),
                                'font_size': avg_font_size,
                                'page': page_num + 1
                            })
                            all_text += line_text
        
        doc.close()
        return text_blocks, all_text
    
    def identify_title_and_headings(self, text_blocks):
        """Identify title and headings based on font size and content."""
        if not text_blocks:
            return None, []
        
        # Calculate font size statistics
        font_sizes = [block['font_size'] for block in text_blocks]
        avg_font_size = np.mean(font_sizes)
        max_font_size = max(font_sizes)
        
        title = None
        headings = []
        
        # Find title (usually the largest text on first page)
        title_candidates = [
            block for block in text_blocks[:10]  # Look in first 10 blocks
            if block['font_size'] >= max_font_size * 0.9 and len(block['text']) > 10
        ]
        
        if title_candidates:
            # Choose the longest candidate as title
            title = max(title_candidates, key=lambda x: len(x['text']))['text']
        
        # Find headings with hierarchical levels based on font size
        heading_threshold = avg_font_size * 1.2
        
        # Collect headings with font sizes and pages
        heading_candidates = []
        for block in text_blocks:
            text = block['text']
            if (block['font_size'] >= heading_threshold and 
                len(text) > 5 and len(text) < 200 and
                text != title):
                
                # Basic heading patterns
                if (text.isupper() or 
                    re.match(r'^[A-Z][a-z].*[^.]$', text) or
                    re.match(r'^\d+\.?\s+[A-Z]', text)):
                    heading_candidates.append({
                        'text': text,
                        'font_size': block['font_size'],
                        'page': block['page']
                    })
        
        # Sort by font size (largest first) to determine hierarchy
        heading_candidates.sort(key=lambda x: x['font_size'], reverse=True)
        
        # Assign hierarchical levels (H1, H2, H3)
        if heading_candidates:
            # Get unique font sizes for level assignment
            unique_sizes = sorted(list(set([h['font_size'] for h in heading_candidates])), reverse=True)
            
            for heading in heading_candidates:
                # Determine level based on font size ranking
                size_rank = unique_sizes.index(heading['font_size'])
                if size_rank == 0:
                    level = "H1"
                elif size_rank == 1:
                    level = "H2"
                else:
                    level = "H3"
                
                headings.append({
                    'level': level,
                    'text': heading['text'],
                    'page': heading['page']
                })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_headings = []
        for heading in headings:
            heading_key = heading['text']
            if heading_key not in seen:
                seen.add(heading_key)
                unique_headings.append(heading)
        
        return title, unique_headings[:10]  # Limit to 10 headings
    
    def generate_description(self, text, title, outline):
        """Generate a description by finding key content sections."""
        # Remove title and headings from text for description
        clean_text = text
        if title:
            clean_text = clean_text.replace(title, "")
        for heading in outline:
            clean_text = clean_text.replace(heading['text'], "")
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "No descriptive content found."
        
        # If we have few sentences, return them all
        if len(sentences) <= 3:
            return " ".join(sentences[:3]) + "."
        
        # Use AI model to find most representative sentences if available
        if self.model is not None:
            try:
                print("🤖 Using AI model for description generation...")
                embeddings = self.model.encode(sentences)
                
                # Cluster sentences and pick representatives
                n_clusters = min(3, len(sentences))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(embeddings)
                
                # Get one sentence from each cluster
                representative_sentences = []
                for i in range(n_clusters):
                    cluster_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
                    if cluster_sentences:
                        # Pick the sentence closest to cluster center
                        cluster_indices = [j for j in range(len(sentences)) if clusters[j] == i]
                        cluster_embeddings = embeddings[cluster_indices]
                        center = kmeans.cluster_centers_[i]
                        
                        distances = [np.linalg.norm(emb - center) for emb in cluster_embeddings]
                        best_idx = cluster_indices[np.argmin(distances)]
                        representative_sentences.append(sentences[best_idx])
                
                description = " ".join(representative_sentences) + "."
                print("✅ AI-generated description complete!")
                
            except Exception as e:
                print(f"⚠️ AI processing failed, using simple extraction: {e}")
                # Fallback: use first few sentences
                description = " ".join(sentences[:3]) + "."
        else:
            print("📝 Using simple text extraction (no AI model)...")
            # Fallback: use first few sentences
            description = " ".join(sentences[:3]) + "."
        
        return description
    
    def process_pdf(self, pdf_path, output_path):
        """Process a single PDF file and generate JSON output."""
        print(f"📄 Processing: {pdf_path}")
        
        try:
            # Extract text with structure
            text_blocks, full_text = self.extract_text_with_structure(pdf_path)
            
            if not text_blocks:
                result = {
                    "title": "Empty document",
                    "outline": [],
                    "description": "No content found in the PDF."
                }
            else:
                # Identify title and headings
                title, outline = self.identify_title_and_headings(text_blocks)
                
                # Generate description
                description = self.generate_description(full_text, title, outline)
                
                result = {
                    "title": title or "Untitled Document",
                    "outline": outline,
                    "description": description
                }
            
            # Write JSON output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            # Write error result
            error_result = {
                "title": "Processing Error",
                "outline": [],
                "description": f"Error processing PDF: {str(e)}"
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            return False

def main():
    """Main function to process all PDFs in the input directory."""
    print("🚀 Starting PDF processing...")
    start_time = time.time()
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in /app/input")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    success_count = 0
    for pdf_file in pdf_files:
        output_file = output_dir / f"{pdf_file.stem}.json"
        if processor.process_pdf(str(pdf_file), str(output_file)):
            success_count += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n🎉 Processing complete!")
    print(f"✅ Successfully processed: {success_count}/{len(pdf_files)} files")
    print(f"⏱️  Total time: {processing_time:.2f} seconds")
    
    if processing_time > 10:
        print("⚠️  Warning: Processing took longer than 10 seconds")

if __name__ == "__main__":
    main()