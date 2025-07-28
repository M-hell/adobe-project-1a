# 📄 Offline PDF Structure Extractor

A fully offline Dockerized solution that extracts structured information (titles, headings, descriptions) from PDF files using AI, with no internet required at runtime.

## 🎯 What This Project Does

When you run this Docker container, it will:
- Process all `.pdf` files in the `input/` folder
- Extract structured content using AI (titles, headings, descriptions)
- Generate JSON files with the same name as each PDF in the `output/` folder
- Complete processing in ≤10 seconds for PDFs up to 50 pages
- Work completely offline (no internet needed)

**Example:**
```
input/report.pdf → output/report.json
```

## 📋 Requirements Met

✅ **Fully offline** - No internet required at runtime  
✅ **CPU-only** - Runs on amd64 architecture  
✅ **Resource efficient** - Works with 8 CPUs and 16GB RAM  
✅ **Small model** - Uses ~90MB AI model (well under 200MB limit)  
✅ **Fast processing** - Completes in under 10 seconds  
✅ **Dockerized** - Complete container solution  


### Step 3: Build the Docker Image

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

**What happens during build:**
- Downloads Python dependencies (~200MB)
- Downloads AI model (~90MB) and caches it
- Sets up processing environment
- **Total build time:** 3-5 minutes

### Step 4: Run the Container

1. **Place PDF files in input folder:**
```bash
cp your-document.pdf ./input/
```

2. **Run the processing:**
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

3. **Check results:**
```bash
ls output/
cat output/your-document.json
```

## ⚡ Performance Specifications

- **Processing Speed:** <10 seconds for 50-page PDFs
- **Model Size:** ~90MB (well under 200MB limit)
- **Memory Usage:** <4GB RAM typical
- **CPU Usage:** Utilizes all available cores
- **Offline Operation:** No internet required after build

## 🔧 Technical Details

### AI Model Used
- **Model:** `all-MiniLM-L6-v2` from Sentence Transformers
- **Size:** ~90MB
- **Purpose:** Text embeddings for content understanding
- **Offline:** Completely cached during Docker build

### PDF Processing
- **Library:** PyMuPDF (fast and reliable)
- **Structure Detection:** Font size and formatting analysis
- **Content Extraction:** Preserves text hierarchy

### Output Generation
- **Format:** Clean JSON with UTF-8 encoding
- **Structure:** Title, headings array, description string
- **Error Handling:** Graceful fallbacks for problematic PDFs

## 🐛 Troubleshooting

### Build Issues
```bash
# If build fails, try with no cache
docker build --no-cache --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

### Runtime Issues
```bash
# Check if input/output directories exist
ls -la input/ output/

# View container logs
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output mysolutionname:somerandomidentifier
```