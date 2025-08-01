# PDF Heading Detector

A Docker-based application that intelligently detects and extracts headings from PDF documents using advanced typography analysis and smart text clipping.

## Features

- **Smart Heading Detection**: Uses PyMuPDF to analyze font sizes, formatting, and typography
- **Intelligent Text Clipping**: Automatically clips heading text at font changes and formatting boundaries
- **Multi-level Support**: Detects H1, H2, H3, and H4 heading levels based on font hierarchy
- **Title Extraction**: Identifies document titles with center alignment and size analysis
- **JSON Output**: Generates structured JSON files with heading information
- **False Positive Filtering**: Removes common false positives like page numbers and copyright notices

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t pdf-heading-detector-test .
```

### 2. Process PDF Files

```bash
# Process all PDF files in current directory
docker run --rm -v "${PWD}:/app" pdf-heading-detector-test
```

## Usage Examples

### Process Current Directory
```bash
# Mount current directory and run detector
docker run --rm -v "${PWD}:/app" pdf-heading-detector-test
```

### Process Specific Directory
```bash
# Create directory for PDF files
mkdir pdf_files
cp your_document.pdf pdf_files/

# Run detector on specific directory
docker run --rm -v "${PWD}/pdf_files:/app" pdf-heading-detector-test
```

### Interactive Mode
```bash
# Run container interactively for debugging
docker run -it --rm -v "${PWD}:/app" pdf-heading-detector-test bash
```

## Output

The application generates JSON output files with the naming pattern:
- `filename_smart_detected_clipped.json`

### Output Format

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Sub Heading",
      "page": 2
    }
  ]
}
```

## How It Works

### Heading Detection Algorithm

1. **Typography Analysis**: Analyzes document font sizes to establish body text baseline
2. **Format Analysis**: Examines bold, italic, and underline formatting
3. **Alignment Check**: Verifies proper left or center alignment for headings
4. **Spacing Analysis**: Considers spacing before and after text blocks
5. **Content Filtering**: Removes false positives like page numbers and URLs
6. **Hierarchy Classification**: Assigns heading levels based on font size ratios

### Smart Clipping Features

- **Font Change Detection**: Clips text where font size changes significantly
- **Format Boundary Detection**: Stops at bold/underline formatting changes
- **Word Boundary Preservation**: Ensures complete words are maintained
- **Conservative Approach**: Uses shortest valid clipped version

## File Structure

```
project/
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── requirements.txt           # Python dependencies
├── heading_detector.py       # Main application
├── README.md                 # This file
├── .dockerignore             # Docker ignore patterns
├── file01.pdf               # Test PDF files
├── file02.pdf
├── file03.pdf
├── file04.pdf
└── file05.pdf
```

## Requirements

### System Requirements
- Docker
- PDF files to process

### Container Dependencies
- Python 3.9
- PyMuPDF (fitz)
- Build tools and system libraries for PDF processing

## Troubleshooting

### Common Issues

**Permission Errors**
```bash
# Ensure proper permissions on host directory
chmod 755 /path/to/your/pdf/directory
```

**PDF Processing Errors**
- Verify PDF files are not corrupted
- Check PDF files are readable and not password-protected
- Ensure sufficient disk space for output files

**Docker Build Issues**
```bash
# Clean build if needed
docker build --no-cache -t pdf-heading-detector-test .
```

## Development

### Modifying the Application

1. Edit `heading_detector.py` with your changes
2. Rebuild the Docker image:
   ```bash
   docker build -t pdf-heading-detector-test .
   ```
3. Test your changes:
   ```bash
   docker run --rm -v "${PWD}:/app" pdf-heading-detector-test
   ```

### Key Configuration Options

In `heading_detector.py`, you can adjust:
- `min_score`: Minimum confidence score for heading detection (default: 25)
- `heading_indicators`: Patterns and keywords for heading identification
- Font size thresholds for different heading levels

## Example Results

```
=== Processing file02.pdf ===
Title: Overview Foundation Level Extensions...
Found 13 headings
Body font size: 10.0
  H1: International Software Testing Qualifications Board
  H1: Revision History
  H1: Acknowledgements
  H2: 2.3 Learning Objectives
  H2: 2.1 Intended Audience
```

## License

This project is provided as-is for educational and development purposes.

## Contributing

1. Fork the repository
2. Make your changes
3. Test with sample PDF files
4. Submit a pull request

---

**Note**: This application is designed for processing structured documents with clear typography. Results may vary depending on PDF quality and formatting consistency.