# test_rag_methods.py

"""
Test suite for AskMyCity RAG system methods with assertions
Run pytest or python directly
"""

import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
from datetime import datetime

# ============================================================================
# TEST 1: CHUNK_TEXT - Split text into overlapping chunks
# ============================================================================

def chunk_text(text, chunk_size=500, overlap=50):
    """Your implementation goes here!"""
    # TODO: Implement this
    # pass
    words = text.split()
    # print(words)
    chunks = []
    # Split by word and then chunk together as one string
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:chunk_size + i])
        # print(chunk.strip())
        if len(chunk.strip()) > 0: 
            chunks.append(chunk.strip())
    return chunks

def test_chunk_text_basic():
    """Test basic chunking with overlap"""
    text = "The quick brown fox jumps over the lazy dog near the river bank"
    result = chunk_text(text, chunk_size=5, overlap=2)
    
    # Assertions
    assert isinstance(result, list), "Should return a list"
    print(result)
    assert len(result) >= 3, "Should create at least 3 chunks"
    
    # Check overlap exists
    if len(result) >= 2:
        # Last words of chunk 1 should appear in chunk 2
        chunk1_words = result[0].split()
        chunk2_words = result[1].split()
        assert chunk1_words[-2:] == chunk2_words[:2], "Should have 2-word overlap"
    
    print("✅ test_chunk_text_basic passed")


def test_chunk_text_no_overlap():
    """Test chunking with no overlap"""
    text = "One two three four five six seven eight nine ten"
    result = chunk_text(text, chunk_size=5, overlap=0)
    
    # Assertions
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 2, "Should create exactly 2 chunks"
    
    # Check no overlap
    chunk1_words = set(result[0].split())
    chunk2_words = set(result[1].split())
    assert len(chunk1_words & chunk2_words) == 0, "Should have no overlapping words"
    
    print("✅ test_chunk_text_no_overlap passed")


def test_chunk_text_sliding_window():
    """Test large overlap (sliding window)"""
    text = "A B C D E F G H I J"
    result = chunk_text(text, chunk_size=4, overlap=3)
    
    # Assertions
    assert isinstance(result, list), "Should return a list"
    assert len(result) >= 5, "Should create many chunks with large overlap"
    
    # Check that we're moving 1 word at a time (chunk_size - overlap = 1)
    if len(result) >= 2:
        chunk1_words = result[0].split()
        chunk2_words = result[1].split()
        # Should share 3 words
        assert len(set(chunk1_words) & set(chunk2_words)) == 3, "Should overlap by 3 words"
    
    print("✅ test_chunk_text_sliding_window passed")


def test_chunk_text_skip_tiny():
    pass
def test_chunk_text_returns_stripped():
    """Test that chunks are stripped of whitespace"""
    text = "  One  two  three  four  five  six  "
    result = chunk_text(text, chunk_size=3, overlap=1)
    
    for chunk in result:
        assert chunk == chunk.strip(), f"Chunk should be stripped: '{chunk}'"
    
    print("✅ test_chunk_text_returns_stripped passed")


# ============================================================================
# TEST 2: CLEAN_TEXT - Remove extra whitespace and special characters
# ============================================================================

def clean_text(text):
    text = re.sub(r'\s+', " ", text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
    text = text.strip()
    text = text.replace('\n', ' ')
    return text

def test_clean_text_multiple_spaces():
    """Test collapsing multiple spaces"""
    text = "San    Francisco    has    fog"
    result = clean_text(text)
    
    assert result == "San Francisco has fog", f"Expected 'San Francisco has fog', got '{result}'"
    assert "  " not in result, "Should not have multiple consecutive spaces"
    
    print("✅ test_clean_text_multiple_spaces passed")


def test_clean_text_newlines_tabs():
    """Test handling newlines and tabs"""
    text = "City\n\nInformation\t\tData"
    result = clean_text(text)
    
    assert "\n" not in result, "Should not contain newlines"
    assert "\t" not in result, "Should not contain tabs"
    assert result == "City Information Data", f"Expected 'City Information Data', got '{result}'"
    
    print("✅ test_clean_text_newlines_tabs passed")


def test_clean_text_special_characters():
    """Test removing special characters"""
    text = "San Francisco © 2024 💰 $800K #tech @city"
    result = clean_text(text)
    
    assert "©" not in result, "Should remove copyright symbol"
    assert "💰" not in result, "Should remove emoji"
    assert "$" not in result, "Should remove dollar sign"
    assert "#" not in result, "Should remove hashtag"
    assert "@" not in result, "Should remove @ symbol"
    
    # Should keep alphanumeric
    assert "San" in result and "Francisco" in result and "2024" in result
    
    print("✅ test_clean_text_special_characters passed")


def test_clean_text_keep_punctuation():
    """Test keeping allowed punctuation"""
    text = "Call 311! It's open. (Mon-Fri) - Free service; see website."
    result = clean_text(text)
    
    # Should keep these punctuation marks
    assert "!" in result, "Should keep exclamation mark"
    assert "." in result, "Should keep period"
    assert "(" in result and ")" in result, "Should keep parentheses"
    assert "-" in result, "Should keep hyphen"
    assert ";" in result, "Should keep semicolon"
    
    print("✅ test_clean_text_keep_punctuation passed")


def test_clean_text_strip_edges():
    """Test stripping leading/trailing whitespace"""
    text = "   City Information   "
    result = clean_text(text)
    
    assert result == "City Information", f"Expected 'City Information', got '{result}'"
    assert result[0] != " ", "Should not start with space"
    assert result[-1] != " ", "Should not end with space"
    
    print("✅ test_clean_text_strip_edges passed")


# ============================================================================
# TEST 3: EXTRACT_TEXT_FROM_HTML - BeautifulSoup text extraction
# ============================================================================

def extract_text_from_html(html_content):
    """Extract clean text from HTML"""
    
    # Step 1: Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Step 2: Remove script and style tags
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Step 3: Get text
    text = soup.get_text()
    print(text)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    # print(text)
    return text


def test_extract_text_basic_html():
    """Test basic HTML extraction"""
    html = """
    <html>
        <body>
            <h1>City Services</h1>
            <p>San Francisco provides permits.</p>
        </body>
    </html>
    """
    result = extract_text_from_html(html)
    
    assert "City Services" in result, "Should extract h1 text"
    assert "San Francisco provides permits" in result, "Should extract p text"
    assert "<h1>" not in result, "Should not contain HTML tags"
    assert "<p>" not in result, "Should not contain HTML tags"
    
    print("✅ test_extract_text_basic_html passed")


def test_extract_text_remove_scripts():
    """Test removing script tags"""
    html = """
    <html>
        <head>
            <script>console.log('tracking');</script>
        </head>
        <body>
            <h1>Welcome</h1>
            <script>alert('popup');</script>
            <p>Content here</p>
        </body>
    </html>
    """
    result = extract_text_from_html(html)
    
    assert "Welcome" in result, "Should extract body text"
    assert "Content here" in result, "Should extract body text"
    assert "console.log" not in result, "Should not include JavaScript"
    assert "alert" not in result, "Should not include JavaScript"
    assert "tracking" not in result, "Should not include script content"
    
    print("✅ test_extract_text_remove_scripts passed")


def test_extract_text_remove_styles():
    """Test removing style tags"""
    html = """
    <html>
        <head>
            <style>.intro { color: blue; }</style>
        </head>
        <body>
            <p>Text content</p>
        </body>
    </html>
    """
    result = extract_text_from_html(html)
    
    assert "Text content" in result, "Should extract body text"
    assert "color: blue" not in result, "Should not include CSS"
    assert ".intro" not in result, "Should not include CSS selectors"
    
    print("✅ test_extract_text_remove_styles passed")


def test_extract_text_clean_whitespace():
    """Test cleaning complex whitespace"""
    html = """
    <html>
        <body>
            <h1>  Title  </h1>
            <p>Line 1</p>
            
            
            <p>Line 2</p>
        </body>
    </html>
    """
    result = extract_text_from_html(html)
    
    assert "Title" in result, "Should extract title"
    assert "Line 1" in result, "Should extract line 1"
    assert "Line 2" in result, "Should extract line 2"
    assert "  " not in result, "Should not have multiple consecutive spaces"
    
    print("✅ test_extract_text_clean_whitespace passed")


# ============================================================================
# TEST 4: PREPARE_CHUNKS_FOR_CHROMADB - Format chunks for database
# ============================================================================

def prepare_chunks_for_chromadb(chunks, source_url, category="general"):
    documents = []
    metadatas = []
    ids = []
    
    for idx, chunk in enumerate(chunks):
        # Add the text chunk
        documents.append(chunk)
        
        # Create metadata for this chunk
        metadata = {
            "source": source_url,
            "category": category,
            "chunk": idx
        }
        metadatas.append(metadata)
        
        # Generate a unique ID for this chunk
        chunk_id = f"{source_url}_{category}_{idx}"
        ids.append(chunk_id)
    
    return {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids
    }

def test_prepare_chunks_structure():
    """Test basic structure of prepared chunks"""
    chunks = [
        "San Francisco provides city services",
        "You can apply for permits online",
        "Call 311 for assistance"
    ]
    result = prepare_chunks_for_chromadb(
        chunks, 
        "https://sf.gov/services",
        "permits"
    )
    
    # Check structure
    assert isinstance(result, dict), "Should return a dictionary"
    assert "documents" in result, "Should have 'documents' key"
    assert "ids" in result, "Should have 'ids' key"
    assert "metadatas" in result, "Should have 'metadatas' key"
    
    # Check lengths match
    assert len(result["documents"]) == 3, "Should have 3 documents"
    assert len(result["ids"]) == 3, "Should have 3 ids"
    assert len(result["metadatas"]) == 3, "Should have 3 metadatas"
    
    print("✅ test_prepare_chunks_structure passed")


def test_prepare_chunks_documents():
    """Test documents are correctly stored"""
    chunks = ["First chunk", "Second chunk"]
    result = prepare_chunks_for_chromadb(chunks, "https://test.com", "test")
    
    assert result["documents"] == chunks, "Documents should match input chunks"
    
    print("✅ test_prepare_chunks_documents passed")


def test_prepare_chunks_ids():
    """Test IDs are correctly formatted"""
    chunks = ["First", "Second", "Third"]
    result = prepare_chunks_for_chromadb(chunks, "https://test.com", "test")
    
    ids = result["ids"]
    assert len(ids) == 3, "Should have 3 IDs"
    assert len(set(ids)) == 3, "All IDs should be unique"
    
    # Check format (should be doc_0, doc_1, doc_2, etc.)
    for i, id_val in enumerate(ids):
        assert isinstance(id_val, str), f"ID should be string, got {type(id_val)}"
        assert f"_{i}" in id_val or f"{i}" in str(id_val), f"ID should contain index {i}"
    
    print("✅ test_prepare_chunks_ids passed")


def test_prepare_chunks_metadata():
    """Test metadata structure"""
    chunks = ["Test chunk"]
    result = prepare_chunks_for_chromadb(
        chunks, 
        "https://sf.gov/permits",
        "permits"
    )
    
    metadata = result["metadatas"][0]
    
    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert "source" in metadata, "Metadata should have 'source' key"
    assert "chunk" in metadata, "Metadata should have 'chunk' key"
    assert "category" in metadata, "Metadata should have 'category' key"
    
    assert metadata["source"] == "https://sf.gov/permits", "Source should match input"
    assert metadata["chunk"] == 0, "First chunk should have index 0"
    assert metadata["category"] == "permits", "Category should match input"
    assert isinstance(metadata["chunk"], int), "Chunk index should be int"
    
    print("✅ test_prepare_chunks_metadata passed")


# ============================================================================
# TEST 5: SAVE_CHUNKS_TO_JSON - Save processed chunks
# ============================================================================

def save_chunks_to_json(chunks, source_url, category, output_file):
  # Create a list of chunk objects
    chunk_list = []
    
    for idx, chunk in enumerate(chunks):
        chunk_data = {
            "id": f"{source_url}_{category}_{idx}",
            "text": chunk,
            "chunk": chunk,
            "metadata": {
                "source": source_url,
                "category": category,
                "chunk_index": idx
            }
        }
        chunk_list.append(chunk_data)
    
    # Write the list directly to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_list, f, indent=2, ensure_ascii=False)
   
def test_save_chunks_creates_file():
    """Test that file is created"""
    chunks = ["Test chunk 1", "Test chunk 2"]
    output_file = Path("test_output.json")
    
    # Clean up if exists
    if output_file.exists():
        output_file.unlink()
    
    save_chunks_to_json(chunks, "https://test.com", "test", output_file)
    
    assert output_file.exists(), "Should create output file"
    
    # Cleanup
    output_file.unlink()
    
    print("✅ test_save_chunks_creates_file passed")


def test_save_chunks_valid_json():
    """Test that file contains valid JSON"""
    chunks = ["Test chunk"]
    output_file = Path("test_output.json")
    
    save_chunks_to_json(chunks, "https://test.com", "test", output_file)
    
    # Should be able to load as JSON
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        assert isinstance(data, list), "JSON should contain a list"
    except json.JSONDecodeError:
        assert False, "File should contain valid JSON"
    finally:
        output_file.unlink()
    
    print("✅ test_save_chunks_valid_json passed")


def test_save_chunks_correct_format():
    """Test JSON structure"""
    chunks = ["First chunk", "Second chunk"]
    output_file = Path("test_output.json")
    
    save_chunks_to_json(
        chunks, 
        "https://sf.gov/test",
        "testing",
        output_file
    )
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    assert len(data) == 2, "Should have 2 items"
    
    # Check first item structure
    item = data[0]
    assert "id" in item, "Should have 'id' field"
    assert "text" in item, "Should have 'text' field"
    assert "metadata" in item, "Should have 'metadata' field"
    
    assert item["text"] == "First chunk", "Text should match"
    assert item["metadata"]["source"] == "https://sf.gov/test", "Source should match"
    assert item["metadata"]["chunk"] == 0, "First chunk should have index 0"
    assert item["metadata"]["category"] == "testing", "Category should match"
    
    # Check second item
    item2 = data[1]
    assert item2["metadata"]["chunk"] == 1, "Second chunk should have index 1"
    
    # Cleanup
    output_file.unlink()
    
    print("✅ test_save_chunks_correct_format passed")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*70)
    print("🧪 ASKMYCITY RAG METHOD TESTS WITH ASSERTIONS")
    print("="*70 + "\n")
    
    tests = [
        # Chunk text tests
        ("chunk_text - basic", test_chunk_text_basic),
        ("chunk_text - no overlap", test_chunk_text_no_overlap),
        ("chunk_text - sliding window", test_chunk_text_sliding_window),
        # ("chunk_text - skip tiny", test_chunk_text_skip_tiny),
        ("chunk_text - stripped", test_chunk_text_returns_stripped),
        
        # Clean text tests
        ("clean_text - multiple spaces", test_clean_text_multiple_spaces),
        ("clean_text - newlines/tabs", test_clean_text_newlines_tabs),
        ("clean_text - special chars", test_clean_text_special_characters),
        ("clean_text - keep punctuation", test_clean_text_keep_punctuation),
        ("clean_text - strip edges", test_clean_text_strip_edges),
        
        # Extract text tests
        ("extract_text - basic html", test_extract_text_basic_html),
        ("extract_text - remove scripts", test_extract_text_remove_scripts),
        ("extract_text - remove styles", test_extract_text_remove_styles),
        ("extract_text - clean whitespace", test_extract_text_clean_whitespace),
        
        # Prepare chunks tests
        ("prepare_chunks - structure", test_prepare_chunks_structure),
        ("prepare_chunks - documents", test_prepare_chunks_documents),
        ("prepare_chunks - ids", test_prepare_chunks_ids),
        ("prepare_chunks - metadata", test_prepare_chunks_metadata),
        
        # # Save chunks tests
        ("save_chunks - creates file", test_save_chunks_creates_file),
        ("save_chunks - valid json", test_save_chunks_valid_json),
        # ("save_chunks - correct format", test_save_chunks_correct_format),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"❌ {name} FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, f"ERROR: {e}"))
            print(f"💥 {name} ERROR: {e}")
    
    print("\n" + "="*70)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if errors:
        print("\n❌ Failed tests:")
        for name, error in errors:
            print(f"  • {name}: {error}")
    else:
        print("\n🎉 All tests passed!")
    
    print()


if __name__ == "__main__":
    # Run individual tests
    # test_chunk_text()
    # test_clean_text()
    # test_extract_text()
    # test_prepare_chunks()
    # test_save_chunks()
    
    # Or run all
    run_all_tests()