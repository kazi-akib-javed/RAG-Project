import os
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.document_loader import load_pdf, load_all_pdfs


def test_load_pdf_file_not_found():
    """Should raise FileNotFoundError for missing file"""
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent/path/file.pdf")


def test_load_pdf_success(tmp_path):
    """Should load PDF and return pages"""
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"fake pdf content")

    mock_pages = [MagicMock(page_content="page 1 content")]

    with patch("src.ingestion.document_loader.PyPDFLoader") as mock_loader:
        mock_loader.return_value.load.return_value = mock_pages
        pages = load_pdf(str(fake_pdf))

    assert len(pages) == 1
    assert pages[0].page_content == "page 1 content"


def test_load_all_pdfs_empty_folder(tmp_path):
    """Should return empty list if no PDFs in folder"""
    result = load_all_pdfs(str(tmp_path))
    assert result == []


def test_load_all_pdfs_folder_not_found():
    """Should raise FileNotFoundError for missing folder"""
    with pytest.raises(FileNotFoundError):
        load_all_pdfs("nonexistent/folder")


def test_load_all_pdfs_success(tmp_path):
    """Should load all PDFs from folder"""
    pdf1 = tmp_path / "file1.pdf"
    pdf2 = tmp_path / "file2.pdf"
    pdf1.write_bytes(b"pdf1")
    pdf2.write_bytes(b"pdf2")

    mock_pages = [MagicMock(page_content="content")]

    with patch("src.ingestion.document_loader.PyPDFLoader") as mock_loader:
        mock_loader.return_value.load.return_value = mock_pages
        result = load_all_pdfs(str(tmp_path))

    assert len(result) == 2