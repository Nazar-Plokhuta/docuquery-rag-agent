"""Ingestion pipeline for the DocuQuery RAG Agent.

This package implements the text-processing stages that transform raw
documents into indexed, retrievable ``DocumentChunk`` objects:

- ``chunker``: Token-bounded sliding-window text segmentation via tiktoken.
- ``loaders``: Format-specific text extraction (plain text, Markdown, PDF).
- ``pipeline``: End-to-end orchestration — read → chunk → embed → index.
"""
