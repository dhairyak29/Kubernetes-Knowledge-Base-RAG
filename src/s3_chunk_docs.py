# """
# Step 3: Chunk
# Splits cleaned documents into retrievable chunks using markdown structure
# (headers) as the primary split point, with a size cap and sentence-aware
# sub-splitting as a fallback for oversized sections.
 
# Usage:
#     python chunk_docs.py /path/to/concepts
#     python chunk_docs.py /path/to/concepts --file cluster-networking.md
# """
 
# import re
# import sys
# from dataclasses import dataclass, field
 
# from transformers import AutoTokenizer
 
# from s1_load_docs import load_markdown_docs
# from s2_clean_docs import clean_all

# # --- Config -----------------------------------------------------------
 
# # Sized against the embedding model's own limit, not the LLM's context window.
# # all-MiniLM-L6-v2 truncates silently at 256 tokens, so we cap comfortably
# # below that to leave room for the header we re-attach to every chunk.
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MAX_CHUNK_TOKENS = 220
# OVERLAP_TOKENS = 30
# MIN_CHUNK_TOKENS = 15

# _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

# def count_tokens(text: str) -> int:
#     # add_special_tokens=False so we're counting raw content tokens,
#     # matching how we compare against MAX_CHUNK_TOKENS below
#     return len(_tokenizer.encode(text, add_special_tokens=False))

# @dataclass
# class Chunk:
#     chunk_id: str
#     doc_title: str
#     source_file: str
#     section_header: str
#     text: str
#     token_count: int = field(init=False)

#     def __post_init__(self):
#         self.token_count = count_tokens(self.text)


# # --- Step 1: split by markdown headers --------------------------------

# def split_by_headers(clean_text: str) -> list[tuple[str, str]]:
#     """
#     Splits markdown text into (header, section_body) pairs using ## and ###
#     as split points. Text before the first header is kept as an "Introduction"
#     section so nothing is lost.
#     """

#     header_pattern = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)
#     matches = list(header_pattern.finditer(clean_text))

#     if not matches:
#         return [("Introduction", clean_text.strip())]

#     sections = []

#     #Text before the first header
#     intro = clean_text[: matches[0].start()].strip()

#     if intro:
#         sections.append(("Introduction", intro))

#     for i, match in enumerate(matches):
#         header_text = match.group(2).strip()
#         start = match.end()
#         end = matches[i+1].start() if i+1 < len(matches) else len(clean_text)
#         body = clean_text[start:end].strip()
#         sections.append((header_text, body))
#     return sections


# # --- Step 2: sub-split oversized sections ------------------------------
# def split_into_sentences(text: str) -> list[str]:
#     """Naive sentence splitter — good enough for prose docs, avoids a heavy NLP dependency."""
#     # Split on sentence-ending punctuation followed by whitespace + capital letter,
#     # but keep code blocks intact by not splitting inside triple-backticks.
#     parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", text)
#     return [p.strip() for p in parts if p.strip()]


# def subsplit_section(header:str, body:str) -> list[str]:
#     """
#     If a section's token count exceeds MAX_CHUNK_TOKENS, split it into
#     multiple overlapping sub-chunks along sentence boundaries.
#     Returns a list of text blocks (header re-attached to each for context).
#     """

#     full_text = f"## {header}\n\n{body}" if header != "Introduction" else body

#     if count_tokens(full_text) <= MAX_CHUNK_TOKENS:
#         return [full_text]

#     sentences = split_into_sentences(body)
#     sub_chunks = []
#     current = []
#     current_tokens = 0

#     for sentence in sentences:
#         sentence_tokens = count_tokens(sentence)

#         if current_tokens + sentence_tokens > MAX_CHUNK_TOKENS and current:
#             chunk_text = f"## {header}\n\n" + " ".join(current)
#             sub_chunks.append(chunk_text)

#             # carry the last ~OVERLAP_TOKENS worth of sentences forward
#             overlap_sentences = []
#             overlap_tokens = 0
#             for s in reversed(current):
#                 t = count_tokens(s)
#                 if overlap_tokens + t > OVERLAP_TOKENS:
#                     break
#                 overlap_sentences.insert(0,s)
#                 overlap_tokens += t

#             current = overlap_sentences
#             current_tokens = overlap_tokens

#             current.append(sentence)
#             current_tokens += sentence_tokens

#         if current:
#             chunk_text = f"## {header}\n\n" + " ".join(current)
#             sub_chunks.append(chunk_text)

#         return sub_chunks
   


# # --- Step 3: orchestrate per document ----------------------------------


# def chunk_document(doc: dict) -> list[Chunk]:
#     """Runs the full chunking pipeline on one cleaned document."""

#     sections = split_by_headers(doc["clean_text"])
#     chunks = []
#     chunk_index = 0

#     for header, body in sections:
#         if not body.strip():
#             continue

#         for text_block in subsplit_section(header, body):
#             token_count = count_tokens(text_block)
#             if token_count < MAX_CHUNK_TOKENS:
#                 continue
#             chunk_index += 1
#             chunks.append(
#                 Chunk(
#                     chunk_id = f"{doc['filename']}::{chunk_index}",
#                     doc_title = doc["title"],
#                     source_file = doc["filename"],
#                     section_header= header,
#                     text = text_block
#                 )
#             )
#     return chunks

# def chunk_all(cleaned_docs : list[dict]) -> list[Chunk]:
#     all_chunks = [] 
#     for doc in cleaned_docs:
#         all_chunks.extend(chunk_document(doc))
#     return all_chunks

# # --- Reporting -----------------------------------------------------------

# def print_summary(chunks: list[Chunk]) -> None:
#     if not chunks:
#         print("No chunks produced.")
#         return
 
#     token_counts = [c.token_count for c in chunks]
#     print(f"Produced {len(chunks)} chunks\n")
#     print(f"Avg tokens/chunk: {sum(token_counts) / len(token_counts):.0f}")
#     print(f"Min tokens: {min(token_counts)}  Max tokens: {max(token_counts)}")
#     over_cap = sum(1 for t in token_counts if t > MAX_CHUNK_TOKENS + OVERLAP_TOKENS)
#     print(f"Chunks over cap ({MAX_CHUNK_TOKENS}): {over_cap}")
 
#     print("\nSample chunk:")
#     sample = chunks[0]
#     print(f"  ID: {sample.chunk_id}")
#     print(f"  Doc title: {sample.doc_title}")
#     print(f"  Section: {sample.section_header}")
#     print(f"  Tokens: {sample.token_count}")
#     print(f"  Text:\n{sample.text[:400]}")
            


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage:")
#         print("  python chunk_docs.py /path/to/concepts")
#         print("  python chunk_docs.py /path/to/concepts --file cluster-networking.md")
#         sys.exit(1)

#     folder = sys.argv[1]
#     raw_docs = load_markdown_docs(folder)
#     cleaned_docs = clean_all(raw_docs)

#     if "--file" in sys.argv:
#         target = sys.argv[sys.argv.index("--file") + 1]
#         cleaned_docs = [d for d in cleaned_docs if d["filename"] == target]
#         if not cleaned_docs:
#             print(f"No file named '{target}' found.")
#             sys.exit(1)
 
#     chunks = chunk_all(cleaned_docs)
 
#     if "--file" in sys.argv:
#         for c in chunks:
#             print(f"--- {c.chunk_id} ({c.token_count} tokens) — section: {c.section_header} ---")
#             print(c.text)
#             print()
#     else:
#         print_summary(chunks)




# """
# Step 3: Chunk
# Splits cleaned documents into retrievable chunks using markdown structure
# (headers) as the primary split point, with a size cap and sentence-aware
# sub-splitting as a fallback for oversized sections.

# Usage:
#     python chunk_docs.py /path/to/concepts
#     python chunk_docs.py /path/to/concepts --file cluster-networking.md
# """

# import re
# import sys
# from dataclasses import dataclass, field

# from transformers import AutoTokenizer

# from s1_load_docs import load_markdown_docs
# from s2_clean_docs import clean_all

# # --- Config -----------------------------------------------------------

# # Sized against the embedding model's own limit, not the LLM's context window.
# # all-MiniLM-L6-v2 truncates silently at 256 tokens, so we cap comfortably
# # below that to leave room for the header we re-attach to every chunk.
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MAX_CHUNK_TOKENS = 220
# OVERLAP_TOKENS = 30
# MIN_CHUNK_TOKENS = 15

# _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


# def count_tokens(text: str) -> int:
#     # add_special_tokens=False so we're counting raw content tokens,
#     # matching how we compare against MAX_CHUNK_TOKENS below
#     return len(_tokenizer.encode(text, add_special_tokens=False))


# @dataclass
# class Chunk:
#     chunk_id: str
#     doc_title: str
#     source_file: str
#     section_header: str
#     text: str
#     token_count: int = field(init=False)

#     def __post_init__(self):
#         self.token_count = count_tokens(self.text)


# # --- Step 1: split by markdown headers --------------------------------

# def split_by_headers(clean_text: str) -> list[tuple[str, str]]:
#     """
#     Splits markdown text into (header, section_body) pairs using ## and ###
#     as split points. Text before the first header is kept as an "Introduction"
#     section so nothing is lost.
#     """
#     header_pattern = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)
#     matches = list(header_pattern.finditer(clean_text))

#     if not matches:
#         return [("Introduction", clean_text.strip())]

#     sections = []

#     # Text before the first header
#     intro = clean_text[: matches[0].start()].strip()
#     if intro:
#         sections.append(("Introduction", intro))

#     for i, match in enumerate(matches):
#         header_text = match.group(2).strip()
#         start = match.end()
#         end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
#         body = clean_text[start:end].strip()
#         sections.append((header_text, body))

#     return sections


# # --- Step 2: sub-split oversized sections ------------------------------

# def split_into_sentences(text: str) -> list[str]:
#     """Naive sentence splitter — good enough for prose docs, avoids a heavy NLP dependency."""
#     # Split on sentence-ending punctuation followed by whitespace + capital letter,
#     # but keep code blocks intact by not splitting inside triple-backticks.
#     parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", text)
#     return [p.strip() for p in parts if p.strip()]


# def subsplit_section(header: str, body: str) -> list[str]:
#     """
#     If a section's token count exceeds MAX_CHUNK_TOKENS, split it into
#     multiple overlapping sub-chunks along sentence boundaries.
#     Returns a list of text blocks (header re-attached to each for context).
#     """
#     full_text = f"## {header}\n\n{body}" if header != "Introduction" else body

#     if count_tokens(full_text) <= MAX_CHUNK_TOKENS:
#         return [full_text]

#     sentences = split_into_sentences(body)
#     sub_chunks = []
#     current = []
#     current_tokens = 0

#     for sentence in sentences:
#         sentence_tokens = count_tokens(sentence)

#         if current_tokens + sentence_tokens > MAX_CHUNK_TOKENS and current:
#             chunk_text = f"## {header}\n\n" + " ".join(current)
#             sub_chunks.append(chunk_text)

#             # carry the last ~OVERLAP_TOKENS worth of sentences forward
#             overlap_sentences = []
#             overlap_tokens = 0
#             for s in reversed(current):
#                 t = count_tokens(s)
#                 if overlap_tokens + t > OVERLAP_TOKENS:
#                     break
#                 overlap_sentences.insert(0, s)
#                 overlap_tokens += t
#             current = overlap_sentences
#             current_tokens = overlap_tokens

#         current.append(sentence)
#         current_tokens += sentence_tokens

#     if current:
#         chunk_text = f"## {header}\n\n" + " ".join(current)
#         sub_chunks.append(chunk_text)

#     return sub_chunks


# # --- Step 3: orchestrate per document ----------------------------------

# def chunk_document(doc: dict) -> list[Chunk]:
#     """Runs the full chunking pipeline on one cleaned document."""
#     sections = split_by_headers(doc["clean_text"])
#     chunks = []
#     chunk_index = 0

#     for header, body in sections:
#         if not body.strip():
#             continue

#         for text_block in subsplit_section(header, body):
#             token_count = count_tokens(text_block)
#             if token_count < MIN_CHUNK_TOKENS:
#                 continue  # discard near-empty chunks (e.g. header with no real content)

#             chunk_index += 1
#             chunks.append(
#                 Chunk(
#                     chunk_id=f"{doc['filename']}::{chunk_index}",
#                     doc_title=doc["title"],
#                     source_file=doc["filename"],
#                     section_header=header,
#                     text=text_block,
#                 )
#             )

#     return chunks


# def chunk_all(cleaned_docs: list[dict]) -> list[Chunk]:
#     all_chunks = []
#     zero_chunk_docs = []
#     for doc in cleaned_docs:
#         doc_chunks = chunk_document(doc)
#         if not doc_chunks:
#             zero_chunk_docs.append(doc["filename"])
#         all_chunks.extend(doc_chunks)

#     print(f"[debug] {len(cleaned_docs)} cleaned docs in -> {len(all_chunks)} chunks out")
#     print(f"[debug] {len(zero_chunk_docs)} docs produced ZERO chunks")
#     if zero_chunk_docs:
#         print(f"[debug] examples: {zero_chunk_docs[:10]}")

#     return all_chunks


# # --- Reporting -----------------------------------------------------------

# def print_summary(chunks: list[Chunk]) -> None:
#     if not chunks:
#         print("No chunks produced.")
#         return

#     token_counts = [c.token_count for c in chunks]
#     print(f"Produced {len(chunks)} chunks\n")
#     print(f"Avg tokens/chunk: {sum(token_counts) / len(token_counts):.0f}")
#     print(f"Min tokens: {min(token_counts)}  Max tokens: {max(token_counts)}")
#     over_cap = sum(1 for t in token_counts if t > MAX_CHUNK_TOKENS + OVERLAP_TOKENS)
#     print(f"Chunks over cap ({MAX_CHUNK_TOKENS}): {over_cap}")

#     print("\nSample chunk:")
#     sample = chunks[0]
#     print(f"  ID: {sample.chunk_id}")
#     print(f"  Doc title: {sample.doc_title}")
#     print(f"  Section: {sample.section_header}")
#     print(f"  Tokens: {sample.token_count}")
#     print(f"  Text:\n{sample.text[:400]}")


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage:")
#         print("  python chunk_docs.py /path/to/concepts")
#         print("  python chunk_docs.py /path/to/concepts --file cluster-networking.md")
#         sys.exit(1)

#     folder = sys.argv[1]
#     raw_docs = load_markdown_docs(folder)
#     print(f"[debug] loaded {len(raw_docs)} raw docs")
#     cleaned_docs = clean_all(raw_docs)
#     print(f"[debug] cleaned {len(cleaned_docs)} docs")

#     if "--file" in sys.argv:
#         target = sys.argv[sys.argv.index("--file") + 1]
#         cleaned_docs = [d for d in cleaned_docs if d["filename"] == target]
#         if not cleaned_docs:
#             print(f"No file named '{target}' found.")
#             sys.exit(1)

#     chunks = chunk_all(cleaned_docs)

#     if "--file" in sys.argv:
#         for c in chunks:
#             print(f"--- {c.chunk_id} ({c.token_count} tokens) — section: {c.section_header} ---")
#             print(c.text)
#             print()
#     else:
#         print_summary(chunks)







# """
# Step 3: Chunk
# Splits cleaned documents into retrievable chunks using markdown structure
# (headers) as the primary split point, with a size cap and sentence-aware
# sub-splitting as a fallback for oversized sections.

# Usage:
#     python chunk_docs.py /path/to/concepts
#     python chunk_docs.py /path/to/concepts --file cluster-networking.md
# """

# import re
# import sys
# from dataclasses import dataclass, field

# from transformers import AutoTokenizer

# from s1_load_docs import load_markdown_docs
# from s2_clean_docs import clean_all

# # --- Config -----------------------------------------------------------

# # Sized against the embedding model's own limit, not the LLM's context window.
# # all-MiniLM-L6-v2 truncates silently at 256 tokens, so we cap comfortably
# # below that to leave room for the header we re-attach to every chunk.
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MAX_CHUNK_TOKENS = 220
# OVERLAP_TOKENS = 30
# MIN_CHUNK_TOKENS = 15

# _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


# def count_tokens(text: str) -> int:
#     # add_special_tokens=False so we're counting raw content tokens,
#     # matching how we compare against MAX_CHUNK_TOKENS below
#     return len(_tokenizer.encode(text, add_special_tokens=False))


# @dataclass
# class Chunk:
#     chunk_id: str
#     doc_title: str
#     source_file: str
#     section_header: str
#     text: str
#     token_count: int = field(init=False)

#     def __post_init__(self):
#         self.token_count = count_tokens(self.text)


# # --- Step 1: split by markdown headers --------------------------------

# def split_by_headers(clean_text: str) -> list[tuple[str, str]]:
#     """
#     Splits markdown text into (header, section_body) pairs using ## and ###
#     as split points. Text before the first header is kept as an "Introduction"
#     section so nothing is lost.
#     """
#     header_pattern = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)
#     matches = list(header_pattern.finditer(clean_text))

#     if not matches:
#         return [("Introduction", clean_text.strip())]

#     sections = []

#     # Text before the first header
#     intro = clean_text[: matches[0].start()].strip()
#     if intro:
#         sections.append(("Introduction", intro))

#     for i, match in enumerate(matches):
#         header_text = match.group(2).strip()
#         start = match.end()
#         end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
#         body = clean_text[start:end].strip()
#         sections.append((header_text, body))

#     return sections


# # --- Step 2: sub-split oversized sections ------------------------------

# def split_into_sentences(text: str) -> list[str]:
#     """Naive sentence splitter — good enough for prose docs, avoids a heavy NLP dependency."""
#     # Split on sentence-ending punctuation followed by whitespace + capital letter,
#     # but keep code blocks intact by not splitting inside triple-backticks.
#     parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", text)
#     return [p.strip() for p in parts if p.strip()]


# def hard_split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
#     """
#     Last-resort splitter: encodes text to tokens and slices by raw token
#     windows. Used when a "sentence" (from the naive splitter) is itself
#     still too large — e.g. long bullet lists or code blocks with no
#     sentence-ending punctuation for the regex splitter to latch onto.
#     Guarantees no chunk piece exceeds max_tokens, regardless of content shape.
#     """
#     token_ids = _tokenizer.encode(text, add_special_tokens=False)
#     if len(token_ids) <= max_tokens:
#         return [text]

#     pieces = []
#     start = 0
#     step = max_tokens - overlap_tokens
#     while start < len(token_ids):
#         window = token_ids[start : start + max_tokens]
#         pieces.append(_tokenizer.decode(window))
#         start += step

#     return pieces


# def subsplit_section(header: str, body: str) -> list[str]:
#     """
#     If a section's token count exceeds MAX_CHUNK_TOKENS, split it into
#     multiple overlapping sub-chunks along sentence boundaries. Any piece
#     that's still oversized after sentence splitting (e.g. lists, code
#     blocks with no clean sentence breaks) falls back to a hard token-window
#     split, so no chunk can ever exceed the cap.
#     """
#     full_text = f"## {header}\n\n{body}" if header != "Introduction" else body

#     if count_tokens(full_text) <= MAX_CHUNK_TOKENS:
#         return [full_text]

#     raw_sentences = split_into_sentences(body)

#     # Safety net: break up any oversized "sentence" before the grouping pass
#     sentences = []
#     for s in raw_sentences:
#         if count_tokens(s) > MAX_CHUNK_TOKENS:
#             sentences.extend(hard_split_by_tokens(s, MAX_CHUNK_TOKENS, OVERLAP_TOKENS))
#         else:
#             sentences.append(s)

#     sub_chunks = []
#     current = []
#     current_tokens = 0

#     for sentence in sentences:
#         sentence_tokens = count_tokens(sentence)

#         if current_tokens + sentence_tokens > MAX_CHUNK_TOKENS and current:
#             chunk_text = f"## {header}\n\n" + " ".join(current)
#             sub_chunks.append(chunk_text)

#             # carry the last ~OVERLAP_TOKENS worth of sentences forward
#             overlap_sentences = []
#             overlap_tokens = 0
#             for s in reversed(current):
#                 t = count_tokens(s)
#                 if overlap_tokens + t > OVERLAP_TOKENS:
#                     break
#                 overlap_sentences.insert(0, s)
#                 overlap_tokens += t
#             current = overlap_sentences
#             current_tokens = overlap_tokens

#         current.append(sentence)
#         current_tokens += sentence_tokens

#     if current:
#         chunk_text = f"## {header}\n\n" + " ".join(current)
#         sub_chunks.append(chunk_text)

#     # Final safety pass: verify nothing slipped through oversized (e.g. the
#     # header + one hard-split piece could still tip over the cap slightly)
#     verified = []
#     for chunk_text in sub_chunks:
#         if count_tokens(chunk_text) > MAX_CHUNK_TOKENS:
#             verified.extend(hard_split_by_tokens(chunk_text, MAX_CHUNK_TOKENS, OVERLAP_TOKENS))
#         else:
#             verified.append(chunk_text)

#     return verified


# # --- Step 3: orchestrate per document ----------------------------------

# def chunk_document(doc: dict) -> list[Chunk]:
#     """Runs the full chunking pipeline on one cleaned document."""
#     sections = split_by_headers(doc["clean_text"])
#     chunks = []
#     chunk_index = 0

#     for header, body in sections:
#         if not body.strip():
#             continue

#         for text_block in subsplit_section(header, body):
#             token_count = count_tokens(text_block)
#             if token_count < MIN_CHUNK_TOKENS:
#                 continue  # discard near-empty chunks (e.g. header with no real content)

#             chunk_index += 1
#             chunks.append(
#                 Chunk(
#                     chunk_id=f"{doc['filename']}::{chunk_index}",
#                     doc_title=doc["title"],
#                     source_file=doc["filename"],
#                     section_header=header,
#                     text=text_block,
#                 )
#             )

#     return chunks


# def chunk_all(cleaned_docs: list[dict]) -> list[Chunk]:
#     all_chunks = []
#     zero_chunk_docs = []
#     for doc in cleaned_docs:
#         doc_chunks = chunk_document(doc)
#         if not doc_chunks:
#             zero_chunk_docs.append(doc["filename"])
#         all_chunks.extend(doc_chunks)

#     print(f"[debug] {len(cleaned_docs)} cleaned docs in -> {len(all_chunks)} chunks out")
#     print(f"[debug] {len(zero_chunk_docs)} docs produced ZERO chunks")
#     if zero_chunk_docs:
#         print(f"[debug] examples: {zero_chunk_docs[:10]}")

#     return all_chunks


# # --- Reporting -----------------------------------------------------------

# def print_summary(chunks: list[Chunk]) -> None:
#     if not chunks:
#         print("No chunks produced.")
#         return

#     token_counts = [c.token_count for c in chunks]
#     print(f"Produced {len(chunks)} chunks\n")
#     print(f"Avg tokens/chunk: {sum(token_counts) / len(token_counts):.0f}")
#     print(f"Min tokens: {min(token_counts)}  Max tokens: {max(token_counts)}")
#     over_cap = sum(1 for t in token_counts if t > MAX_CHUNK_TOKENS + OVERLAP_TOKENS)
#     print(f"Chunks over cap ({MAX_CHUNK_TOKENS}): {over_cap}")

#     print("\nSample chunk:")
#     sample = chunks[0]
#     print(f"  ID: {sample.chunk_id}")
#     print(f"  Doc title: {sample.doc_title}")
#     print(f"  Section: {sample.section_header}")
#     print(f"  Tokens: {sample.token_count}")
#     print(f"  Text:\n{sample.text[:400]}")


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage:")
#         print("  python chunk_docs.py /path/to/concepts")
#         print("  python chunk_docs.py /path/to/concepts --file cluster-networking.md")
#         sys.exit(1)

#     folder = sys.argv[1]
#     raw_docs = load_markdown_docs(folder)
#     print(f"[debug] loaded {len(raw_docs)} raw docs")
#     cleaned_docs = clean_all(raw_docs)
#     print(f"[debug] cleaned {len(cleaned_docs)} docs")

#     if "--file" in sys.argv:
#         target = sys.argv[sys.argv.index("--file") + 1]
#         cleaned_docs = [d for d in cleaned_docs if d["filename"] == target]
#         if not cleaned_docs:
#             print(f"No file named '{target}' found.")
#             sys.exit(1)

#     chunks = chunk_all(cleaned_docs)

#     if "--file" in sys.argv:
#         for c in chunks:
#             print(f"--- {c.chunk_id} ({c.token_count} tokens) — section: {c.section_header} ---")
#             print(c.text)
#             print()
#     else:
#         print_summary(chunks)








"""
Step 3: Chunk
Splits cleaned documents into retrievable chunks using markdown structure
(headers) as the primary split point, with a size cap and sentence-aware
sub-splitting as a fallback for oversized sections.

Usage:
    python chunk_docs.py /path/to/concepts
    python chunk_docs.py /path/to/concepts --file cluster-networking.md
"""

import re
import sys
from dataclasses import dataclass, field

from transformers import AutoTokenizer

from s1_load_docs import load_markdown_docs
from s2_clean_docs import clean_all

# --- Config -----------------------------------------------------------

# Sized against the embedding model's own limit, not the LLM's context window.
# all-MiniLM-L6-v2 truncates silently at 256 tokens, so we cap comfortably
# below that to leave room for the header we re-attach to every chunk.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_TOKENS = 220
OVERLAP_TOKENS = 30
MIN_CHUNK_TOKENS = 15

_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def count_tokens(text: str) -> int:
    # add_special_tokens=False so we're counting raw content tokens,
    # matching how we compare against MAX_CHUNK_TOKENS below
    return len(_tokenizer.encode(text, add_special_tokens=False))


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    source_file: str
    section_header: str
    text: str
    token_count: int = field(init=False)

    def __post_init__(self):
        self.token_count = count_tokens(self.text)


# --- Step 1: split by markdown headers --------------------------------

def split_by_headers(clean_text: str) -> list[tuple[str, str]]:
    """
    Splits markdown text into (header, section_body) pairs using ## and ###
    as split points. Text before the first header is kept as an "Introduction"
    section so nothing is lost.
    """
    header_pattern = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)
    matches = list(header_pattern.finditer(clean_text))

    if not matches:
        return [("Introduction", clean_text.strip())]

    sections = []

    # Text before the first header
    intro = clean_text[: matches[0].start()].strip()
    if intro:
        sections.append(("Introduction", intro))

    for i, match in enumerate(matches):
        header_text = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
        body = clean_text[start:end].strip()
        sections.append((header_text, body))

    return sections


# --- Step 2: sub-split oversized sections ------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for prose docs, avoids a heavy NLP dependency."""
    # Split on sentence-ending punctuation followed by whitespace + capital letter,
    # but keep code blocks intact by not splitting inside triple-backticks.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", text)
    return [p.strip() for p in parts if p.strip()]


def hard_split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Last-resort splitter: encodes text to tokens and slices by raw token
    windows. Used when a "sentence" (from the naive splitter) is itself
    still too large — e.g. long bullet lists or code blocks with no
    sentence-ending punctuation for the regex splitter to latch onto.
    Guarantees no chunk piece exceeds max_tokens, regardless of content shape.
    """
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    pieces = []
    start = 0
    step = max_tokens - overlap_tokens
    while start < len(token_ids):
        window = token_ids[start : start + max_tokens]
        pieces.append(_tokenizer.decode(window))
        start += step

    return pieces


def subsplit_section(header: str, body: str) -> list[str]:
    """
    If a section's token count exceeds MAX_CHUNK_TOKENS, split it into
    multiple overlapping sub-chunks along sentence boundaries. Any piece
    that's still oversized after sentence splitting (e.g. lists, code
    blocks with no clean sentence breaks) falls back to a hard token-window
    split, so no chunk can ever exceed the cap.
    """
    full_text = f"## {header}\n\n{body}" if header != "Introduction" else body

    if count_tokens(full_text) <= MAX_CHUNK_TOKENS:
        return [full_text]

    raw_sentences = split_into_sentences(body)

    # Safety net: break up any oversized "sentence" before the grouping pass
    sentences = []
    for s in raw_sentences:
        if count_tokens(s) > MAX_CHUNK_TOKENS:
            sentences.extend(hard_split_by_tokens(s, MAX_CHUNK_TOKENS, OVERLAP_TOKENS))
        else:
            sentences.append(s)

    sub_chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > MAX_CHUNK_TOKENS and current:
            chunk_text = f"## {header}\n\n" + " ".join(current)
            sub_chunks.append(chunk_text)

            # carry the last ~OVERLAP_TOKENS worth of sentences forward
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current):
                t = count_tokens(s)
                if overlap_tokens + t > OVERLAP_TOKENS:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += t
            current = overlap_sentences
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunk_text = f"## {header}\n\n" + " ".join(current)
        sub_chunks.append(chunk_text)

    # Final safety pass: verify nothing slipped through oversized (e.g. the
    # header + one hard-split piece could still tip over the cap slightly)
    verified = []
    for chunk_text in sub_chunks:
        if count_tokens(chunk_text) > MAX_CHUNK_TOKENS:
            verified.extend(hard_split_by_tokens(chunk_text, MAX_CHUNK_TOKENS, OVERLAP_TOKENS))
        else:
            verified.append(chunk_text)

    return verified


# --- Step 3: orchestrate per document ----------------------------------

def chunk_document(doc: dict) -> list[Chunk]:
    """Runs the full chunking pipeline on one cleaned document."""
    sections = split_by_headers(doc["clean_text"])
    chunks = []
    chunk_index = 0

    for header, body in sections:
        if not body.strip():
            continue

        for text_block in subsplit_section(header, body):
            token_count = count_tokens(text_block)
            if token_count < MIN_CHUNK_TOKENS:
                continue  # discard near-empty chunks (e.g. header with no real content)

            chunk_index += 1
            # Use the full path (not just filename) for chunk_id — Hugo docs
            # reuse filenames like "_index.md" across many subfolders, so
            # filename alone isn't unique. Path is. Sanitize separators so
            # the ID stays a clean single string.
            path_key = doc["path"].replace("\\", "/").replace(" ", "_")
            chunks.append(
                Chunk(
                    chunk_id=f"{path_key}::{chunk_index}",
                    doc_title=doc["title"],
                    source_file=doc["filename"],
                    section_header=header,
                    text=text_block,
                )
            )

    return chunks


def chunk_all(cleaned_docs: list[dict]) -> list[Chunk]:
    all_chunks = []
    zero_chunk_docs = []
    for doc in cleaned_docs:
        doc_chunks = chunk_document(doc)
        if not doc_chunks:
            zero_chunk_docs.append(doc["filename"])
        all_chunks.extend(doc_chunks)

    print(f"[debug] {len(cleaned_docs)} cleaned docs in -> {len(all_chunks)} chunks out")
    print(f"[debug] {len(zero_chunk_docs)} docs produced ZERO chunks")
    if zero_chunk_docs:
        print(f"[debug] examples: {zero_chunk_docs[:10]}")

    return all_chunks


# --- Reporting -----------------------------------------------------------

def print_summary(chunks: list[Chunk]) -> None:
    if not chunks:
        print("No chunks produced.")
        return

    token_counts = [c.token_count for c in chunks]
    print(f"Produced {len(chunks)} chunks\n")
    print(f"Avg tokens/chunk: {sum(token_counts) / len(token_counts):.0f}")
    print(f"Min tokens: {min(token_counts)}  Max tokens: {max(token_counts)}")
    over_cap = sum(1 for t in token_counts if t > MAX_CHUNK_TOKENS + OVERLAP_TOKENS)
    print(f"Chunks over cap ({MAX_CHUNK_TOKENS}): {over_cap}")

    print("\nSample chunk:")
    sample = chunks[0]
    print(f"  ID: {sample.chunk_id}")
    print(f"  Doc title: {sample.doc_title}")
    print(f"  Section: {sample.section_header}")
    print(f"  Tokens: {sample.token_count}")
    print(f"  Text:\n{sample.text[:400]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python chunk_docs.py /path/to/concepts")
        print("  python chunk_docs.py /path/to/concepts --file cluster-networking.md")
        sys.exit(1)

    folder = sys.argv[1]
    raw_docs = load_markdown_docs(folder)
    print(f"[debug] loaded {len(raw_docs)} raw docs")
    cleaned_docs = clean_all(raw_docs)
    print(f"[debug] cleaned {len(cleaned_docs)} docs")

    if "--file" in sys.argv:
        target = sys.argv[sys.argv.index("--file") + 1]
        cleaned_docs = [d for d in cleaned_docs if d["filename"] == target]
        if not cleaned_docs:
            print(f"No file named '{target}' found.")
            sys.exit(1)

    chunks = chunk_all(cleaned_docs)

    if "--file" in sys.argv:
        for c in chunks:
            print(f"--- {c.chunk_id} ({c.token_count} tokens) — section: {c.section_header} ---")
            print(c.text)
            print()
    else:
        print_summary(chunks)