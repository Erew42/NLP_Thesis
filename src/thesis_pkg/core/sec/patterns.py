from __future__ import annotations

import re


FILENAME_PATTERN = re.compile(
    r"(\d{8})_"          # 1: Date (YYYYMMDD)
    r"([^_]+)_"          # 2: Type
    r"edgar_data_"
    r"(\d+)_"            # 3: CIK
    r"([\d-]+)"          # 4: Accession Number
    r"\.txt$",
    re.IGNORECASE,
)

CIK_HEADER_PATTERN = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)")
DATE_HEADER_PATTERN = re.compile(r"FILED AS OF DATE:\s*(\d{8})")
PERIOD_END_HEADER_PATTERN = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")
ACC_HEADER_PATTERN = re.compile(r"ACCESSION NUMBER:\s*([\d-]+)")

EDGAR_BLOCK_TAG_PATTERN = re.compile(
    r"(?is)<\s*(?P<tag>sec-header|header|filestats|file-stats|xml_chars|xml-chars)\b[^>]*>"
    r".*?<\s*/\s*(?P=tag)\s*>"
)
EDGAR_TRAILING_TAG_PATTERN = re.compile(
    r"(?is)^\s*<\s*(sec-header|header)\b[^>]*>.*",
)

TOC_MARKER_PATTERN = re.compile(
    # Be conservative: "INDEX" appears in many non-TOC contexts (e.g., "Exhibit Index"),
    # and false TOC masking can suppress real item headings.
    r"\btable\s+of\s+contents?\b|\btable\s+of\s+content\b",
    re.IGNORECASE,
)
TOC_HEADER_LINE_PATTERN = re.compile(r"^\s*table\s+of\s+contents?\s*$", re.IGNORECASE)
TOC_INDEX_MARKER_PATTERN = re.compile(
    r"\btable\s+of\s+contents?\b|\btable\s+of\s+content\b|\bindex\b|\b(?:form\s+)?10-?k\s+summary\b",
    re.IGNORECASE,
)
SUMMARY_MARKER_PATTERN = re.compile(r"\b(?:form\s+)?10-?k\s+summary\b", re.IGNORECASE)
SUMMARY_HEADER_LINE_PATTERN = re.compile(
    r"^\s*(?:form\s+)?10-?k\s+summary\s*$", re.IGNORECASE
)
INDEX_HEADER_LINE_PATTERN = re.compile(r"^\s*index(?:\b|$)", re.IGNORECASE)
START_TOC_SUMMARY_MARKER_PATTERN = re.compile(
    r"(?i)\b(?:table\s+of\s+contents?|index|summary|form\s+10-?k\s+summary|10-?k\s+summary)\b"
)
ITEM_WORD_PATTERN = re.compile(r"\bITEM\b", re.IGNORECASE)
PART_MARKER_PATTERN = re.compile(r"\bPART\s+(?P<part>IV|III|II|I)\b(?!\s*,)", re.IGNORECASE)
PART_LINESTART_PATTERN = re.compile(r"^\s*PART\s+(?P<part>IV|III|II|I)\b", re.IGNORECASE)
ITEM_CANDIDATE_PATTERN = re.compile(
    r"\bITEM\s+(?P<num>\d+|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?",
    re.IGNORECASE,
)
COMBINED_PART_ITEM_PATTERN = re.compile(
    r"^[ \t\-\*\u2022\u00b7\u2013\u2014]*"
    r"PART\s+(?P<part>IV|III|II|I)\s*[,:\-\u2013\u2014]+\s*"
    r"(?P<item>ITEM\s+(?P<num>\d+|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?)",
    re.IGNORECASE,
)
ITEM_LINESTART_PATTERN = re.compile(
    r"^\s*(?:PART\s+[IVXLCDM]+\s*[:\-]?\s*)?ITEM\s+(?P<num>\d+|[IVXLCDM]+)"
    r"(?P<let>[A-Z])?(?=\b|(?-i:[A-Z]))",
    re.IGNORECASE,
)
NUMERIC_DOT_HEADING_PATTERN = re.compile(
    r"^\s*(?P<num>\d{1,2})(?P<let>[A-Z])?\s*[\.\):\-]?\s+(?P<title>.+?)\s*$",
    re.IGNORECASE,
)
DOT_LEADER_PATTERN = re.compile(r"(?:\.{4,}|(?:\.\s*){4,})")
TOC_DOT_LEADER_PATTERN = re.compile(r"(?:\.{4,}|(?:\.\s*){4,})\s*\d{1,4}\s*$")
CONTINUED_PATTERN = re.compile(r"\bcontinued\b", re.IGNORECASE)
ITEM_MENTION_PATTERN = re.compile(r"\bITEM\s+(?:\d+|[IVXLCDM]+)[A-Z]?\b", re.IGNORECASE)
PART_ONLY_PREFIX_PATTERN = re.compile(
    r"^\s*PART\s+(?:IV|III|II|I)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
PART_PREFIX_TAIL_PATTERN = re.compile(
    r"(?:^|[\s:;,.])PART\s+(?P<part>IV|III|II|I)\s*$",
    re.IGNORECASE,
)
CROSS_REF_PREFIX_PATTERN = re.compile(
    r"(?i)\bsee\b|\brefer\b|\bas discussed\b|\bas described\b|\bas set forth\b|\bas noted\b"
    r"|\bpursuant to\b|\bunder\b|\bin accordance with\b",
)
CROSS_REF_PART_PATTERN = re.compile(
    r"(?i)\bin\s+part\s+(?:IV|III|II|I)\b",
)
COVER_PAGE_MARKER_PATTERN = re.compile(
    r"UNITED STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
    re.IGNORECASE,
)
FORM_10K_PATTERN = re.compile(r"\bFORM\s+10-?K\b", re.IGNORECASE)

PAGE_HYPHEN_PATTERN = re.compile(r"^\s*-\d{1,4}-\s*$")
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")
PAGE_ROMAN_PATTERN = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE)
PAGE_OF_PATTERN = re.compile(r"^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)

EMPTY_ITEM_PATTERN = re.compile(
    r"^\s*(?:\(?[a-z]\)?\s*)?(?:none\.?|n/?a\.?|not applicable\.?|not required\.?|\[reserved\]|reserved)\s*$",
    re.IGNORECASE,
)
START_TRAILING_PAGE_NUMBER_PATTERN = re.compile(r"\s+\d+\s*$")
PROSE_SENTENCE_BREAK_PATTERN = re.compile(r"[.!?]\s+[A-Za-z]")
CROSS_REF_SUFFIX_START_PATTERN = re.compile(
    r"(?i)^(?:see|refer|as discussed|as described|as set forth|as noted|pursuant to|under|"
    r"in accordance with|in part\s+(?:IV|III|II|I))\b",
)
PART_REF_PATTERN = re.compile(r"(?i)\bpart\s+(?:IV|III|II|I)\b")
HEADING_TAIL_STRONG_PATTERN = re.compile(
    r"(?i)\b(?:is incorporated(?: herein)? by reference|herein by reference|set forth)\b"
)
HEADING_TAIL_SOFT_PATTERN = re.compile(
    r"\b(?:THE|SEE|REFER|PAGES|INCLUDED|BEGIN)\b",
    re.IGNORECASE,
)

COVER_PAGE_ANCHOR_PATTERN = re.compile(
    r"(?im)^\s*(?:table\s+of\s+contents?\b|part\s+(?:iv|iii|ii|i)\b|item\s+\d+)",
)

_WRAPPED_TABLE_OF_CONTENTS_PATTERN = re.compile(
    r"(?im)^\s*table(?:\s*\n+\s*|\s+)of(?:\s*\n+\s*|\s+)contents?\b"
)
_WRAPPED_PART_PATTERN = re.compile(r"(?im)^(?P<lead>\s*part)\s*\n+\s*(?P<part>iv|iii|ii|i)\b")
_WRAPPED_ITEM_PATTERN = re.compile(
    r"(?im)^(?P<lead>\s*item)\s*\n+\s*(?P<num>\d+|[ivxlcdm]+)\s*(?P<let>[a-z])?\s*(?P<punc>[\.:])?"
)
_WRAPPED_ITEM_LETTER_PATTERN = re.compile(
    r"(?im)^(?P<lead>\s*item\s+(?P<num>\d+|[ivxlcdm]+))\s*(?P<num_punc>[\.\):\-])?"
    r"\s*\n+\s*(?P<let>[a-z])\s*(?P<punc>[\.\):\-])?(?=\s|$)"
)
_ITEM_SUFFIX_PAREN_PATTERN = re.compile(
    r"(?im)^(?P<lead>\s*item\s+\d+[a-z])\s*\((?P<suffix>[a-z])\)\s*(?P<punc>[\.\):\-])?"
)

TOC_ENTRY_PATTERN = re.compile(
    r"\bITEM\s+(?P<num>\d+|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?\s+.{0,120}?\b(?P<page>\d{1,3})\b(?=\s+(?:ITEM|PART)\b|\s*$)",
    re.IGNORECASE | re.DOTALL,
)

# Embedded heading / TOC patterns.
EMBEDDED_ITEM_PATTERN = re.compile(
    r"(?i)^[ \t]*ITEM[ \t]+(?P<num>\d{1,2})(?P<let>[A-Z])?\b"
)
# Corpus scan shows occasional ITEM I-style tokens; accept roman numerals for embedded checks.
EMBEDDED_ITEM_ROMAN_PATTERN = re.compile(r"(?i)^[ \t]*ITEM[ \t]+(?P<roman>[IVXLCDM]+)\b")
EMBEDDED_PART_PATTERN = re.compile(
    r"(?i)^[ \t]*PART[ \t]+(?P<roman>[IVX]+)[ \t]*[.\-\u2013\u2014:]?[ \t]*$"
)
STRICT_PART_PATTERN = re.compile(
    r"(?im)^\s*PART\s+(?P<part>I|II|III|IV)\s*[:.\-\u2013\u2014]?\s*$"
)
EMBEDDED_CONTINUATION_PATTERN = re.compile(r"(?i)\b(?:continued|cont\.|concluded)\b")
EMBEDDED_CROSS_REF_PATTERN = re.compile(
    r"(?i)\b(?:see|refer to|refer back to|as discussed|as described|as set forth|as noted|"
    r"included in|incorporated by reference|incorporated herein by reference|of this form|"
    r"in this form|set forth in|pursuant to|under|in accordance with)\b"
)
EMBEDDED_RESERVED_PATTERN = re.compile(r"(?i)\[\s*reserved\s*\]")
GIJ_OMIT_BLOCK_START_RE = re.compile(
    r"(?i)\bthe\s+following\s+items\s+have\s+been\s+omitted\s+in\s+accordance\s+with\s+"
    r"general\s+instruction\s+j\s+to\s+form\s+10-?k\b"
)
GIJ_SUBSTITUTE_BLOCK_RE = re.compile(
    r"(?i)\bsubstitute\s+information\s+provided\s+in\s+accordance\s+with\s+"
    r"general\s+instruction\s+j\s+to\s+form\s+10-?k\b"
)
GIJ_GENERAL_RE = re.compile(r"(?i)\bgeneral\s+instruction\s+j\s+to\s+form\s+10-?k\b")
STANDARD_ITEM_TOKEN_RE = re.compile(r"(?i)\bItem\s+(?P<id>\d{1,2}[A-Z]?)\b")
EMPTY_SECTION_LINE_RE = re.compile(
    r"(?i)^\s*[\(\[]?\s*(?:reserved|omitted|not applicable|nothing to report|none)\s*[\)\]]?\.?\s*$"
)
EMBEDDED_TOC_DOT_LEADER_PATTERN = re.compile(r"(?:\.{8,}|(?:\.\s*){8,})")
EMBEDDED_TOC_TRAILING_PAGE_PATTERN = re.compile(r"\s+\d{1,4}\s*$")
EMBEDDED_TOC_HEADER_PATTERN = re.compile(
    r"(?i)\b(?:table\s+of\s+contents?|index|(?:form\s+)?10-?k\s+summary)\b"
)
EMBEDDED_TOC_WINDOW_HEADER_PATTERN = re.compile(
    r"(?i)^\s*(?:table\s+of\s+contents?|index|(?:form\s+)?10-?k\s+summary)\b"
)
EMBEDDED_TOC_PART_ITEM_PATTERN = re.compile(
    r"(?i)^\s*PART\s+(?P<part>[IVX]+)\s*(?:[,/\-:;]\s*|\s+)?ITEM\s+"
    r"(?P<num>\d{1,2})(?P<let>[A-Z])?\b"
)
EMBEDDED_TOC_ITEM_ONLY_PATTERN = re.compile(r"(?i)^\s*ITEM\s+\d{1,2}[A-Z]?\b")
EMBEDDED_SEPARATOR_PATTERN = re.compile(r"^[\s\-=*]{3,}$")
