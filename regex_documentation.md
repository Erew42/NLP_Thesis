# Regex Expressions Used in the Project

This document lists all regular expressions identified in the codebase, grouped by file.

## `extract_regexes.py`

| Line | Pattern | Context | 
|---|---|---|
| 58 | `...` | Raw String Context Match |
| 60 | `r[\'"](.*?)[\'"](?![\'"])` | Function Call: re.findall |
| 60 | `r[\'` | Raw String Context Match |

## `run_impossible_by_regime_diagnostics.py`

| Line | Pattern | Context | 
|---|---|---|
| 80 | `[^A-Za-z0-9_-]+` | Function Call: re.sub |
| 85 | `\s+\d{1,4}\s*$` | Function Call: re.search |
| 97 | `(?i)\bITEM\s+\d{1,2}[A-Z]?` | Function Call: re.search |
| 109 | `(?i)\bITEM\s+\d{1,2}\s*\([A-Z0-9]\)` | Function Call: re.search |
| 115 | `(?i)^\s*(see\|refer to\|as discussed)\b` | Function Call: re.search |
| 334 | `\s+` | Function Call: re.sub |

## `run_item_regime_benchmark.py`

| Line | Pattern | Context | 
|---|---|---|
| 82 | `\\s+` | Function Call: re.sub |
| 92 | `(?i)\\bITEM\\s+14\\b.{0,200}` | Function Call: re.search |
| 94 | `\\s+` | Function Call: re.sub |

## `run_item_regime_comparison_v3.py`

| Line | Pattern | Context | 
|---|---|---|
| 15 | `Total filings processed:\s*(\d+)` | Function Call: re.search |
| 18 | `Total extracted items:\s*baseline=(\d+)\s*regime-aware=(\d+)` | Function Call: re.search |
| 24 | `Items per filing:\s*baseline=([\d\.]+)\s*regime-aware=([\d\.]+)` | Function Call: re.search |
| 30 | `Modern filings coverage .*: filings=(\d+)\s*items/filing baseline=([\d\.]+)\s*regime-aware=([\d\.]+)` | Function Call: re.search |
| 59 | `item_id\s+(\S+):\s*(\d+)` | Function Call: re.search |

## `src\thesis_pkg\core\ccm\canonical_links.py`

| Line | Pattern | Context | 
|---|---|---|
| 67 | `\.0$` | Raw String Context Match |
| 68 | `\D` | Raw String Context Match |
| 538 | `^\d{10}$` | Function Call: pl.col('cik_10').str.contains |

## `src\thesis_pkg\core\ccm\sec_ccm_premerge.py`

| Line | Pattern | Context | 
|---|---|---|
| 51 | `\D` | Raw String Context Match |
| 170 | `^\d{10}$` | Function Call: pl.col('cik_10').str.contains |

## `src\thesis_pkg\core\ccm\transforms.py`

| Line | Pattern | Context | 
|---|---|---|
| 850 | `\.0$` | Raw String Context Match |
| 851 | `\D` | Raw String Context Match |
| 885 | `\.0$` | Raw String Context Match |
| 886 | `\D` | Raw String Context Match |

## `src\thesis_pkg\core\sec\embedded_headings.py`

| Line | Pattern | Context | 
|---|---|---|
| 97 | `\s+` | Function Call: re.sub |
| 113 | `\s+` | Function Call: re.sub |
| 116 | `(\d+)` | Function Call: re.match |
| 125 | `\s+` | Function Call: re.sub |
| 166 | `(?i)(?:and\|&\|/)` | Function Call: re.fullmatch |
| 168 | `[A-Za-z]` | Function Call: re.search |
| 270 | `\.[a-z]` | Function Call: re.match |
| 298 | `[.!?]\s*$` | Function Call: re.search |
| 300 | `[A-Za-z][A-Za-z'&-]*` | Function Call: re.findall |
| 300 | `[A-Za-z][A-Za-z` | Raw String Context Match |
| 320 | `(?i)\bPART\s+[IVX]+\s*,\s*ITEM\b` | Function Call: re.search |
| 322 | `(?i)\bITEM\s+\d{1,2}[A-Z]?\s*,` | Function Call: re.search |
| 324 | `\s{2,}` | Function Call: re.search |
| 335 | `\s{2,}\d{1,4}\s*$` | Function Call: re.search |
| 431 | `[A-Za-z]` | Function Call: re.findall |
| 440 | `[.!?]\s*$` | Function Call: re.search |
| 447 | `[.;]\s*$` | Function Call: re.search |
| 449 | `[A-Za-z][A-Za-z'&-]*` | Function Call: re.findall |
| 449 | `[A-Za-z][A-Za-z` | Raw String Context Match |

## `src\thesis_pkg\core\sec\extraction.py`

| Line | Pattern | Context | 
|---|---|---|
| 150 | `</\s*(sec-header\|header)\s*>` | Function Call: re.search |
| 251 | `(?i)form\s+10-q\|quarterly report` | Function Call: re.search |
| 393 | `(?i)^\s*[\(\[]?\s*continued\b` | Function Call: re.search |
| 394 | `(?i)^\s*[\(\[]\s*[a-z0-9]` | Function Call: re.match |
| 593 | `(?i)^\s*[\(\[]?\s*continued\b` | Function Call: re.search |
| 595 | `(?i)^\s*[\(\[]\s*[a-z0-9]` | Function Call: re.match |
| 752 | `\s+` | Function Call: re.sub |
| 762 | `md\s*&\s*a` | Function Call: re.search |
| 861 | `^10-?K[/-]A` | Function Call: re.match |
| 861 | `^10-?Q[/-]A` | Function Call: re.match |

## `src\thesis_pkg\core\sec\heuristics.py`

| Line | Pattern | Context | 
|---|---|---|
| 101 | `[A-Z]` | Function Call: re.fullmatch |
| 111 | `\s+\d{1,4}\s*$` | Function Call: re.sub |
| 114 | `[^A-Z0-9' ]+` | Function Call: re.sub |
| 114 | `[^A-Z0-9` | Raw String Context Match |
| 115 | `\s+` | Function Call: re.sub |
| 192 | `\s*\(\s*[A-Z]\s*\)\s*` | Function Call: re.match |
| 200 | `\bITEM\b\|\bPART\b` | Function Call: re.split |
| 251 | `[A-Za-z0-9]` | Function Call: re.search |
| 258 | `[\.!?]` | Function Call: re.search |
| 268 | `\bsee\b\|\brefer\b` | Function Call: re.search |
| 320 | `^\(\s*[A-Z]\s*\)\s*` | Function Call: re.sub |
| 347 | `[A-Za-z]+` | Function Call: re.findall |
| 393 | `[^A-Za-z0-9]+` | Raw String Context Match |
| 413 | `[A-Za-z]+` | Function Call: re.findall |
| 426 | `[A-Za-z]+` | Function Call: re.findall |
| 482 | `\s+\d{1,4}\s*$` | Function Call: re.search |
| 522 | `[A-Za-z]` | Function Call: re.search |
| 658 | `^(?P<num>\d{1,2})(?P<let>[A-Z])?$` | Function Call: re.match |
| 963 | `\s+\d{1,4}\s*$` | Function Call: re.search |
| 1116 | `[\s\-=*]{3,}` | Function Call: re.fullmatch |
| 1118 | `[A-Za-z]` | Function Call: re.findall |
| 1126 | `[.!?]\s*$` | Function Call: re.search |
| 1160 | `\s+\d{1,4}\s*$` | Function Call: re.search |
| 1236 | `\s+\d{1,4}\s*$` | Function Call: re.search |
| 1241 | `\s+\d{1,4}\s*$` | Function Call: re.search |
| 1300 | `table\s+of\s+contents?` | Function Call: re.search |
| 1372 | `\n{3,}` | Function Call: re.sub |
| 1386 | `(?i)^\s*PART\s+(?:IV\|III\|II\|I)\b` | Function Call: re.sub |
| 1413 | `^(\d+)` | Function Call: re.match |
| 1489 | `[.!?]\s*$` | Function Call: re.search |
| 1491 | `[A-Za-z][A-Za-z'&-]*` | Function Call: re.findall |
| 1491 | `[A-Za-z][A-Za-z` | Raw String Context Match |
| 1510 | `[\s\-=*]{3,}` | Function Call: re.fullmatch |

## `src\thesis_pkg\core\sec\html_audit.py`

| Line | Pattern | Context | 
|---|---|---|
| 155 | `[^A-Za-z0-9_-]+` | Function Call: re.sub |
| 185 | `^(\d+)([A-Z]?)$` | Function Call: re.match |

## `src\thesis_pkg\core\sec\parquet_stream.py`

| Line | Pattern | Context | 
|---|---|---|
| 33 | `\d{4}` | Function Call: re.fullmatch |

## `src\thesis_pkg\core\sec\patterns.py`

| Line | Pattern | Context | 
|---|---|---|
| 6 | `(\d{8})_([^_]+)_edgar_data_(\d+)_([\d-]+)\.txt$` | Function Call: re.compile |
| 16 | `CENTRAL INDEX KEY:\s*(\d+)` | Function Call: re.compile |
| 17 | `FILED AS OF DATE:\s*(\d{8})` | Function Call: re.compile |
| 18 | `CONFORMED PERIOD OF REPORT:\s*(\d{8})` | Function Call: re.compile |
| 19 | `ACCESSION NUMBER:\s*([\d-]+)` | Function Call: re.compile |
| 21 | `(?is)<\s*(?P<tag>sec-header\|header\|filestats\|file-stats\|xml_chars\|xml-chars)\b[^>]*>.*?<\s*/\s*(?P=tag)\s*>` | Function Call: re.compile |
| 25 | `(?is)^\s*<\s*(sec-header\|header)\b[^>]*>.*` | Function Call: re.compile |
| 29 | `\btable\s+of\s+contents?\b\|\btable\s+of\s+content\b` | Function Call: re.compile |
| 35 | `^\s*table\s+of\s+contents?\s*$` | Function Call: re.compile |
| 36 | `\btable\s+of\s+contents?\b\|\btable\s+of\s+content\b\|\bindex\b\|\b(?:form\s+)?10-?k\s+summary\b` | Function Call: re.compile |
| 40 | `\b(?:form\s+)?10-?k\s+summary\b` | Function Call: re.compile |
| 41 | `^\s*(?:form\s+)?10-?k\s+summary\s*$` | Function Call: re.compile |
| 42 | `^\s*(?:form\s+)?10-?k\s+summary\s*$` | Raw String Context Match |
| 44 | `^\s*index(?:\b\|$)` | Function Call: re.compile |
| 45 | `(?i)\b(?:table\s+of\s+contents?\|index\|summary\|form\s+10-?k\s+summary\|10-?k\s+summary)\b` | Function Call: re.compile |
| 48 | `\bITEM\b` | Function Call: re.compile |
| 49 | `\bPART\s+(?P<part>IV\|III\|II\|I)\b(?!\s*,)` | Function Call: re.compile |
| 50 | `^\s*PART\s+(?P<part>IV\|III\|II\|I)\b` | Function Call: re.compile |
| 51 | `\bITEM\s+(?P<num>\d+\|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?` | Function Call: re.compile |
| 55 | `^[ \t\-\*\u2022\u00b7\u2013\u2014]*PART\s+(?P<part>IV\|III\|II\|I)\s*[,:\-\u2013\u2014]+\s*(?P<item>ITEM\s+(?P<num>\d+\|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?)` | Function Call: re.compile |
| 61 | `^\s*(?:PART\s+[IVXLCDM]+\s*[:\-]?\s*)?ITEM\s+(?P<num>\d+\|[IVXLCDM]+)(?P<let>[A-Z])?(?=\b\|(?-i:[A-Z]))` | Function Call: re.compile |
| 66 | `^\s*(?P<num>\d{1,2})(?P<let>[A-Z])?\s*[\.\):\-]?\s+(?P<title>.+?)\s*$` | Function Call: re.compile |
| 70 | `(?:\.{4,}\|(?:\.\s*){4,})` | Function Call: re.compile |
| 71 | `(?:\.{4,}\|(?:\.\s*){4,})\s*\d{1,4}\s*$` | Function Call: re.compile |
| 72 | `\bcontinued\b` | Function Call: re.compile |
| 73 | `\bITEM\s+(?:\d+\|[IVXLCDM]+)[A-Z]?\b` | Function Call: re.compile |
| 74 | `^\s*PART\s+(?:IV\|III\|II\|I)\s*[:\-]?\s*$` | Function Call: re.compile |
| 78 | `(?:^\|[\s:;,.])PART\s+(?P<part>IV\|III\|II\|I)\s*$` | Function Call: re.compile |
| 82 | `(?i)\bsee\b\|\brefer\b\|\bas discussed\b\|\bas described\b\|\bas set forth\b\|\bas noted\b\|\bpursuant to\b\|\bunder\b\|\bin accordance with\b` | Function Call: re.compile |
| 86 | `(?i)\bin\s+part\s+(?:IV\|III\|II\|I)\b` | Function Call: re.compile |
| 89 | `UNITED STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION` | Function Call: re.compile |
| 93 | `\bFORM\s+10-?K\b` | Function Call: re.compile |
| 95 | `^\s*-\d{1,4}-\s*$` | Function Call: re.compile |
| 96 | `^\s*\d{1,4}\s*$` | Function Call: re.compile |
| 97 | `^\s*[ivxlcdm]{1,6}\s*$` | Function Call: re.compile |
| 98 | `^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$` | Function Call: re.compile |
| 100 | `^\s*(?:\(?[a-z]\)?\s*)?(?:none\.?\|n/?a\.?\|not applicable\.?\|not required\.?\|\[reserved\]\|reserved)\s*$` | Function Call: re.compile |
| 104 | `\s+\d+\s*$` | Function Call: re.compile |
| 105 | `[.!?]\s+[A-Za-z]` | Function Call: re.compile |
| 106 | `(?i)^(?:see\|refer\|as discussed\|as described\|as set forth\|as noted\|pursuant to\|under\|in accordance with\|in part\s+(?:IV\|III\|II\|I))\b` | Function Call: re.compile |
| 110 | `(?i)\bpart\s+(?:IV\|III\|II\|I)\b` | Function Call: re.compile |
| 111 | `(?i)\b(?:is incorporated(?: herein)? by reference\|herein by reference\|set forth)\b` | Function Call: re.compile |
| 114 | `\b(?:THE\|SEE\|REFER\|PAGES\|INCLUDED\|BEGIN)\b` | Function Call: re.compile |
| 119 | `(?im)^\s*(?:table\s+of\s+contents?\b\|part\s+(?:iv\|iii\|ii\|i)\b\|item\s+\d+)` | Function Call: re.compile |
| 123 | `(?im)^\s*table(?:\s*\n+\s*\|\s+)of(?:\s*\n+\s*\|\s+)contents?\b` | Function Call: re.compile |
| 126 | `(?im)^(?P<lead>\s*part)\s*\n+\s*(?P<part>iv\|iii\|ii\|i)\b` | Function Call: re.compile |
| 127 | `(?im)^(?P<lead>\s*item)\s*\n+\s*(?P<num>\d+\|[ivxlcdm]+)\s*(?P<let>[a-z])?\s*(?P<punc>[\.:])?` | Function Call: re.compile |
| 130 | `(?im)^(?P<lead>\s*item\s+(?P<num>\d+\|[ivxlcdm]+))\s*(?P<num_punc>[\.\):\-])?\s*\n+\s*(?P<let>[a-z])\s*(?P<punc>[\.\):\-])?(?=\s\|$)` | Function Call: re.compile |
| 134 | `(?im)^(?P<lead>\s*item\s+\d+[a-z])\s*\((?P<suffix>[a-z])\)\s*(?P<punc>[\.\):\-])?` | Function Call: re.compile |
| 138 | `\bITEM\s+(?P<num>\d+\|[IVXLCDM]+)(?P<let>[A-Z])?\s*[\.:]?\s+.{0,120}?\b(?P<page>\d{1,3})\b(?=\s+(?:ITEM\|PART)\b\|\s*$)` | Function Call: re.compile |
| 144 | `(?i)^[ \t]*ITEM[ \t]+(?P<num>\d{1,2})(?P<let>[A-Z])?\b` | Function Call: re.compile |
| 148 | `(?i)^[ \t]*ITEM[ \t]+(?P<roman>[IVXLCDM]+)\b` | Function Call: re.compile |
| 149 | `(?i)^[ \t]*PART[ \t]+(?P<roman>[IVX]+)[ \t]*[.\-\u2013\u2014:]?[ \t]*$` | Function Call: re.compile |
| 152 | `(?im)^\s*PART\s+(?P<part>I\|II\|III\|IV)\s*[:.\-\u2013\u2014]?\s*$` | Function Call: re.compile |
| 155 | `(?i)\b(?:continued\|cont\.\|concluded)\b` | Function Call: re.compile |
| 156 | `(?i)\b(?:see\|refer to\|refer back to\|as discussed\|as described\|as set forth\|as noted\|included in\|incorporated by reference\|incorporated herein by reference\|of this form\|in this form\|set forth in\|pursuant to\|under\|in accordance with)\b` | Function Call: re.compile |
| 161 | `(?i)\[\s*reserved\s*\]` | Function Call: re.compile |
| 162 | `(?i)\bthe\s+following\s+items\s+have\s+been\s+omitted\s+in\s+accordance\s+with\s+general\s+instruction\s+j\s+to\s+form\s+10-?k\b` | Function Call: re.compile |
| 166 | `(?i)\bsubstitute\s+information\s+provided\s+in\s+accordance\s+with\s+general\s+instruction\s+j\s+to\s+form\s+10-?k\b` | Function Call: re.compile |
| 170 | `(?i)\bgeneral\s+instruction\s+j\s+to\s+form\s+10-?k\b` | Function Call: re.compile |
| 171 | `(?i)\bItem\s+(?P<id>\d{1,2}[A-Z]?)\b` | Function Call: re.compile |
| 172 | `(?i)^\s*[\(\[]?\s*(?:reserved\|omitted\|not applicable\|nothing to report\|none)\s*[\)\]]?\.?\s*$` | Function Call: re.compile |
| 175 | `(?:\.{8,}\|(?:\.\s*){8,})` | Function Call: re.compile |
| 176 | `\s+\d{1,4}\s*$` | Function Call: re.compile |
| 177 | `(?i)\b(?:table\s+of\s+contents?\|index\|(?:form\s+)?10-?k\s+summary)\b` | Function Call: re.compile |
| 180 | `(?i)^\s*(?:table\s+of\s+contents?\|index\|(?:form\s+)?10-?k\s+summary)\b` | Function Call: re.compile |
| 183 | `(?i)^\s*PART\s+(?P<part>[IVX]+)\s*(?:[,/\-:;]\s*\|\s+)?ITEM\s+(?P<num>\d{1,2})(?P<let>[A-Z])?\b` | Function Call: re.compile |
| 187 | `(?i)^\s*ITEM\s+\d{1,2}[A-Z]?\b` | Function Call: re.compile |
| 188 | `^[\s\-=*]{3,}$` | Function Call: re.compile |

## `src\thesis_pkg\core\sec\regime.py`

| Line | Pattern | Context | 
|---|---|---|
| 189 | `[-/]\s*A$` | Function Call: re.search |

## `src\thesis_pkg\core\sec\suspicious_boundary_diagnostics.py`

| Line | Pattern | Context | 
|---|---|---|
| 112 | `\bITEM\s+(?:\d+\|[IVXLCDM]+)[A-Z]?\b` | Function Call: re.compile |
| 113 | `(?m)^[ \t]*PART[ \t]+[IVX]+\b` | Function Call: re.compile |
| 114 | `(?m)^[ \t]*ITEM[ \t]+\d+[A-Z]?\b` | Function Call: re.compile |
| 129 | `\bPART\s+(I\|II)\b` | Function Call: re.compile |
| 130 | `\bITEM\b` | Function Call: re.compile |
| 131 | `\bFORM\s+10-Q\b\|\bQUARTERLY\s+REPORT\b` | Function Call: re.compile |
| 132 | `\bFORM\s+10-Q\b\|\bQUARTERLY\s+REPORT\b` | Raw String Context Match |
| 134 | `\bPART\b[^\n]{0,80}[,\-][^\n]{0,80}\bITEM\b\|\bITEM\b[^\n]{0,80}[,\-][^\n]{0,80}\bPART\b` | Function Call: re.compile |
| 666 | `\d+[A-Z]$` | Function Call: re.search |
| 809 | `(?i)\bITEM\s+(?P<num>\d{1,2})(?P<let>[A-Z])?\b` | Function Call: re.search |
| 818 | `(?i)\bPART\s+(?P<roman>[IVX]+)\b` | Function Call: re.search |
| 1001 | `[A-Za-z]` | Function Call: re.search |
| 1008 | `[A-Za-z]` | Function Call: re.search |
| 1010 | `\u2026\|\.{2,}` | Function Call: re.search |
| 1059 | `[,\s]+` | Function Call: re.split |
| 1076 | `[^A-Z0-9]` | Function Call: re.sub |
| 2577 | `^\s*[ABC]\s*[\.\):\-]` | Function Call: re.match |
| 3467 | `^\s*[ABC]\s*[\.\):\-]` | Function Call: re.match |

## `src\thesis_pkg\core\sec\utilities.py`

| Line | Pattern | Context | 
|---|---|---|
| 10 | `\D` | Function Call: re.sub |
| 42 | `\d{8}` | Function Call: re.fullmatch |
| 47 | `\d{4}-\d{2}-\d{2}` | Function Call: re.fullmatch |
| 61 | `[IVXLCDM]+` | Function Call: re.fullmatch |
| 83 | `^(?P<num>\d{1,2})` | Function Call: re.match |
| 101 | `[\s\-\*\u2022\u00b7\u2013\u2014]+` | Function Call: re.fullmatch |

## `src\thesis_pkg\pipelines\sec_pipeline.py`

| Line | Pattern | Context | 
|---|---|---|
| 685 | `(?P<year>\d{4})_batch_` | Function Call: re.match |

## `tests\docs\test_scaffold_docs.py`

| Line | Pattern | Context | 
|---|---|---|
| 80 | `^\s*-\s*__init__:` | Function Call: re.match |

## `tools\docs_check.py`

| Line | Pattern | Context | 
|---|---|---|
| 109 | `^- (.+)$` | Function Call: re.match |
| 114 | `^[A-Z]+ - ` | Function Call: re.match |
| 198 | `\[[^\]]*\]\(([^)]+)\)` | Function Call: re.findall |

