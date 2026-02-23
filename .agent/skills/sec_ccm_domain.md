---
name: sec_ccm_domain
description: Defines differences between identifiers and states strict join rules for market data.
---

# SEC CCM Domain

## Identifiers
- **cik_10**: SEC entity
- **gvkey**: Compustat firm
- **kypermno**: CRSP security

## Market Data Joins Rule
Market data joins must strictly anchor to or follow the `filing_date` to avoid look-ahead bias.
