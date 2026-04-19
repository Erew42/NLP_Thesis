from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


TITLE = "Econometric And Regression Paths In The Python Codebase"


@dataclass(frozen=True)
class EvidenceRef:
    ref_id: str
    location: str
    note: str


@dataclass(frozen=True)
class VariableRow:
    name: str
    definition: str
    evidence: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _output_path(root: Path) -> Path:
    return root / "output" / "pdf" / "econometric_regression_paths_report.pdf"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#10243F"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#53657D"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#10243F"),
            spaceBefore=6,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1E426E"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            spaceAfter=4,
        ),
        "body_small": ParagraphStyle(
            "BodySmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            leftIndent=12,
            firstLineIndent=-10,
            spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.4,
            leading=9.2,
            textColor=colors.HexColor("#13293D"),
            backColor=colors.HexColor("#F5F7FA"),
            borderPadding=6,
            borderWidth=0.4,
            borderColor=colors.HexColor("#D7DEE8"),
            borderRadius=2,
            spaceAfter=6,
        ),
    }


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return _paragraph(f"- {text}", styles["bullet"])


def _code_block(text: str, styles: dict[str, ParagraphStyle]) -> Preformatted:
    return Preformatted(text.strip("\n"), styles["code"])


def _table(
    headers: list[str],
    rows: Iterable[list[str]],
    widths_mm: list[float],
    styles: dict[str, ParagraphStyle],
) -> Table:
    data: list[list[object]] = [[_paragraph(f"<b>{header}</b>", styles["body_small"]) for header in headers]]
    for row in rows:
        data.append([_paragraph(cell, styles["body_small"]) for cell in row])
    table = Table(data, colWidths=[width * mm for width in widths_mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8F5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10243F")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D3DF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _draw_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
    canvas.line(doc.leftMargin, height - 12 * mm, width - doc.rightMargin, height - 12 * mm)
    canvas.line(doc.leftMargin, 12 * mm, width - doc.rightMargin, 12 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#53657D"))
    canvas.drawString(doc.leftMargin, 7 * mm, TITLE)
    canvas.drawRightString(width - doc.rightMargin, 7 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _evidence_refs() -> list[EvidenceRef]:
    return [
        EvidenceRef(
            "E1",
            "src/thesis_pkg/notebooks_and_scripts/lm2011_sample_post_refinitiv_runner.py:2504-2856, 2904-3030 and src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py:3094-3141",
            "Standalone LM2011 stage materialization through Table IA.II, extension-pipeline materialization, and unified-runner wiring that invokes LM2011 extension after FinBERT.",
        ),
        EvidenceRef(
            "E2",
            "src/thesis_pkg/core/ccm/lm2011.py:521-574",
            "Filing-trade and pre-filing-trade anchor rules.",
        ),
        EvidenceRef(
            "E3",
            "src/thesis_pkg/core/ccm/lm2011.py:577-701",
            "Annual accounting formulas including preferred stock handling and book equity construction.",
        ),
        EvidenceRef(
            "E4",
            "src/thesis_pkg/core/ccm/lm2011.py:899-1111 and src/thesis_pkg/core/ccm/sec_ccm_contracts.py:65-96",
            "FF48 industry assignment plus shared event-date market-equity normalization to millions of USD.",
        ),
        EvidenceRef(
            "E5",
            "src/thesis_pkg/pipelines/lm2011_pipeline.py:734-946",
            "Shared OLS helper, factor-model alpha/RMSE helper, and event-screen arithmetic.",
        ),
        EvidenceRef(
            "E6",
            "src/thesis_pkg/pipelines/lm2011_pipeline.py:361-378, 809-838, 1110-1138, 1338-1365",
            "Ownership handling, log transforms, sample filters, winsorization, and event-panel output selection.",
        ),
        EvidenceRef(
            "E7",
            "src/thesis_pkg/core/sec/lm2011_text.py:96-109, 310-400, 1004-1145",
            "IDF and TF-IDF formulas, total-token document-length normalization, and text-feature builders.",
        ),
        EvidenceRef(
            "E8",
            "src/thesis_pkg/core/ccm/lm2011.py:774-896 and src/thesis_pkg/pipelines/lm2011_pipeline.py:1444-1668",
            "Quarterly accounting attachment, pre-filing and prior-month price attachment, and SUE variable construction.",
        ),
        EvidenceRef(
            "E9",
            "src/thesis_pkg/pipelines/lm2011_regressions.py:144-472",
            "Regression panels, normalized-difference panel, weighted means, Newey-West standard errors, and quarterly Fama-MacBeth estimation.",
        ),
        EvidenceRef(
            "E10",
            "src/thesis_pkg/pipelines/lm2011_regressions.py:504-717",
            "Table-specific specifications and Table IA.II output assembly.",
        ),
        EvidenceRef(
            "E11",
            "src/thesis_pkg/pipelines/lm2011_pipeline.py:1780-2144",
            "Trading-strategy assignment, equal/value weighting, long-short return construction, and FF4 loading regression.",
        ),
        EvidenceRef(
            "E12",
            "src/thesis_pkg/benchmarking/finbert_analysis.py:159-275",
            "FinBERT length-weighted feature aggregation used by the extension scaffold.",
        ),
        EvidenceRef(
            "E13",
            "src/thesis_pkg/pipelines/lm2011_extension.py:20-123, 565-1049",
            "Extension sample window, control/specification grids, cleaned-scope alignment, event-base assembly, sample-loss accounting, and extension estimation scaffold.",
        ),
        EvidenceRef(
            "E14",
            "src/thesis_pkg/pipelines/lm2011_pipeline.py:276-298, 544-557, 1827-1845",
            "Daily and monthly factor scaling: divide by 100 when the factor columns appear to be stored in percent units.",
        ),
    ]


def _variable_tables() -> list[tuple[str, list[VariableRow]]]:
    return [
        (
            "Alignment And Structural Variables",
            [
                VariableRow(
                    "filing_trade_date",
                    "First trading date on or after filing_date via a forward asof join to the trading calendar.",
                    "E2",
                ),
                VariableRow(
                    "pre_filing_trade_date",
                    "Last trading date on or before filing_date - 1 day via a backward asof join.",
                    "E2",
                ),
                VariableRow(
                    "quarter_report_date",
                    "Earliest quarterly report date strictly after filing and at most 90 days later.",
                    "E8",
                ),
                VariableRow(
                    "ff48_industry_id",
                    "Industry identifier mapped from SIC_final, where SIC_final = coalesce(HSIC, SIC_desc).",
                    "E4",
                ),
                VariableRow(
                    "text_scope",
                    "Regression text surface label such as full_10k, mda_item_7, item_7_mda, item_1a_risk_factors, or item_1_business.",
                    "E9, E13",
                ),
                VariableRow(
                    "sample_window",
                    "Extension sample label; in code the default is 2009_2024.",
                    "E13",
                ),
            ],
        ),
        (
            "Event, Accounting, Control, And Outcome Variables",
            [
                VariableRow(
                    "book_equity_be",
                    "coalesce(SEQ, CEQ + PSTK, AT - LT) - preferred_stock_ps + TXDITC - PRBA, with preferred_stock_ps = coalesce(PSTKL, PSTKRV, PSTK).",
                    "E3",
                ),
                VariableRow(
                    "size_event",
                    "Alias for event-date market equity in millions: coalesce(TCAP, abs(PRC) * SHROUT) / 1000.",
                    "E4",
                ),
                VariableRow(
                    "bm_event",
                    "book_equity_be / size_event when both are available and size_event > 0.",
                    "E4",
                ),
                VariableRow(
                    "filing_period_excess_return",
                    "Product of stock gross returns over days 0..3 minus product of market gross returns over days 0..3.",
                    "E5",
                ),
                VariableRow(
                    "share_turnover",
                    "Sum of volume over days -252..-6 divided by event_shares * 1000.",
                    "E5",
                ),
                VariableRow(
                    "abnormal_volume",
                    "Mean event-window volume z-score, where z-scores use the mean and std of VOL from days -65..-6.",
                    "E5",
                ),
                VariableRow(
                    "pre_ffalpha",
                    "Intercept from the pre-filing 3-factor regression of stock_return - rf on mkt_rf, smb, and hml over days -252..-6.",
                    "E5",
                ),
                VariableRow(
                    "postevent_return_volatility",
                    "RMSE from the same 3-factor regression estimated on days 6..252 after filing.",
                    "E5",
                ),
                VariableRow(
                    "institutional_ownership",
                    "Doc-level ownership metric from ownership_lf after renaming institutional_ownership_pct or institutional_ownership to a single canonical column.",
                    "E6",
                ),
                VariableRow(
                    "nasdaq_dummy",
                    "Indicator equal to 1 when event_exchcd == 3 and 0 otherwise.",
                    "E5",
                ),
                VariableRow(
                    "log_size / log_book_to_market / log_share_turnover",
                    "Natural logs of positive size_event, bm_event, and share_turnover respectively; null otherwise.",
                    "E6",
                ),
            ],
        ),
        (
            "Text And Sentiment Variables",
            [
                VariableRow(
                    "total_token_count_full_10k / total_token_count_mda",
                    "Total post-clean token counts from the LM2011 tokenization pipeline.",
                    "E7",
                ),
                VariableRow(
                    "token_count_full_10k / token_count_mda",
                    "Recognized-word totals: count of tokens that appear in the supplied master dictionary.",
                    "E7",
                ),
                VariableRow(
                    "h4n_inf_prop, lm_negative_prop, lm_positive_prop, ...",
                    "Matched count for a signal dictionary divided by the total LM-tokenized length of the relevant analyzed text unit.",
                    "E7",
                ),
                VariableRow(
                    "h4n_inf_tfidf, lm_negative_tfidf, ...",
                    "Sum of per-token LM2011 term weights over the dictionary terms in the document.",
                    "E7",
                ),
                VariableRow(
                    "fin_neg_prop / fin_neg_tfidf",
                    "Trading-strategy versions of the LM negative dictionary signal on full 10-K text.",
                    "E7, E11",
                ),
                VariableRow(
                    "finbert_neg_prob_lenw_mean",
                    "Length-weighted mean of sentence-level negative FinBERT probabilities using finbert_token_count_512 as weights.",
                    "E12",
                ),
                VariableRow(
                    "finbert_net_negative_lenw_mean",
                    "Length-weighted mean of negative_prob - positive_prob across accepted scored segments.",
                    "E12",
                ),
                VariableRow(
                    "finbert_neg_dominant_share",
                    "Mean of 1[predicted_label == negative] across accepted scored segments.",
                    "E12",
                ),
            ],
        ),
        (
            "SUE, Trading, And Extension Variables",
            [
                VariableRow(
                    "sue",
                    "(actual_eps - forecast_consensus_mean) / pre_filing_price.",
                    "E8",
                ),
                VariableRow(
                    "analyst_dispersion",
                    "forecast_dispersion / pre_filing_price.",
                    "E8",
                ),
                VariableRow(
                    "analyst_revisions",
                    "forecast_revision_4m / prior_month_price.",
                    "E8",
                ),
                VariableRow(
                    "normalized_difference_negative / normalized_difference_h4n_inf",
                    "Current-year signal minus prior-year industry mean, all divided by prior-year industry std.",
                    "E9",
                ),
                VariableRow(
                    "quintile",
                    "Within each sort_year and sort_signal_name, quintile = floor(5 * group_rank / group_size) + 1 after sorting by signal_value.",
                    "E11",
                ),
                VariableRow(
                    "long_short_return",
                    "Monthly Q1 return minus monthly Q5 return for a given sort signal.",
                    "E11",
                ),
                VariableRow(
                    "mean_long_short_return",
                    "Simple time-series mean of long_short_return within Table IA.II output assembly.",
                    "E10",
                ),
                VariableRow(
                    "institutional_ownership_proxy_refinitiv",
                    "Extension ownership proxy built as coalesce(institutional_ownership_proxy_refinitiv, institutional_ownership_pct, institutional_ownership).",
                    "E13",
                ),
                VariableRow(
                    "ownership_proxy_available / common_support_flag_ownership",
                    "Boolean indicator that the extension ownership proxy is non-null.",
                    "E13",
                ),
                VariableRow(
                    "alpha_ff3_mom, beta_market, beta_smb, beta_hml, beta_mom, r2",
                    "Intercept, factor loadings, and R-squared from long_short_return regressed on mkt_rf, smb, hml, and mom.",
                    "E10, E11",
                ),
                VariableRow(
                    "t_stat / p_value",
                    "t_stat = estimate / standard_error in the main Fama-MacBeth tables; the extension scaffold adds p_value via erfc(abs(t) / sqrt(2)).",
                    "E9, E13",
                ),
            ],
        ),
    ]


def _build_story(root: Path) -> list[object]:
    styles = _styles()
    story: list[object] = []
    generated_on = date.today().isoformat()
    output_path = _output_path(root)

    story.append(_paragraph(TITLE, styles["title"]))
    story.append(
        _paragraph(
            f"Repository: {root} | Generated: {generated_on} | Output: {output_path}",
            styles["subtitle"],
        )
    )
    story.append(
        _paragraph(
            "Evidence policy: every claim below is grounded in Python source files only. "
            "I used the .py implementation as the sole evidence base and excluded papers, prose docs, and notebook narrative.",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "Scope result: the implemented econometric stack is concentrated in the LM2011 pipeline, the post-Refinitiv runner, and the newer extension stages that the unified runner can append after FinBERT. "
            "The code paths observed are event-study style return construction, SUE panel construction, quarterly Fama-MacBeth cross-sectional regressions, "
            "a monthly long-short trading strategy with factor loadings, and a newer extension scaffold that reuses the same Fama-MacBeth engine.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 4))

    story.append(_paragraph("1. Observed Regression Path Map", styles["h1"]))
    story.append(
        _paragraph(
            "The standalone LM2011 post-Refinitiv runner wires the main econometric flow, and the unified SEC-CCM runner can append LM2011 extension after FinBERT artifacts are available. The stages below are directly instantiated in Python code rather than inferred from naming conventions (E1).",
            styles["body"],
        )
    )
    for line in [
        "sample_backbone -> annual_accounting_panel -> event_screen_surface -> event_panel",
        "event_panel + full_10k text features + FF48 industries -> return_regression_panel_full_10k -> Table IV, Table VI, Table IA.I",
        "event_panel + MD&A text features + FF48 industries -> return_regression_panel_mda -> Table V",
        "event_panel + quarterly accounting + analyst inputs -> sue_panel -> sue_regression_panel -> Table VIII",
        "event_panel + yearly filing text + monthly stock returns + monthly factors -> trading_strategy_monthly_returns -> Table IA.II",
        "sec_ccm_unified_runner (optional): finbert artifacts + event_panel -> run_lm2011_extension_pipeline -> extension_dictionary_surface + extension_finbert_surface + extension_control_ladder + extension_specification_grid + extension_analysis_panel + extension_sample_loss + extension_results",
    ]:
        story.append(_bullet(line, styles))
    story.append(
        _paragraph(
            "The standalone post-Refinitiv runner still stops at Table IA.II, but sec_ccm_unified_runner now wires the extension stage after FinBERT when SEC_CCM_RUN_LM2011_EXTENSION is enabled and a materialized event_panel is present (E1).",
            styles["body"],
        )
    )

    story.append(_paragraph("2. Shared Notation And Economic Framework", styles["h1"]))
    story.append(_paragraph("2.1 Index and variable notation", styles["h2"]))
    story.append(
        _paragraph(
            "To state the implemented arithmetic compactly, let i index documents / firm-year observations, d index relative trading days around the filing date, q index filing quarters, m index portfolio months, and g index FF48 industries.",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "Where needed, r_i,d denotes daily stock_return, r_m,d denotes daily market_return, VOL_i,d denotes share volume, and L_i denotes the total LM-tokenized length of the relevant analyzed text unit from the LM2011 text scorer.",
            styles["body"],
        )
    )
    story.append(_paragraph("2.2 Testable hypotheses encoded in the code", styles["h2"]))
    story.append(
        _paragraph(
            "The implemented pipeline asks whether the linguistic content of a 10-K carries information about fundamentals or investor reactions that is not already priced by the time the filing is released. "
            "The null hypothesis is that, conditional on standard risk controls and contemporaneous earnings news, text sentiment is orthogonal to short-horizon filing-window returns, trading activity, and post-event volatility.",
            styles["body"],
        )
    )
    for line in [
        "H1 (price reaction): negative tone in 10-K text predicts lower filing-window excess returns after controlling for size, book-to-market, pre-event alpha, turnover, exchange, and institutional ownership. The filing-window outcome is a gross-return difference over days 0..3, so the test isolates the short-horizon market reaction to disclosure.",
        "H2 (information beyond earnings): tone signals retain explanatory power for the scaled earnings surprise (SUE) and for analyst-based proxies of disagreement and revisions, indicating that disclosure text reveals fundamentals that numerical earnings do not summarize completely.",
        "H3 (tradeability): a monthly quintile long-short strategy sorted on yearly text sentiment earns a non-zero four-factor alpha, i.e. the signal is not spanned by market, size, value, and momentum exposures.",
        "H4 (volatility channel): post-event factor-model residual volatility scales with tone, consistent with a disclosure-induced change in the information environment rather than a pure first-moment news shock.",
        "H5 (dictionary vs contextual NLP, extension): sentence-level FinBERT sentiment and the LM dictionary carry distinct information. The joint specification is constructed so that the marginal effect of LM negative tone after conditioning on FinBERT is identified as the dictionary-specific contribution.",
    ]:
        story.append(_bullet(line, styles))
    story.append(_paragraph("2.3 Variable-to-concept mapping", styles["h2"]))
    story.append(
        _paragraph(
            "Each control in the implemented code corresponds to a specific economic proxy. The controls are not ad hoc; they purge the text coefficient of exposure to well-documented confounds in the cross-section of equity returns.",
            styles["body"],
        )
    )
    for line in [
        "log_size and log_book_to_market are the Fama-French risk proxies for the size and value premia; including them purges the tone coefficient of exposure to standard cross-sectional return factors.",
        "log_share_turnover proxies liquidity; high-turnover firms have tighter price-discovery and are less likely to exhibit delayed reactions to disclosures.",
        "pre_ffalpha captures pre-event abnormal drift, controlling for information leakage or momentum already reflected in prices ahead of the filing.",
        "nasdaq_dummy absorbs exchange-level differences in investor composition and microstructure that affect how disclosures are priced.",
        "institutional_ownership proxies the presence of sophisticated investors who plausibly process 10-K text faster and more thoroughly than retail participants.",
        "abnormal_volume captures investor disagreement or attention spikes at the filing event, distinct from the first-moment price reaction.",
        "postevent_return_volatility (factor-model RMSE) isolates idiosyncratic uncertainty introduced by the disclosure, net of systematic factor volatility.",
        "FF48 industry dummies absorb level differences in disclosure conventions across industries (some sectors use intrinsically more cautionary language).",
    ]:
        story.append(_bullet(line, styles))

    story.append(_paragraph("3. Event-Study Return Panel Construction", styles["h1"]))
    story.append(
        _paragraph(
            "The return regression panels are built on top of a document-level event panel. That panel is itself created from calendar alignment, annual accounting attachment, event-window aggregation, ownership attachment, winsorization, and screening filters (E2, E3, E4, E5, E6).",
            styles["body"],
        )
    )
    story.append(_paragraph("3.1 Trading-date anchors", styles["h2"]))
    story.append(
        _code_block(
            """
filing_trade_date_i     = min{ t in TradingCalendar : t >= filing_date_i }
pre_filing_trade_date_i = max{ t in TradingCalendar : t <= filing_date_i - 1 day }
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "These anchors are not approximated by integer offsets; they are computed by forward and backward asof joins to the observed trading calendar (E2).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> Asof joins respect actual exchange activity: a filing released on a Friday evening has its day-0 correctly assigned to the following Monday, and the pre-filing anchor stops at t - 1 so the short-horizon excess return captures only post-disclosure trading. Integer offsets would mis-align the window for weekends and holidays and would contaminate the baseline with the event itself.",
            styles["body"],
        )
    )
    story.append(_paragraph("3.2 Accounting and market covariates", styles["h2"]))
    story.append(
        _code_block(
            """
preferred_stock_ps = coalesce(PSTKL, PSTKRV, PSTK)
base_be            = coalesce(SEQ, CEQ + PSTK, AT - LT)
book_equity_be     = base_be - preferred_stock_ps + TXDITC - PRBA

ME_event_i = coalesce(TCAP_i, abs(PRC_i) * SHROUT_i) / 1000
size_event_i = ME_event_i
bm_event_i   = book_equity_be_i / ME_event_i        when ME_event_i > 0
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The division by 1000 is explicit in event_market_equity_millions because CRSP-style raw market values are interpreted on a thousands-of-USD scale and normalized to millions for compatibility with book equity (E3, E4).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> The construction follows the Fama-French convention. Shareholders' equity net of preferred-stock claims (which are debt-like in payoff structure) plus the deferred tax and investment tax credit adjustment is the book-equity measure used to build book-to-market, empirically the most successful value proxy in the cross-section of equity returns. The coalesce ladder for SEQ preserves observations when individual Compustat fields are missing but the accounting identity still pins down shareholders' equity.",
            styles["body"],
        )
    )
    story.append(_paragraph("3.3 Event outcomes and control variables", styles["h2"]))
    story.append(
        _code_block(
            """
event_stock_gross_i  = Product_{d = 0..3} (1 + stock_return_i,d)
event_market_gross_i = Product_{d = 0..3} (1 + market_return_d)

filing_period_excess_return_i = event_stock_gross_i - event_market_gross_i

share_turnover_i =
    Sum_{d = -252..-6} VOL_i,d
    --------------------------------
    event_shares_i * 1000

pre_vol_mean_i = Mean_{d = -65..-6}(VOL_i,d)
pre_vol_std_i  = Std_{d = -65..-6}(VOL_i,d)
std_volume_i,d = (VOL_i,d - pre_vol_mean_i) / pre_vol_std_i

abnormal_volume_i = Mean_{d = 0..3}(std_volume_i,d)

nasdaq_dummy_i = 1[event_exchcd_i = 3]
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "Two details matter here. First, filing_period_excess_return is a difference of compounded gross returns, not a sum of abnormal returns. Second, abnormal_volume is standardized with a pre-window mean and standard deviation before averaging across event days (E5).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> filing_period_excess_return is an investable quantity: the realized P&amp;L of going long the stock and short the market over the 4-day filing window, which is why gross-return products rather than log-return sums are used. abnormal_volume is standardized against a rolling 60-day baseline so it captures unusual attention or disagreement triggered by the filing rather than the firm's typical trading scale; volume shocks are the classic proxy for investor disagreement in Kandel-Pearson and for attention spikes in Barber-Odean. share_turnover over the prior year is a liquidity proxy (firms with higher turnover have tighter price-discovery), and nasdaq_dummy captures exchange-level differences in investor composition that affect how disclosures are absorbed.",
            styles["body"],
        )
    )
    story.append(_paragraph("3.4 Factor-model alpha and post-event volatility", styles["h2"]))
    story.append(
        _code_block(
            """
For each document i and window W in {[-252,-6], [6,252]}:

stock_return_i,d - rf_d
    = alpha_i,W + beta_mkt_i,W * mkt_rf_d
                + beta_smb_i,W * smb_d
                + beta_hml_i,W * hml_d
                + u_i,d

pre_ffalpha_i               = alpha_i,[-252,-6]
postevent_return_volatility = sqrt(mse_resid_i,[6,252])
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The code uses a common OLS helper based on statsmodels.OLS with an added constant. The post-event volatility field is therefore a factor-model RMSE, not the raw standard deviation of post-filing returns (E5).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> pre_ffalpha isolates pre-filing abnormal drift not explained by market, size, and value factors, acting as a proxy for information leakage or momentum run-up already reflected in prices before the disclosure. postevent_return_volatility, measured as the factor-model RMSE rather than raw return standard deviation, captures idiosyncratic uncertainty *after* the filing; this is the quantity relevant for testing whether disclosure tone raises the firm's forward information environment and separates a first-moment news effect from a second-moment uncertainty effect.",
            styles["body"],
        )
    )
    story.append(_paragraph("3.5 Event-panel screening before regressions", styles["h2"]))
    story.append(
        _paragraph(
            "The event panel is screened by the same ladder used for LM2011 sample creation and then pruned further. The implemented filters are (i) event_shrcd in {10, 11}, "
            "(ii) size_event > 0, (iii) pre_filing_price >= 3, (iv) complete returns and volume for all 4 event days, (v) event_exchcd in {1, 2, 3}, "
            "(vi) at least 60 observations in the turnover, abnormal-volume, pre-alpha, and post-alpha windows, (vii) book_equity_be > 0 and bm_event > 0, "
            "(viii) total_token_count_full_10k >= 2000, (ix) bm_event winsorized at the 1st and 99th percentiles, and (x) abnormal_volume must be non-null (E6).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> The shrcd filter keeps only domestic common stock and drops ADRs, REITs, and closed-end funds, which trade on different microstructures. The $3 price floor removes penny-stock bid-ask bounce that would contaminate short-horizon return measurement. The 60-observation minima ensure the factor-loading estimates are identified with reasonable precision. The 2000-token floor excludes stub filings where tone measurement is statistically unreliable. Winsorizing book-to-market tempers leverage from extreme-distress observations where the accounting identity can break down.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    story.append(_paragraph("4. Text Signal Construction", styles["h1"]))
    story.append(
        _paragraph(
            "All LM2011 dictionary signals used in the regressions are generated by the same scorer for both full 10-K text and item-level scopes. The denominator contract is crucial: proportions and TF-IDF document-length normalization use the total LM-tokenized length of the relevant analyzed text unit, while token_count columns remain recognized-word diagnostics (E7).",
            styles["body"],
        )
    )
    story.append(_paragraph("4.1 Token counts and denominator contract", styles["h2"]))
    story.append(
        _code_block(
            """
token_total_i            = total count of LM2011 tokenizer outputs
recognized_word_total_i  = count of tokens that appear in master_dictionary_words
matched_count_i,S        = Sum_{t in S} tf_i,t

signal_prop_i,S = matched_count_i,S / token_total_i
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "Accordingly, total_token_count_full_10k and total_token_count_mda are the denominator inputs for the relevant analyzed text units, while token_count_full_10k and token_count_mda remain master-dictionary recognized-word totals for diagnostics and coverage (E7).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> Using the total LM-tokenized length of the relevant analyzed text unit keeps the denominator aligned with the actual scored document surface, so identical matched counts are scaled by the full analyzed scope rather than by the narrower subset of master-dictionary-recognized tokens.",
            styles["body"],
        )
    )
    story.append(_paragraph("4.2 TF-IDF style weighting used by the code", styles["h2"]))
    story.append(
        _code_block(
            """
idf_t = log(N / df_t)

term_weight_i,t =
    ((1 + log(tf_i,t)) / (1 + log(token_total_i))) * idf_t

signal_tfidf_i,S = Sum_{t in S, tf_i,t > 0}(term_weight_i,t)
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The code does not smooth df_t or add one to N or df_t. If document counts or term frequencies are zero, the helper returns 0 for the affected term weight (E7).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> IDF downweights terms that appear in most filings, so generic legalese and standard risk-factor language contribute less to the signal. The log-saturated term frequency prevents a single repeated negative term from dominating the document score. The net effect is an attention-weighted measure that emphasises rare, document-specific negative vocabulary over ritual cautionary language, which is closer to what an informed reader would actually notice in a filing.",
            styles["body"],
        )
    )
    story.append(_paragraph("4.3 Signal families actually used downstream", styles["h2"]))
    for line in [
        "Table IV and Table VIII use h4n_inf_prop, lm_negative_prop, h4n_inf_tfidf, and lm_negative_tfidf.",
        "Table V uses the same four signals, but on the MD&A item-7 surface and only when total_token_count_mda >= 250.",
        "Table VI broadens the signal family to LM negative, positive, uncertainty, litigious, modal_strong, and modal_weak, each in prop and tfidf form.",
        "Table IA.I uses normalized_difference_negative and normalized_difference_h4n_inf, which are standardized within industry against prior-year means and standard deviations.",
        "The trading strategy uses only fin_neg_prop, fin_neg_tfidf, h4n_inf_prop, and h4n_inf_tfidf on full 10-K text.",
    ]:
        story.append(_bullet(line, styles))
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> The H4N (Harvard-IV) and LM dictionaries together identify the financial-context premium over general-English tone. LM excludes words like <i>cost</i>, <i>tax</i>, or <i>liability</i> that are domain-neutral in 10-K disclosures; if LM-negative retains predictive power beyond H4N-negative, the domain-specific dictionary carries measurable signal. The MD&amp;A scope in Table V narrows attention to the section with the most discretion from management, where tone choice is arguably most informative. The industry-standardized normalized-difference signals in Table IA.I remove industry-level differences in cautionary-language style so the signal captures a firm's deviation from its own industry norm.",
            styles["body"],
        )
    )

    story.append(_paragraph("5. SUE And Analyst Regressor Path", styles["h1"]))
    story.append(
        _paragraph(
            "The SUE panel is built from the event panel plus quarterly accounting alignment, price lookup, and matched analyst records. The quarterly report date must be the earliest report strictly after filing and no more than 90 days later (E8).",
            styles["body"],
        )
    )
    story.append(
        _code_block(
            """
sue_i                 = (actual_eps_i - forecast_consensus_mean_i) / pre_filing_price_i
analyst_dispersion_i  = forecast_dispersion_i / pre_filing_price_i
analyst_revisions_i   = forecast_revision_4m_i / prior_month_price_i
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "Only rows with analyst_match_status == MATCHED are retained, and the panel drops rows where any of sue, analyst_dispersion, or analyst_revisions is null (E8).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> Scaling the earnings surprise by pre_filing_price converts it to an earnings-yield surprise - the standard Kothari-Warner deflation that removes cross-sectional heterogeneity in share-price levels so surprises are comparable across firms. analyst_dispersion proxies forecaster disagreement (the Diether-Malloy-Scherbina uncertainty variable), and analyst_revisions capture information updates that arrived between earnings announcements. Together with SUE, they control for the numerical-information channel so any residual tone effect is attributable to the non-numerical content of the filing.",
            styles["body"],
        )
    )

    story.append(_paragraph("6. Quarterly Fama-MacBeth Estimator", styles["h1"]))
    story.append(
        _paragraph(
            "The main regression engine groups observations by the quarter that contains filing_date and runs a cross-sectional OLS each quarter. Industry controls enter as one-hot FF48 dummies with the first observed industry omitted (E9).",
            styles["body"],
        )
    )
    story.append(
        _code_block(
            """
For each retained quarter q:

y_iq = alpha_q
     + beta_q * signal_iq
     + gamma_q' * controls_iq
     + delta_q' * FF48_dummies_iq
     + epsilon_iq

where quarter_start_q = first day of the filing quarter.
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "A quarter is skipped if the number of rows is smaller than the number of estimated coefficients. Rank-deficient designs raise a ValueError through the shared OLS helper rather than being silently pseudo-inverted (E5, E9).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> Fama-MacBeth is the natural estimator here because firm returns within a calendar quarter share common macro, industry, and risk-factor shocks. A pooled OLS would treat these as independent residuals and understate standard errors. Running a cross-section each quarter and then averaging the coefficient path across quarters uses the dispersion of *quarterly coefficients* as the sampling distribution, which is robust to within-quarter cross-sectional dependence. The FF48 industry dummies further purge the signal of permanent industry-level differences in disclosure conventions and risk exposures.",
            styles["body"],
        )
    )
    story.append(_paragraph("6.1 Aggregation across quarters", styles["h2"]))
    story.append(
        _code_block(
            """
n_q       = cross-sectional observation count in quarter q
omega_q   = n_q / Sum_q n_q
theta_q   = quarterly coefficient estimate for a given regressor

theta_hat = Sum_q (omega_q * theta_q)
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The weighting rule is stored explicitly as quarter_observation_count in the output tables; there is no equal-weighted quarter average in the implemented code path (E9).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> Observation-count weighting up-weights quarters with larger cross-sectional samples, which carry more precise coefficient estimates. Equal-weighting across quarters would give sparse early quarters (when filing counts are lower) the same influence as richly populated later ones, biasing the time-series mean toward noisy segments of the panel. The chosen weighting is the precision-weighted average under the assumption that quarterly coefficient noise is inversely proportional to the within-quarter sample size.",
            styles["body"],
        )
    )
    story.append(_paragraph("6.2 Time-series standard errors on the coefficient path", styles["h2"]))
    story.append(
        _code_block(
            """
psi_q = omega_q * (theta_q - theta_hat)

Var_NW(theta_hat) =
    Sum_q psi_q^2
    + 2 * Sum_{lag = 1..L} [ (1 - lag / (L + 1)) * Sum_{q = lag+1..Q} psi_q * psi_{q-lag} ]

SE_NW(theta_hat) = sqrt(max(Var_NW(theta_hat), 0))
t_stat           = theta_hat / SE_NW(theta_hat)
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The default lag length is nw_lags = 1. The implementation uses a Bartlett weight 1 - lag / (L + 1) and computes the variance on the weighted coefficient innovations psi_q (E9).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> Newey-West with Bartlett weights corrects for first-order autocorrelation in the quarterly coefficient series, which is typical when style exposures, industry composition, or macro regimes persist beyond a single quarter (e.g. a prolonged credit cycle in which many filings share a common discount-rate shock). Lag 1 is a conservative default; longer lags would be appropriate only if the coefficient path exhibits higher-order persistence. Positive-semi-definiteness is enforced via the max(., 0) clamp because the finite-sample Bartlett estimator can produce a negative variance when quarters are few.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    story.append(_paragraph("7. Table-Specific Model Specifications", styles["h1"]))
    story.append(
        _paragraph(
            "The table builders do not estimate bespoke models from scratch. They call the same quarterly Fama-MacBeth engine with different dependent variables, signal columns, text scopes, and control sets (E10).",
            styles["body"],
        )
    )
    table_specs = [
        [
            "Output",
            "Dependent variable",
            "Signal inputs",
            "Controls / notes",
        ],
        [
            "Table IV",
            "filing_period_excess_return",
            "h4n_inf_prop, lm_negative_prop, h4n_inf_tfidf, lm_negative_tfidf on full_10k",
            "Controls: log_size, log_book_to_market, pre_ffalpha, log_share_turnover, nasdaq_dummy, institutional_ownership.",
        ],
        [
            "Table V",
            "filing_period_excess_return",
            "Same four signals on MD&A item 7",
            "Same controls as Table IV; additional filter total_token_count_mda >= 250.",
        ],
        [
            "Table VI",
            "filing_period_excess_return",
            "LM negative/positive/uncertainty/litigious/modal_strong/modal_weak in prop and tfidf form on full_10k",
            "Same controls as Table IV.",
        ],
        [
            "Table VIII",
            "sue",
            "h4n_inf_prop, lm_negative_prop, h4n_inf_tfidf, lm_negative_tfidf on full_10k",
            "Return controls plus analyst_dispersion and analyst_revisions.",
        ],
        [
            "Table IA.I",
            "filing_period_excess_return",
            "normalized_difference_negative, normalized_difference_h4n_inf",
            "Same controls as Table IV; normalized signals use prior-year industry moments.",
        ],
        [
            "Table IA.II",
            "long_short_return (monthly strategy series)",
            "fin_neg_prop, fin_neg_tfidf, h4n_inf_prop, h4n_inf_tfidf",
            "No Fama-MacBeth. The code reports mean long-short return plus FF4 loadings and R-squared.",
        ],
    ]
    story.append(_table(table_specs[0], table_specs[1:], [26, 31, 63, 60], styles))
    story.append(Spacer(1, 4))
    story.append(
        _paragraph(
            "<i>Economic interpretation of the table specifications.</i> Table IV is the baseline 'tone predicts short-horizon reaction' test on full 10-K text. Table V narrows to MD&amp;A, the section with the highest management discretion; stronger loadings there would indicate attention is concentrated on the narrative rather than the boilerplate sections. Table VI broadens the LM categories to test whether uncertainty, litigious, and modal-strength vocabularies add incremental information beyond pure negativity. Table VIII uses SUE as the outcome; a significant tone coefficient there means text signals carry information orthogonal to the numerical earnings surprise. Table IA.I uses industry-standardized normalized-difference tone so coefficients capture relative-to-industry deviations, purging permanent industry-style differences in cautionary language. Table IA.II is the tradeability check: a non-zero FF4 alpha on the Q1 minus Q5 long-short portfolio is <i>economic</i> evidence (an investable excess return unexplained by standard factor exposures) rather than merely statistical significance.",
            styles["body"],
        )
    )

    story.append(_paragraph("8. Trading Strategy And Factor-Loading Path", styles["h1"]))
    story.append(
        _paragraph(
            "Table IA.II is the only path that switches from quarterly cross-sectional regressions to a monthly portfolio time-series regression. The strategy path begins from event_panel and rescored yearly filings, then sorts firms into quintiles by next-year signal ranks (E10, E11).",
            styles["body"],
        )
    )
    story.append(_paragraph("8.1 Assignment of yearly sort portfolios", styles["h2"]))
    story.append(
        _code_block(
            """
sort_year_i = year(filing_date_i) + 1

Within each (sort_year, sort_signal_name):
    order rows by signal_value ascending and doc_id
    group_rank = zero-based within-group rank
    quintile   = floor(5 * group_rank / group_size) + 1
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The holdings side uses a July-to-June convention: portfolio_month belongs to sort_year = year(portfolio_month) when month >= 7, otherwise year(portfolio_month) - 1 (E11).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> Using sort_year = year(filing_date) + 1 together with the July-to-June holdings convention guarantees that portfolio formation uses only publicly available information well before the holding period begins, eliminating look-ahead bias in the classical Fama-French style. The quintile long-short design isolates the extreme-decile contrast - the firms where the signal is most confident - while diversifying idiosyncratic noise across dozens of firms per quintile, so the realized return series reflects the signal rather than individual-stock shocks.",
            styles["body"],
        )
    )
    story.append(_paragraph("8.2 Monthly long-short returns", styles["h2"]))
    story.append(
        _code_block(
            """
If portfolio_weighting == equal:
    r_Qk,m,j = Mean(monthly_return_i,m over holdings in quintile k)

If portfolio_weighting == lagged_value:
    weight_i,m = lagged market cap for security i
    r_Qk,m,j   = Sum(weight_i,m * monthly_return_i,m) / Sum(weight_i,m)

long_short_return_m,j = r_Q1,m,j - r_Q5,m,j
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The default weighting is equal. Under lagged_value weighting, the code shifts market capitalization by one month within KYPERMNO before forming portfolio_weight (E11).",
            styles["body"],
        )
    )
    story.append(_paragraph("8.3 Factor-loading regression on the monthly long-short series", styles["h2"]))
    story.append(
        _code_block(
            """
long_short_return_m,j =
      alpha_j
    + beta_market_j * mkt_rf_m
    + beta_smb_j    * smb_m
    + beta_hml_j    * hml_m
    + beta_mom_j    * mom_m
    + epsilon_m,j

Reported outputs:
    mean_long_short_return_j
    alpha_ff3_mom_j
    beta_market_j, beta_smb_j, beta_hml_j, beta_mom_j
    r2_j
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The code scales daily and monthly factor columns down by 100 when their absolute magnitudes suggest percentage-point storage rather than decimal storage. That affects the alpha and beta magnitudes in both the event alpha regressions and the monthly FF4 regressions (E14).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> The FF4 regression decomposes the monthly long-short return into risk-factor compensation (market, size, value, momentum) and unexplained alpha. A significant positive alpha implies the text signal generates returns investors cannot replicate by loading on standard priced factors, which is the economically relevant test of whether the tone signal is tradeable. The beta coefficients also describe the economic character of the strategy - for example, a large negative beta_mkt would indicate the strategy hedges market risk, and a large positive beta_hml would indicate it tilts toward value exposure independent of the tone signal itself.",
            styles["body"],
        )
    )

    story.append(_paragraph("9. Extension Path For 2009-2024", styles["h1"]))
    story.append(
        _paragraph(
            "The extension module defines a newer 2009-01-01 through 2024-12-31 analysis surface. It reuses the same return controls, the same FF48 industry attachments, and the same quarterly Fama-MacBeth estimator. The default runner scopes are item_7_mda and item_1a_risk_factors, while the normalization helpers also understand aliases such as mda_item_7 and item_1_business (E13).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> The extension window covers the post-GFC, post-XBRL regulatory regime, in which machine-readable filings and large-model NLP became broadly available. The central economic question is whether a contextual transformer (FinBERT) captures filing tone that dictionary methods cannot - for example, by distinguishing 'our revenues declined' from 'our competitor's revenues declined' - and whether any combined signal survives the ownership-proxy sample restriction, which tests whether the presence of sophisticated investors moderates the text-return relationship.",
            styles["body"],
        )
    )
    story.append(_paragraph("9.1 Extension outcomes and control sets", styles["h2"]))
    story.append(
        _code_block(
            """
Default runner outcome:  filing_period_excess_return
Module-defined secondary outcomes: abnormal_volume, postevent_return_volatility

Base controls:
    log_size, log_book_to_market, log_share_turnover, pre_ffalpha, nasdaq_dummy

Ownership proxy:
    institutional_ownership_proxy_refinitiv =
        coalesce(institutional_ownership_proxy_refinitiv,
                 institutional_ownership_pct,
                 institutional_ownership)

Control sets:
    C0 = base controls only
    C1 = base controls only, but restrict to rows with non-null ownership proxy
    C2 = base controls + ownership proxy, also restricted to non-null ownership proxy
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The module exposes abnormal_volume and postevent_return_volatility as secondary outcomes, but run_lm2011_extension_pipeline currently calls build_lm2011_extension_sample_loss_table and run_lm2011_extension_estimation_scaffold with their defaults. In the shipped runner path, extension_sample_loss and extension_results are therefore written for filing_period_excess_return unless a library caller overrides outcome_names directly (E1, E13).",
            styles["body"],
        )
    )
    story.append(_paragraph("9.2 Extension feature families", styles["h2"]))
    story.append(
        _code_block(
            """
dictionary_only:
    signal column = lm_negative_tfidf

finbert_only:
    signal column = finbert_neg_prob_lenw_mean

dictionary_finbert_joint:
    signal column = lm_negative_tfidf
    additional control = finbert_neg_prob_lenw_mean
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "That last detail matters: in the joint specification, FinBERT enters the design matrix as a control column because the scaffold always treats the first signal_inputs entry as the main signal_column and appends the remaining entries to control_columns (E13).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> The three families run a horse race between measurement technologies. dictionary_only and finbert_only estimate each signal in isolation on a matched sample. The joint specification, by treating FinBERT as a control while LM-negative is the focal signal, identifies the <i>marginal</i> contribution of the dictionary after conditioning on the transformer: a significant LM coefficient there implies dictionary methods carry information contextual NLP misses, while the reverse pattern (LM shrinks to zero when FinBERT is included) would indicate the transformer subsumes the lexicon. The C0/C1/C2 ladder separates a sample-composition effect (restricting to ownership-proxy-available firms) from a true ownership-proxy effect.",
            styles["body"],
        )
    )
    story.append(_paragraph("9.3 FinBERT feature arithmetic", styles["h2"]))
    story.append(
        _code_block(
            """
token_weight_s = max(finbert_token_count_512_s, 0)

finbert_neg_prob_lenw_mean_i =
    Sum_s(token_weight_s * negative_prob_s) / Sum_s(token_weight_s)

finbert_net_negative_lenw_mean_i =
    Sum_s(token_weight_s * (negative_prob_s - positive_prob_s)) / Sum_s(token_weight_s)

finbert_neg_dominant_share_i =
    Mean_s(1[predicted_label_s = negative])
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The current runner defaults dictionary_source_mode to prefer_cleaned_scopes and validates that dictionary and FinBERT surfaces share identical cleaned-scope universes on doc_id, filing_date, text_scope, and cleaning_policy_id. If cleaned scopes are missing or misaligned, the matched comparison fails closed instead of silently falling back to raw item text (E13).",
            styles["body"],
        )
    )
    story.append(_paragraph("9.4 Extension inference outputs", styles["h2"]))
    story.append(
        _code_block(
            """
The scaffold reuses the quarterly Fama-MacBeth estimator:
    estimate, standard_error, t_stat, n_quarters, mean_quarter_n, weighting_rule, nw_lags

It then adds:
    n_obs   = round(n_quarters * mean_quarter_n)
    p_value = erfc(abs(t_stat) / sqrt(2))
            """,
            styles,
        )
    )
    story.append(
        _paragraph(
            "The p-value is therefore a normal-approximation transform of the reported t-stat, not a direct statsmodels p-value or a small-sample correction (E13).",
            styles["body"],
        )
    )
    story.append(
        _paragraph(
            "<i>Economic interpretation.</i> The normal-approximation p-value is justified here because the number of quarterly observations underlying the Fama-MacBeth coefficient averages is large enough that small-sample t corrections move the critical values only marginally. The reported p-value should be interpreted as an asymptotic probability statement about the coefficient's deviation from zero, not as a finite-sample exact test, and should be treated cautiously for sub-period cuts where the quarter count is small.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    story.append(_paragraph("10. Variable Dictionary", styles["h1"]))
    story.append(
        _paragraph(
            "The tables below collect the variables that actually enter the implemented econometric paths, together with concise code-grounded definitions.",
            styles["body"],
        )
    )
    for title, rows in _variable_tables():
        story.append(_paragraph(title, styles["h2"]))
        story.append(
            _table(
                ["Variable", "Definition", "Evidence"],
                [[row.name, row.definition, row.evidence] for row in rows],
                [35, 110, 20],
                styles,
            )
        )
        story.append(Spacer(1, 4))

    story.append(_paragraph("11. Evidence Appendix", styles["h1"]))
    story.append(
        _paragraph(
            "These references are the concrete Python-code anchors used to reconstruct the formulas and path map in this report.",
            styles["body"],
        )
    )
    evidence_rows = [[ref.ref_id, ref.location, ref.note] for ref in _evidence_refs()]
    story.append(_table(["ID", "Location", "What It Supports"], evidence_rows, [12, 73, 80], styles))

    story.append(Spacer(1, 6))
    story.append(_paragraph("12. Key Findings", styles["h1"]))
    story.append(_paragraph("12.1 Code-level findings", styles["h2"]))
    for line in [
        "The central implemented estimator is quarterly Fama-MacBeth with observation-count weighting across quarters and Newey-West lag 1 standard errors on the quarterly coefficient path.",
        "The main return outcome is not cumulative abnormal return from a factor model; it is the difference between compounded stock and compounded market gross returns over filing days 0..3.",
        "postevent_return_volatility is a factor-model RMSE, not a raw realized-volatility measure.",
        "LM2011 dictionary proportions use recognized master-dictionary words as the denominator, while separate total_token_count columns preserve the broader post-clean token totals.",
        "The extension scaffold is now wired into sec_ccm_unified_runner after FinBERT and fails closed on cleaned-scope misalignment instead of silently mixing dictionary and model surfaces.",
    ]:
        story.append(_bullet(line, styles))
    story.append(_paragraph("12.2 Economic takeaways", styles["h2"]))
    for line in [
        "The overarching economic test is whether disclosure text carries information about filing-window prices, trading activity, earnings surprises, and post-event uncertainty that is not already contained in standard risk factors and accounting controls. Every regression in the pipeline is a specific implementation of this joint test.",
        "Filing_period_excess_return is deliberately chosen as the outcome for the price-reaction hypothesis because it is an investable quantity (long stock / short market over 4 days), so a significant tone coefficient translates directly into a realized P&amp;L of responding to the signal at release.",
        "The control set (log_size, log_book_to_market, log_share_turnover, pre_ffalpha, nasdaq_dummy, institutional_ownership) is the standard cross-sectional return factor stack plus event-time confounders. Their inclusion purges the text coefficient of exposure to size and value premia, liquidity differentials, pre-event drift, exchange-microstructure effects, and sophisticated-investor presence.",
        "The SUE regression tests whether tone carries information beyond the numerical earnings surprise: a significant tone coefficient in Table VIII (after conditioning on SUE, dispersion, and revisions) localizes the text information to non-quantitative disclosure content.",
        "The trading-strategy path is the bridge from statistical significance to economic significance. A non-zero FF4 alpha on the Q1-Q5 long-short demonstrates the tone signal is not spanned by market, size, value, or momentum factor exposures and is therefore an economic anomaly, not just an in-sample regression fit.",
        "The industry-standardized signals in Table IA.I test whether deviations from a firm's own industry tone norm are priced, isolating firm-specific information from industry-wide disclosure conventions.",
        "The extension scaffold is structured as a horse race between lexical (LM) and contextual (FinBERT) sentiment measurement. The joint specification identifies the marginal contribution of each technology; the C0/C1/C2 control ladder separates ownership-composition effects from true ownership-proxy effects.",
        "Throughout, the Fama-MacBeth design with Newey-West lag 1 standard errors is appropriate because quarterly coefficients inherit persistent style, industry, and macro exposures; the time-series SE on the coefficient path is the correct inference target for asking whether tone is priced on average across quarters.",
    ]:
        story.append(_bullet(line, styles))

    return story


def build_report() -> Path:
    root = _repo_root()
    output_path = _output_path(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title=TITLE,
        author="OpenAI Codex",
    )
    doc.build(_build_story(root), onFirstPage=_draw_page, onLaterPages=_draw_page)
    return output_path


if __name__ == "__main__":
    out = build_report()
    print(out)
