"""Look and feel for the dashboard: colours, CSS and small HTML building blocks.

Everything here returns plain strings, so it can be tested without Streamlit. Colours that sit on the
page background come from Streamlit's own theme variables, so the dashboard follows light and dark mode.
"""

from __future__ import annotations

from html import escape
from typing import Dict, Iterable, Optional

LEVEL_COLORS = {"OK": "#2FA66A", "CAUTION": "#E0A800", "WARNING": "#F0832E", "CRITICAL": "#E5484D"}
TONE_COLORS = {"good": "#2FA66A", "info": "#5B8DEF", "warn": "#F0832E", "bad": "#E5484D"}
ACCENT = "#5B8DEF"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.01em; }
:root { --da-line: rgba(128,128,128,.22); --da-card: var(--secondary-background-color, rgba(128,128,128,.08)); }

.da-brand { display:flex; align-items:center; gap:.7rem; margin: .2rem 0 1.2rem 0; }
.da-logo { width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, #5B8DEF, #8A6CF0); color:#fff; font-weight:700; font-size:1.05rem; }
.da-brand-name { font-weight:650; font-size:1.05rem; line-height:1.1; }
.da-brand-sub { font-size:.76rem; opacity:.6; }

.da-page-title { font-size:1.75rem; font-weight:700; margin:0 0 .15rem 0; }
.da-page-sub { opacity:.65; margin:0 0 1.2rem 0; }

.da-card { background: var(--da-card); border:1px solid var(--da-line); border-radius:18px; padding:18px 20px; }
.da-hero { border-radius:22px; padding:24px 26px; border:1px solid var(--da-line); margin-bottom:1rem;
  background: linear-gradient(135deg, color-mix(in srgb, var(--da-tone) 16%, transparent), var(--da-card) 70%); }
.da-hero-top { display:flex; gap:1.4rem; align-items:center; flex-wrap:wrap; }
.da-score { font-size:3.6rem; font-weight:700; line-height:1; color: var(--da-tone); }
.da-score small { font-size:1rem; font-weight:500; opacity:.55; margin-left:.25rem; }
.da-hero-label { font-size:1.35rem; font-weight:650; }
.da-hero-sub { opacity:.65; font-size:.9rem; margin-top:.15rem; }
.da-chips { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.9rem; }
.da-chip { font-size:.78rem; padding:4px 11px; border-radius:999px; border:1px solid var(--da-line); opacity:.9; }

.da-bar { display:flex; height:12px; border-radius:999px; overflow:hidden; margin-top:1.1rem; background:var(--da-line); }
.da-bar span { display:block; height:100%; }
.da-legend { display:flex; gap:1.1rem; flex-wrap:wrap; margin-top:.55rem; font-size:.78rem; opacity:.8; }
.da-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.38rem; }

.da-kpi { background:var(--da-card); border:1px solid var(--da-line); border-radius:16px; padding:14px 16px;
  height:100%; border-left:4px solid var(--da-tone, var(--da-line)); }
.da-kpi-label { font-size:.7rem; letter-spacing:.07em; text-transform:uppercase; opacity:.6; font-weight:600; }
.da-kpi-value { font-size:1.85rem; font-weight:680; line-height:1.2; }
.da-kpi-sub { font-size:.76rem; opacity:.58; }

.da-insight { display:flex; gap:.85rem; background:var(--da-card); border:1px solid var(--da-line);
  border-radius:14px; padding:13px 16px; margin-bottom:.65rem; }
.da-bullet { flex:0 0 5px; border-radius:6px; background:var(--da-tone); }
.da-insight-title { font-weight:620; font-size:.95rem; }
.da-insight-text { opacity:.75; font-size:.86rem; margin-top:.1rem; }

.da-step { background:var(--da-card); border:1px solid var(--da-line); border-radius:16px; padding:16px 18px; height:100%; }
.da-step-n { width:26px; height:26px; border-radius:50%; background:var(--da-tone); color:#fff; font-weight:700;
  font-size:.82rem; display:inline-flex; align-items:center; justify-content:center; margin-bottom:.5rem; }
.da-step-title { font-weight:620; }
.da-step-text { opacity:.72; font-size:.86rem; margin-top:.15rem; }
.da-step code { font-size:.78rem; }

.da-status { display:inline-block; padding:5px 15px; border-radius:999px; color:#fff; font-weight:650; letter-spacing:.03em; }
.da-gauge { height:14px; border-radius:999px; background:var(--da-line); overflow:hidden; margin:.6rem 0 .25rem 0; }
.da-gauge span { display:block; height:100%; border-radius:999px; }
.da-live-risk { font-size:2.6rem; font-weight:700; line-height:1.1; }

section[data-testid="stSidebar"] { border-right:1px solid var(--da-line); }
div[data-testid="stSidebar"] .stRadio > div { gap:.15rem; }
div[data-testid="stSidebar"] .stRadio label { padding:.45rem .7rem; border-radius:10px; }
button[kind="primary"] { border-radius:11px; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { gap:.35rem; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.55rem 1rem; }
"""


def _tone(color: str) -> str:
    return f"--da-tone:{color}"


def level_color(name: str) -> str:
    return LEVEL_COLORS.get(name.upper(), ACCENT)


def risk_color(risk: float) -> str:
    if risk >= 75:
        return LEVEL_COLORS["CRITICAL"]
    if risk >= 50:
        return LEVEL_COLORS["WARNING"]
    if risk >= 25:
        return LEVEL_COLORS["CAUTION"]
    return LEVEL_COLORS["OK"]


def brand() -> str:
    return ('<div class="da-brand"><div class="da-logo">DA</div><div><div class="da-brand-name">Driver Assist</div>'
            '<div class="da-brand-sub">Drowsiness monitoring</div></div></div>')


def page_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="da-page-sub">{escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="da-page-title">{escape(title)}</div>{sub}'


def kpi_card(label: str, value: str, sub: str = "", color: Optional[str] = None) -> str:
    style = f' style="{_tone(color)}"' if color else ""
    sub_html = f'<div class="da-kpi-sub">{escape(sub)}</div>' if sub else ""
    return (f'<div class="da-kpi"{style}><div class="da-kpi-label">{escape(label)}</div>'
            f'<div class="da-kpi-value">{escape(str(value))}</div>{sub_html}</div>')


def level_bar(seconds_at_level: Dict[str, int]) -> str:
    total = sum(seconds_at_level.values())
    if total <= 0:
        return ""
    segs, legend = [], []
    for name in ("OK", "CAUTION", "WARNING", "CRITICAL"):
        secs = seconds_at_level.get(name, 0)
        color = LEVEL_COLORS[name]
        if secs:
            segs.append(f'<span style="width:{100 * secs / total:.2f}%;background:{color}" title="{name}: {secs}s"></span>')
        legend.append(f'<span><i class="da-dot" style="background:{color}"></i>{name.title()} {100 * secs / total:.0f}%</span>')
    return f'<div class="da-bar">{"".join(segs)}</div><div class="da-legend">{"".join(legend)}</div>'


def hero(score: int, label: str, color: str, subtitle: str, chips: Iterable[str], bar_html: str = "") -> str:
    chip_html = "".join(f'<span class="da-chip">{escape(c)}</span>' for c in chips)
    return (f'<div class="da-hero" style="{_tone(color)}"><div class="da-hero-top">'
            f'<div class="da-score">{score}<small>/100</small></div>'
            f'<div><div class="da-hero-label">{escape(label)}</div><div class="da-hero-sub">{escape(subtitle)}</div></div>'
            f'</div><div class="da-chips">{chip_html}</div>{bar_html}</div>')


def insight_card(tone: str, title: str, text: str) -> str:
    return (f'<div class="da-insight" style="{_tone(TONE_COLORS.get(tone, ACCENT))}"><div class="da-bullet"></div>'
            f'<div><div class="da-insight-title">{escape(title)}</div>'
            f'<div class="da-insight-text">{escape(text)}</div></div></div>')


def step_card(n: int, title: str, text: str, color: str = ACCENT) -> str:
    # text may contain a <code> snippet, so it is passed through as trusted HTML
    return (f'<div class="da-step" style="{_tone(color)}"><div class="da-step-n">{n}</div>'
            f'<div class="da-step-title">{escape(title)}</div><div class="da-step-text">{text}</div></div>')


def status_pill(text: str, color: str) -> str:
    return f'<span class="da-status" style="background:{color}">{escape(text)}</span>'


def live_panel(level: str, message: str, risk: float, activity: str) -> str:
    color = level_color(level)
    return (f'{status_pill(level, color)}'
            f'<div class="da-live-risk" style="color:{color}">{risk:.0f}<small style="font-size:1rem;opacity:.55"> risk</small></div>'
            f'<div class="da-gauge"><span style="width:{max(0, min(100, risk)):.0f}%;background:{color}"></span></div>'
            f'<div class="da-kpi-sub">{escape(message)}</div>'
            f'<div style="margin-top:.7rem;font-weight:600">{escape(activity)}</div>')
