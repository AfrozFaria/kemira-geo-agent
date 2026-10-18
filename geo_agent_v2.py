import json
from datetime import datetime
from urllib.parse import urlparse, urljoin, urldefrag

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()   # reads OPENAI_API_KEY from the .env file

client = OpenAI()
MODEL = "gpt-4o-mini"          # agent model; try a stronger one from the OpenAI docs
REVIEW_MODEL = "gpt-4o-mini"   # reviewer model; a stronger model helps most here
MAX_STEPS = 25
MIN_PAGES = 3
MAX_LINKS_SHOWN = 60
ALLOWED_DOMAIN = "kemira.com"          # guardrail: only this site can be fetched
START_URL = "https://www.kemira.com"
SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
                   ".zip", ".xlsx", ".docx", ".pptx", ".mp4")


def normalize(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")


inspected_urls = set()                  # pages actually inspected (after redirects)
allowed_urls = {normalize(START_URL)}   # guardrail: homepage + links seen on pages
assistant_answers = []                  # evidence for the reviewer
page_evidence = []                      # evidence for the reviewer


def on_allowed_domain(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == ALLOWED_DOMAIN or host.endswith("." + ALLOWED_DOMAIN)


def useful_link(url: str) -> bool:
    path = urlparse(url).path.lower()
    return (on_allowed_domain(url)
            and not path.endswith(SKIP_EXTENSIONS)
            and "/app/uploads/" not in path
            and "?" not in url)


def link_rank(url: str):
    path = urlparse(url).path.strip("/")
    depth = len(path.split("/")) if path else 0
    return (depth, url)   # shallow pages (main sections) first


# ---------- Tools ----------
def ask_ai_assistant(question: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    answer = r.choices[0].message.content
    assistant_answers.append({"question": question, "answer": answer})
    return {"question": question, "answer": answer}


def collect_types(node, found):
    """Find @type values anywhere in JSON-LD, including inside @graph."""
    if isinstance(node, list):
        for n in node:
            collect_types(n, found)
    elif isinstance(node, dict):
        t = node.get("@type")
        if t:
            found.extend(str(x) for x in (t if isinstance(t, list) else [t]))
        for v in node.values():
            if isinstance(v, (dict, list)):
                collect_types(v, found)


def inspect_page(url: str) -> dict:
    if not on_allowed_domain(url):
        return {"error": f"Blocked: only {ALLOWED_DOMAIN} pages are allowed."}
    if normalize(url) not in allowed_urls:
        return {"error": "Blocked: this URL was not found in any 'links' list. "
                         "Choose a URL from the links returned by inspect_page."}
    try:
        r = requests.get(url, timeout=15, allow_redirects=True,
                         headers={"User-Agent": "GEO-audit-prototype/0.1"})
    except requests.RequestException as e:
        return {"error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")

    # Structured data (JSON-LD), including nested @graph entries
    schema_types, ld_blocks = [], 0
    for s in soup.find_all("script", type="application/ld+json"):
        ld_blocks += 1
        try:
            collect_types(json.loads(s.string or ""), schema_types)
        except json.JSONDecodeError:
            schema_types.append("invalid JSON-LD")
    schema_types = sorted(set(schema_types))
    has_org = any(t in ("Organization", "Corporation") for t in schema_types)

    meta = soup.find("meta", attrs={"name": "description"})

    # Links (collected before removing tags). All are allowed; the main ones are shown.
    all_links = {normalize(urljoin(r.url, a["href"]))
                 for a in soup.find_all("a", href=True)}
    all_links = {l for l in all_links if useful_link(l)}
    allowed_urls.update(all_links)
    links = sorted(all_links, key=link_rank)[:MAX_LINKS_SHOWN]

    # Remove scripts/styles so text checks only see visible content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]

    faq_signals = {
        "faqpage_schema": "FAQPage" in schema_types,
        "details_elements": len(soup.find_all("details")),
        "faq_headings": sum(1 for h in headings
                            if "faq" in h.lower() or "frequently asked" in h.lower()),
        "question_headings": sum(1 for h in headings if h.endswith("?")),
    }

    final_url = normalize(r.url)
    inspected_urls.add(final_url)   # a redirect counts as the page it lands on
    word_count = len(text.split())
    h1 = [h.get_text(strip=True) for h in soup.find_all("h1")][:3]
    h2 = [h.get_text(strip=True) for h in soup.find_all("h2")][:10]
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    result = {
        "url": url,
        "final_url": final_url,
        "redirected": final_url != normalize(url),
        "status": r.status_code,
        "title": title,
        "meta_description": meta.get("content") if meta else None,
        "h1": h1,
        "h2": h2,
        "json_ld_blocks": ld_blocks,
        "schema_types": schema_types,
        "has_organization_schema": has_org,
        "faq_signals": faq_signals,
        "visible_word_count": word_count,
        "note": ("Very little visible text: the page may be rendered with "
                 "JavaScript, so findings may be incomplete.") if word_count < 150 else "",
        "links_found": len(all_links),
        "links": links,                 # before the excerpt, so truncation cuts the excerpt
        "text_excerpt": text[:1000],
    }

    page_evidence.append({
        "url": final_url, "title": title, "h1": h1, "h2": h2,
        "json_ld_blocks": ld_blocks, "schema_types": schema_types,
        "has_organization_schema": has_org, "faq_signals": faq_signals,
        "visible_word_count": word_count, "text_excerpt": text[:500],
    })
    return result


TOOL_FUNCS = {"ask_ai_assistant": ask_ai_assistant, "inspect_page": inspect_page}

TOOLS = [
    {"type": "function", "function": {
        "name": "ask_ai_assistant",
        "description": "Ask an AI assistant a question to see how it describes a company.",
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "inspect_page",
        "description": "Fetch a kemira.com page and return its title, headings, "
                       "structured data types, FAQ signals, a text excerpt and a list "
                       "of links found on the page. Only the homepage or URLs from a "
                       "previous 'links' list are allowed.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
]

# Reference facts. Verify against kemira.com and extend before relying on them.
FACTS = """
Reference facts (from Kemira's public information; verify and extend).
This list is NOT exhaustive: anything not listed here is "not confirmed".
- Global leader in sustainable chemistry for water-intensive industries
- More than 100 years of history
- 2025 revenue EUR 2.8 billion (reported); 4,911 employees
- Listed on Nasdaq Helsinki
- Customers: municipal and industrial water treatment companies, fiber industry
"""

SYSTEM = """You are a GEO (Generative Engine Optimization) analyst for a corporate
communications team. Goal: find out how AI assistants describe Kemira and how
kemira.com could be made easier for AI systems to read and cite.

Process:
1. Ask the assistant 4-6 questions about Kemira (what it does, sustainability,
   customers, competitors, size).
2. Inspect at least 3 different kemira.com pages. Start with https://www.kemira.com,
   then choose real pages from the "links" returned by inspect_page. Include a
   sustainability page and a company/about page if the links list contains one.
3. Compare each assistant answer against the reference facts and the website.
4. Write a final report with: Findings (right / wrong / not confirmed / missing),
   Website assessment, and a prioritized list of 5 fixes.

Rules:
- Use only public information, no personal data.
- Compare every assistant answer against the reference facts. Flag any mismatch,
  especially numbers, explicitly as WRONG.
- If an assistant answer includes facts that are not in the reference facts (for
  example other customer segments or competitors), mark them "not confirmed by
  reference facts", never "Right".
- Only inspect URLs taken from the "links" returned by inspect_page (the homepage
  is the only exception). Never guess URLs.
- Make no claim about the website's financial data, content quality or
  accessibility unless inspect_page output shows it. Cite the page URL for every
  website claim. If a page has a note about little visible text, say the page
  could not be fully assessed.
- For each inspected page, report the schema types found and whether
  has_organization_schema is true. If json_ld_blocks > 0, structured data exists.
  Only claim a page lacks schema, an Organization entity or an FAQ if the
  inspect_page output for that page shows it.
- Fixes must be actions a communications/web team can take on kemira.com, such as
  structured data, FAQ content, and making correct facts prominent, consistent and
  machine-readable. Do not suggest publishing competitor lists.
- When an assistant answer is wrong, the fix is to make the correct fact prominent
  and structured on kemira.com, not to "correct" the website.
- Mark claims you could not verify as "unverified". Do not invent product names.
- The report is a draft for human review, not a final decision.
- Stop after the report.
""" + FACTS

REVIEW_SYSTEM = """You are a strict fact-checking editor for a corporate communications
team. You receive a DRAFT report, the reference facts, the raw assistant answers and
the page evidence collected by the audit. Rewrite the report so that it follows these
rules:
1. Keep the same structure (Findings, Website assessment, Prioritized fixes).
2. Any assistant claim that is not in the reference facts must be labelled
   "not confirmed", never "Right". Label a claim "Right" only if it matches the
   reference facts. An answer that adds facts beyond the reference facts (for
   example extra customer segments, or sustainability details) is "not confirmed".
3. Remove, or mark "unverified", any statement about the website that the page
   evidence does not support. Cite the page URL for each website claim.
4. Remove recommendations that a communications/web team cannot act on and any fix
   not tied to a finding. Fixes should be about making correct facts prominent,
   consistent and machine-readable on kemira.com.
5. State which pages were inspected and that the sample is small.
6. Describe pages only with the fields in the page evidence (URL, title, headings,
   schema types, FAQ signals, word count). Do not describe page content or claim
   differences between pages that those fields do not show.
7. Delete any fix that asks the company to publish competitor information, and any
   fix that is not tied to a finding in the report.
8. End with "Review notes" listing only changes that differ between the draft and
   your version.
Do not add new facts."""


def review(draft: str) -> str:
    user = ("REFERENCE FACTS:\n" + FACTS +
            "\n\nASSISTANT ANSWERS:\n" + json.dumps(assistant_answers, indent=1) +
            "\n\nPAGE EVIDENCE:\n" + json.dumps(page_evidence, indent=1) +
            "\n\nDRAFT REPORT:\n" + draft)
    r = client.chat.completions.create(
        model=REVIEW_MODEL,
        messages=[{"role": "system", "content": REVIEW_SYSTEM},
                  {"role": "user", "content": user}],
    )
    return r.choices[0].message.content


# ---------- Agent loop ----------
messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Run the Kemira GEO audit now."},
]

draft_report = None

for step in range(MAX_STEPS):
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    msg = resp.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        if len(inspected_urls) < MIN_PAGES:
            messages.append({"role": "user", "content":
                f"You inspected only {len(inspected_urls)} page(s). Inspect at least "
                f"{MIN_PAGES} different kemira.com pages, chosen from the links "
                "returned by inspect_page, before writing the report."})
            continue
        draft_report = msg.content
        break

    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        print(f"[step {step+1}] {call.function.name}({args})")
        result = TOOL_FUNCS[call.function.name](**args)

        if call.function.name == "inspect_page":
            print("   ->", {k: result.get(k) for k in
                           ("final_url", "status", "json_ld_blocks", "schema_types",
                            "has_organization_schema", "faq_signals",
                            "visible_word_count", "links_found", "error")})

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)[:9000],
        })
else:
    print("Stopped: reached the step limit without a final report.")

# ---------- Review pass and saving ----------
if draft_report:
    print("\n[review] Fact-checking the draft against the collected evidence...")
    final_report = review(draft_report)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    header = "# Kemira GEO audit (AI-generated draft, needs human review)\n\n"
    with open(f"geo_report_{stamp}_draft.md", "w", encoding="utf-8") as f:
        f.write(header + draft_report)
    with open(f"geo_report_{stamp}.md", "w", encoding="utf-8") as f:
        f.write(header + final_report)
    print("\n=== GEO AUDIT REPORT (reviewed; AI-generated, needs human review) ===\n")
    print(final_report)
    print(f"\nSaved: geo_report_{stamp}.md (reviewed) and geo_report_{stamp}_draft.md")