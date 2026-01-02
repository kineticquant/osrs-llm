import streamlit as st
import requests
import os
import re
import time
import mwparserfromhell
import pandas as pd
from datasets import Dataset
import shutil
import json

WIKI_API_URL = "https://oldschool.runescape.wiki/api.php"
CONTACT_INFO = "email@na.com"
DATA_DIR = 'data/raw'
CLEAN_DIR = 'data/clean'
os.makedirs(CLEAN_DIR, exist_ok=True)


st.set_page_config(
    page_title="OSRS LLM Data Pipeline",
    page_icon="🧊",
    #layout="wide",
)

def sanitize_filename(name):
    # Remove or replace invalid Windows filename chars
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name.strip())
    name = re.sub(r'_+', '_', name) # Collapse multiple underscores
    return name if name else "unknown_page"


def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': f'OSRS-Wiki-Full-Scraper/1.0 (contact: {CONTACT_INFO}; GitHub osrs-llm project)'
    })
    return session


def total_pages(session):
    params = {"action": "query", "meta": "siteinfo", "siprop": "statistics", "format": "json"}
    response = session.get(WIKI_API_URL, params=params)
    response.raise_for_status()
    return response.json()["query"]["statistics"]["articles"]


def page_titles(session, table_placeholder, progress_df):
    titles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "apnamespace": 0,
        "apfilterredir": "nonredirects",
        "aplimit": 500
    }
    last_continue = {}
    while True:
        req_params = {**params, **last_continue}
        response = session.get(WIKI_API_URL, params=req_params)
        response.raise_for_status()
        data = response.json()
        batch_count = len(data["query"]["allpages"])
        for page in data["query"]["allpages"]:
            titles.append(page["title"])

        progress_df.loc[1, "Value"] = f"{len(titles)} / {progress_df.loc[0, 'Value']}"
        progress_df.loc[2, "Value"] = "In progress..."
        progress_df.loc[3, "Value"] = f"{len(titles)} titles fetched"
        table_placeholder.table(progress_df)

        if "continue" in data:
            last_continue = data["continue"]
            time.sleep(0.5)
        else:
            break
    return titles


def batch_extr(session, titles):
    results = {}
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "titles": "|".join(titles)
    }
    response = session.get(WIKI_API_URL, params=params)
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    for page_id, page_data in pages.items():
        if "revisions" in page_data:
            content = page_data["revisions"][0]["*"]
            title = page_data["title"]
            results[title] = content
    return results


st.title("OSRS Data Pipeline")
st.write("This will **wipe and redownload ALL configured content in the pipeline.** It will reformat downloaded content for LLM training and fine-tuning.")
st.warning("⚠️ **Important**: This process takes **a few hours**. This browser tab **must remain open** the entire time. Do not close or refresh.")

if 'pipeline_started' not in st.session_state:
    st.session_state.pipeline_started = False

#showing button only if pipeline hasn't started
if not st.session_state.pipeline_started:
    if st.button("🚀 Initiate Pipeline"):
        st.session_state.pipeline_started = True
        st.rerun()  # refreshes to hide button immediately
else:
    st.info("Pipeline in progress... ")

if st.session_state.pipeline_started:
    progress_bar = st.progress(0)
    table_placeholder = st.empty()

    progress_df = pd.DataFrame({
        "Metric": [
            "Total Content Pages Identified",
            "Page Titles Fetched",
            "Pages Downloaded & Processed",
            "Current Progress",
            "Last Batch Processed"
        ],
        "Value": [
            "Detecting...",
            "0",
            "0",
            "0%",
            "None"
        ]
    })
    table_placeholder.table(progress_df)

    with st.spinner("Initializing pipeline..."):
        session = get_session()

        # get total pages
        progress_df.loc[0, "Value"] = "Querying..."
        table_placeholder.table(progress_df)
        total_pages = total_pages(session)
        progress_df.loc[0, "Value"] = str(total_pages)
        progress_df.loc[3, "Value"] = "Fetching titles..."
        table_placeholder.table(progress_df)

        # wipe of both raw and clean directories
        for dir_path in [DATA_DIR, CLEAN_DIR]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

        # get all titles
        all_titles = page_titles(session, table_placeholder, progress_df)

        # Update after titles fetched
        progress_df.loc[1, "Value"] = f"{len(all_titles)} / {total_pages}"
        progress_df.loc[3, "Value"] = "Titles complete → Starting download"
        table_placeholder.table(progress_df)

        # DL and process content
        downloaded = 0
        qa_summaries = []
        qa_general = []
        qa_tables = []
        qa_sections = []
        batch_size = 50

        for i in range(0, len(all_titles), batch_size):
            batch_titles = all_titles[i:i + batch_size]
            batch_content = batch_extr(session, batch_titles)

            for title, wikitext in batch_content.items():
                safe_name = sanitize_filename(title)
                file_path = os.path.join(DATA_DIR, f"{safe_name}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(wikitext)

                #clean proc
                try:
                    parsed = mwparserfromhell.parse(wikitext)

                    # Summary QA
                    clean_text = parsed.strip_code(normalize=True, collapse=True).strip()
                    if clean_text:
                        qa_summaries.append(f"Question: Summarize the OSRS wiki page '{title}'. ||| Answer: {clean_text}")

                    # Adding bulk cleaned text directly (non-QA entry for general knowledge)
                    # formatting uniformity with other QA entries
                    qa_general.append(f"Question: Tell me about {title} in OSRS. ||| Answer: {clean_text}")

                    #  extracting structured stats from key infobox templates (before strip_code destroys them)
                    infoboxes = parsed.filter_templates(matches=lambda t: t.name.strip().lower() in [
                        "infobox item", "infobox monster", "infobox bonuses", "infobox pet"
                    ])
                    if infoboxes:
                        stats_lines = []
                        for tmpl in infoboxes:
                            for param in tmpl.params:
                                key = param.name.strip()
                                value = param.value.strip_code().strip()
                                if value and key.lower() not in ["version", "image", "caption"]:
                                    stats_lines.append(f"{key}: {value}")
                        if stats_lines:
                            stats_text = "\n".join(stats_lines)
                            qa_tables.append(f"Question: What are the stats and properties of '{title}' in OSRS? ||| Answer: {stats_text}")

                    # get recommended equipment from infoboxes
                    rec_eq = parsed.filter_templates(matches=lambda t: "recommended equipment" in t.name.strip().lower())
                    if rec_eq:
                        for tmpl in rec_eq:
                            eq_lines = []
                            for param in tmpl.params:
                                if param.name.strip() and param.value.strip():
                                    eq_lines.append(f"{param.name.strip()}: {param.value.strip_code().strip()}")
                            if eq_lines:
                                eq_text = "\n".join(eq_lines)
                                qa_tables.append(f"Question: What is the recommended equipment for '{title}' in OSRS? ||| Answer: {eq_text}")

                    # get drop tables
                    drops = parsed.filter_templates(matches=lambda t: t.name.strip().lower().startswith(("dropsline", "droptable", "drop")))
                    if drops:
                        drop_lines = []
                        for tmpl in drops:
                            for param in tmpl.params:
                                val = param.value.strip_code().strip()
                                if val:
                                    drop_lines.append(val)
                        if drop_lines:
                            # drop_text = "\n".join(drop_lines)
                            # more model context
                            drop_text = "The following items are dropped: " + ", ".join(drop_lines)
                            qa_tables.append(f"Question: What are the drops from '{title}' in OSRS? ||| Answer: {drop_text}")

                    # section-based QA
                    sections = parsed.get_sections(levels=[2,3], include_headings=True)
                    for section in sections:
                        if not section.filter_headings():
                            continue
                        section_title = section.filter_headings()[0].title.strip()
                        section_text = section.strip_code(normalize=True, collapse=True).strip()
                        if section_text:
                            # General section QA
                            qa_sections.append(f"Question: What does the section '{section_title}' say in the OSRS wiki page '{title}'? ||| Answer: {section_text}")
                            # Multiple specific questions for potentially critical sections
                            # this will always need expanded as we continue to grow on capabilities
                            lower_title = section_title.lower()
                            if "equipment" in lower_title:
                                qa_sections.append(f"Question: What gear should I use for '{title}' in OSRS? ||| Answer: {section_text}")
                                qa_sections.append(f"Question: What is the best in slot equipment for '{title}'? ||| Answer: {section_text}")
                            if "recommended skills" in lower_title:
                                qa_sections.append(f"Question: What skill levels are recommended for '{title}' in OSRS? ||| Answer: {section_text}")
                            if "transportation" in lower_title or "getting there" in lower_title:
                                qa_sections.append(f"Question: How do I get to '{title}' in OSRS? ||| Answer: {section_text}")
                            if "strategy" in lower_title or "guide" in lower_title:
                                qa_sections.append(f"Question: What is the strategy for '{title}' in OSRS? ||| Answer: {section_text}")
                            if "forms" in lower_title:
                                qa_sections.append(f"Question: What are the different forms of '{title}' in OSRS? ||| Answer: {section_text}")
                            if "money making" in lower_title:
                                qa_sections.append(f"Question: How profitable is this money making method for '{title}'? ||| Answer: {section_text}")
                                qa_sections.append(f"Question: What is this money making method for '{title}'? ||| Answer: {section_text}")
                                qa_sections.append(f"Question: How much GP per hour is '{title}'? ||| Answer: {section_text}")
                            if "combat achievements" in lower_title:
                                qa_sections.append(f"Question: What are the Combat Achievements for '{title}'? ||| Answer: {section_text}")
                            if "drops" in lower_title:
                                qa_sections.append(f"Question: What are the applicable drops and drop rates for '{title}'? ||| Answer: {section_text}")
                            if "snakelings" in lower_title:
                                qa_sections.append(f"Question: How do snakelings work during '{title}' in OSRS? ||| Answer: {section_text}")
                            if "special attacks" in lower_title:
                                qa_sections.append(f"Question: What special attacks does '{title}' have in OSRS? ||| Answer: {section_text}")
                            if "fight overview" in lower_title:
                                qa_sections.append(f"Question: What is the fight overview for '{title}'? ||| Answer: {section_text}")
                            if "changes" in lower_title:
                                qa_sections.append(f"Question: What historical changes have been made to '{title}' in OSRS? ||| Answer: {section_text}")
                            if "inventory" in lower_title or "setup" in lower_title:
                                    qa_sections.append(f"Question: What inventory setup is recommended for '{title}'? ||| Answer: {section_text}")
                            if "prayer" in lower_title:
                                qa_sections.append(f"Question: What prayers should I use against '{title}'? ||| Answer: {section_text}")
                            if "weakness" in lower_title or "combat" in lower_title:
                                qa_sections.append(f"Question: What are the weaknesses of '{title}'? ||| Answer: {section_text}")
                            if "requirements" in lower_title:
                                qa_sections.append(f"Question: What are the requirements for '{title}'? ||| Answer: {section_text}")

                except:
                    pass # silent fail on bad parse so we dont interrupt the entire pipeline

                downloaded += 1

            # updates table and progress bar in SL
            progress = min(1.0, downloaded / total_pages)
            progress_bar.progress(progress)
            progress_df.loc[2, "Value"] = f"{downloaded} / {total_pages}"
            progress_df.loc[3, "Value"] = f"{progress * 100:.1f}%"
            progress_df.loc[4, "Value"] = f"{batch_titles[0]} → {batch_titles[-1]}"
            table_placeholder.table(progress_df)

            time.sleep(1) # being respectful to the wiki

        #final save - Generating separate datasets for phased training
        data_categories = {
            "summaries": qa_summaries,
            "general": qa_general,
            "tables": qa_tables,
            "sections": qa_sections
        }

        # need if uploading model to HF
        dataset_info = {
            "description": "Comprehensive instruction dataset for Old School RuneScape (January 2026 snapshot). Contains high-quality Q&A pairs summarizing game content, including stats, equipment, strategies, and mechanics. Ideal for fine-tuning LLMs into OSRS domain experts.",
            "citation": "@misc{osrs_llm_dataset_2026,\n  author = {kineticquant},\n  title = {OSRS Instruction Dataset},\n  year = {2026},\n  publisher = {GitHub},\n  journal = {GitHub repository},\n  howpublished = {\\url{https://github.com/kineticquant/osrs-llm}}\n}",
            "homepage": "https://github.com/kineticquant/osrs-llm",
            "license": "cc-by-sa-4.0",
            "features": {
                "text": {
                    "dtype": "string",
                    "_type": "Value"
                }
            }
        }

        total_saved = 0
        for cat_name, entries in data_categories.items():
            if entries:
                cat_path = os.path.join(CLEAN_DIR, cat_name)
                df = pd.DataFrame({'text': entries})
                dataset = Dataset.from_pandas(df)
                dataset.save_to_disk(cat_path)
                
                # save info into each category folder for portability
                with open(os.path.join(cat_path, "dataset_info.json"), "w", encoding="utf-8") as f:
                    json.dump(dataset_info, f, indent=4)
                
                total_saved += len(entries)

        progress_df.loc[2, "Value"] = f"{total_pages} / {total_pages}"
        progress_df.loc[3, "Value"] = "100.0%"
        progress_df.loc[4, "Value"] = "All complete!"
        table_placeholder.table(progress_df)

        if total_saved > 0:
            st.success(f"🎉 Pipeline Complete! {total_saved:,} total entries saved across {len(data_categories)} categories in `{CLEAN_DIR}`")
        else:
            st.error("No data was processed. Check for errors.")

else:
    # removed because its just redundant and annoying - but keeping logic in case i want some info here at some point
    #st.info("Click the button above to begin. Be prepared to potentially need to keep this tab open for up to multiple hours.")
    pass