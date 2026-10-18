import os
import json
import re
import requests
import google.generativeai as genai
import streamlit as st

# ------------------ CONFIGURATION ------------------
GENAI_API_KEY = os.getenv("GENAI_API_KEY") or "YOUR_GENAI_KEY_HERE"
NHS_API_KEY = os.getenv("NHS_API_KEY") or "YOUR_NHS_KEY_HERE"

os.environ["GENAI_API_KEY"] = GENAI_API_KEY
genai.configure(api_key=GENAI_API_KEY)

MODEL_NAME = "gemini-2.5-pro"   
NHS_BASE = "https://sandbox.api.service.nhs.uk/health-a-to-z/conditions"


# ------------------ UTILITIES ------------------
def safe_json_extract(text):
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


# ------------------ DOCUMENT HANDLING ------------------
def load_documents(folder_path="documents"):
    docs = []
    if not os.path.isdir(folder_path):
        print(f"Document folder not found: {folder_path}")
        return docs

    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".json", ".txt")):
            file_path = os.path.join(folder_path, fname)
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    content = f.read().strip()
                if not content:
                    continue
                data = json.loads(content)
                # Normalize keyword lists if present
                data["keywords"] = [k.lower() for k in data.get("keywords", [])]
                if "synonyms" in data:
                    data["synonyms"] = [s.lower() for s in data.get("synonyms", [])]
                else:
                    # also look for 'keyword_variants' and flatten if present
                    kv = data.get("keyword_variants", {})
                    synonyms = []
                    for v in kv.values():
                        if isinstance(v, list):
                            synonyms.extend(v)
                    data.setdefault("synonyms", [s.lower() for s in synonyms])
                docs.append(data)
                print(f"Loaded JSON from {fname}")
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    print(f"Total documents loaded: {len(docs)}")
    return docs


def load_definitions(path="definitions.json"):
    """
    Load a simple glossary of terms for follow-up questions.
    Returns a dict with lowercase keys so lookups are easier.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k).lower(): str(v) for k, v in data.items()}
        else:
            print("definitions.json is not a JSON object; ignoring.")
            return {}
    except FileNotFoundError:
        print("No definitions.json found – follow-up answers will only use the condition doc and NHS text.")
        return {}
    except Exception as e:
        print(f"Error loading definitions.json: {e}")
        return {}


# ------------------ GEMINI QUERIES ------------------
def query_gemini(prompt: str) -> str | None:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return getattr(response, "text", None)
    except Exception as e:
        print("Error querying Gemini:", e)
        return None


def gemini_select_document(user_input, docs):
    doc_info = [
        {
            "condition": d.get("condition"),
            "keywords": d.get("keywords", []) + d.get("synonyms", [])
        }
        for d in docs
    ]
    prompt = f"""
You are an ER triage assistant helping classify medical symptoms.
User input: "{user_input}"
Below is a list of condition documents with their associated keywords and synonyms:
{json.dumps(doc_info, indent=2)}
Your task:
- Determine which condition best matches the user's input (based on meaning, not exact words).
- Return ONLY a JSON object in this format:
{{ "condition": "<condition name>", "matched_keyword": "<keyword_or_synonym_used>" }}
If nothing matches, return: {{ "condition": null, "matched_keyword": null }}
"""
    response = query_gemini(prompt)
    return safe_json_extract(response)


def gemini_suggest_nhs_slug(user_input):
    prompt = f"""
You are an assistant formatting medical condition names into NHS website slugs.
User input: "{user_input}"
Provide ONLY a JSON object: {{ "nhs_slug": "<slug>" }}
Slug must be lowercase, no punctuation, spaces replaced by hyphens.
"""
    response = query_gemini(prompt)
    data = safe_json_extract(response)
    if data and data.get("nhs_slug"):
        return data["nhs_slug"].strip().lower().replace(" ", "-")
    return None


# ------------------ LOCAL MATCHING ------------------
def local_match_document(user_input, docs):
    ui = user_input.lower()
    for d in docs:
        for kw in d.get("keywords", []) + d.get("synonyms", []):
            if kw and kw in ui:
                return d
    return None


# ------------------ NHS API ------------------
def fetch_nhs_summary(nhs_slug):
    if not nhs_slug:
        return None
    url = f"{NHS_BASE}/{nhs_slug}"
    headers = {
        "Accept": "application/json",
        "Subscription-Key": NHS_API_KEY
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            text = data.get("description") or data.get("content") or data.get("text")
            if isinstance(text, dict):
                text = text.get("text") or json.dumps(text)
            if isinstance(text, list):
                text = " ".join([str(x) for x in text])
            return text
        else:
            print(f"NHS API request failed: {r.status_code} - {r.text}")
            return None
    except Exception as e:
        print("NHS API request failed:", e)
        return None


# ------------------ PROMPT + BUILD ------------------
def build_gemini_prompt(user_condition, user_answers=None, doc=None, nhs_info=None):
    content_info = ""
    if doc:
        desc = doc.get("description", "")
        rf = doc.get("risk_factors", [])
        rf_text = json.dumps(rf, indent=2)
        red = doc.get("red_flags", [])
        red_text = json.dumps(red, indent=2)
        notes = doc.get("notes", []) or doc.get("safety_disclaimer", "")
        content_info = f"""
Condition: {doc.get('condition')}
Description: {desc}
Risk factors: {rf_text}
Red flags: {red_text}
Notes: {json.dumps(notes, indent=2)}
"""
    elif nhs_info:
        content_info = f"Information from NHS: {nhs_info}\n"

    user_answers_text = json.dumps(user_answers, indent=2) if user_answers else "{}"

    prompt = f"""
You are an experienced ER triage nurse. Use ONLY the information provided below to create a friendly, conversational summary for the patient.

Patient reported: {user_condition}

Patient's responses:
{user_answers_text}

{content_info}

Produce a short text that includes:
- A summary of their reported issue.
- A friendly list of symptoms to watch for in the next hours/days (using only the 'red_flags' and 'risk_factors' provided).
- A short list of key risk factors relevant to the patient's answers. Keep all of their answers in mind.
- Present the red flags as urgent items to watch for. Keep in mind the answers they have given to the questions
- Use a conversational, reassuring tone; do not provide medical advice or say 'see a doctor' or 'you should/should not'.
- Do not invent any information; only rephrase and organise the facts provided above.
"""
    return prompt


def build_followup_prompt(user_condition, user_answers, doc, nhs_info, follow_question, glossary=None):
    """
    Build a very restricted prompt for definition-style follow-up questions.
    The model is only allowed to clarify or rephrase terms that appear in:
    - the condition doc,
    - the NHS text (if any),
    - the user's answers,
    - the glossary.
    """

    glossary = glossary or {}

    # Prepare the content we allow the model to use
    condition_name = doc.get("condition") if doc else None
    desc = doc.get("description", "") if doc else ""
    risk_factors = doc.get("risk_factors", []) if doc else []
    red_flags = doc.get("red_flags", []) if doc else []
    notes = doc.get("notes", []) or doc.get("safety_disclaimer", "") if doc else ""
    source_url = doc.get("source") if doc else None

    user_answers_text = json.dumps(user_answers or {}, indent=2)
    risk_text = json.dumps(risk_factors, indent=2, ensure_ascii=False)
    red_text = json.dumps(red_flags, indent=2, ensure_ascii=False)
    notes_text = json.dumps(notes, indent=2, ensure_ascii=False)
    nhs_text = nhs_info or ""
    glossary_text = json.dumps(glossary, indent=2, ensure_ascii=False)

    safe_source_text = source_url or "the official website used as the source for this condition"

    prompt = f"""
You are an assistant helping a patient understand a previous triage summary.

IMPORTANT SAFETY RULES:
- You MUST ONLY use information from:
  * the CONDITION DOC,
  * the RISK FACTORS,
  * the RED FLAGS,
  * the NOTES / SAFETY DISCLAIMER,
  * the NHS TEXT (if any),
  * the USER ANSWERS,
  * and the GLOSSARY provided below.
- You are allowed to:
  * define or explain terms that appear in that content (for example what "dehydration" or "red flag" means),
  * rephrase parts of the given text to make it easier to understand,
  * point the user back to the source website if they want more detail.
- You are NOT allowed to:
  * provide new medical facts that are not clearly stated in the provided content,
  * name new diagnoses, treatments, medicines, or tests,
  * tell the user what they should do next,
  * give medical advice or instructions.

If the follow-up question cannot be safely answered using ONLY the information given, respond with exactly:
"This question is outside what this prototype can safely answer. Please discuss it with a clinician or see the source page: {safe_source_text}."

-------------------------
ORIGINAL CONDITION (if available): {condition_name}
PATIENT REPORTED: {user_condition}

USER ANSWERS FROM TRIAGE:
{user_answers_text}

CONDITION DESCRIPTION (if any):
{desc}

RISK FACTORS:
{risk_text}

RED FLAGS:
{red_text}

NOTES / SAFETY DISCLAIMER:
{notes_text}

NHS-BASED INFORMATION (if available):
{nhs_text}

GLOSSARY (approved short definitions for common terms):
{glossary_text}

-------------------------
FOLLOW-UP QUESTION FROM PATIENT:
{follow_question}

Your task:
- If possible, give a SHORT, clear explanation in simple language that only uses and rephrases the content above.
- If the question goes beyond this information, use the fixed safety message described earlier.
"""
    return prompt


def enqueue_for_clinician(user_condition, doc, user_answers, nhs_info, reason="follow_up_out_of_scope"):
    """
    Append a simple record to an escalation queue file (JSON lines).
    """
    record = {
        "reason": reason,
        "condition_doc": doc.get("condition") if doc else None,
        "source_url": doc.get("source") if doc else None,
        "user_condition_free_text": user_condition,
        "user_answers": user_answers or {},
        "nhs_info_excerpt": (nhs_info[:500] + "...") if nhs_info and len(nhs_info) > 500 else nhs_info,
        "follow_up_blocked": True
    }

    try:
        with open("escalation_queue.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("\n[Info] Case added to clinician review queue.")
    except Exception as e:
        print(f"\n[Warning] Could not write to escalation queue: {e}")


# ------------------ TRIAGE STEP LOGIC (ONE-BY-ONE) ------------------
def get_next_index(nodes, curr_index, ans):
    """
    Given the current node index and the user's answer, decide which
    node index to show next, based on routing keys.
    Mirrors the branching rules of run_nodes() in main.py
    (except for the CLI-specific fallback that references template/current_id).
    Returns the next index (int) or None if we're done with this flow.
    """
    node = nodes[curr_index]
    index_by_id = {n.get("id"): i for i, n in enumerate(nodes) if n.get("id")}

    next_index = curr_index + 1  # default: next in list

    next_id = None
    low = str(ans).strip().lower()

    # 1) Exact branching based on normalized answer
    if "next_if_yes" in node and low == "yes":
        next_id = node["next_if_yes"]
    elif "next_if_no" in node and low == "no":
        next_id = node["next_if_no"]
    elif "next_if_not_sure" in node and low == "not sure":
        next_id = node["next_if_not_sure"]
    elif "next_if_not_applicable" in node and low == "not applicable":
        next_id = node["next_if_not_applicable"]
    elif "always_next" in node:
        # Only used if none of the above matched
        next_id = node["always_next"]

    # 2) Fuzzy yes/no/not sure matching, only if nothing chosen yet
    if next_id is None:
        if "yes" in low and node.get("next_if_yes"):
            next_id = node["next_if_yes"]
        elif "no" in low and node.get("next_if_no"):
            next_id = node["next_if_no"]
        elif "not sure" in low and node.get("next_if_not_sure"):
            next_id = node["next_if_not_sure"]

    # 3) Generic next_if_... (next_if_male, next_if_female, etc.), if still None
    if next_id is None:
        for k, v in node.items():
            if k.startswith("next_if_") and k not in (
                "next_if_yes",
                "next_if_no",
                "next_if_not_sure",
                "next_if_not_applicable",
            ):
                option_key = k.replace("next_if_", "")
                # match as substring in the answer
                if option_key.lower() in low:
                    next_id = v
                    break

    # If next_id points to another node in the same flow, jump there
    if next_id and next_id in index_by_id:
        next_index = index_by_id[next_id]

    # If we run past the last node, this flow is done
    if next_index >= len(nodes):
        return None

    return next_index


def extract_age_from_answers(answers: dict) -> int | None:
    """
    Try to extract an age (int) from any answer whose key mentions 'age'.
    """
    for k, v in answers.items():
        if "age" in str(k).lower():
            digits = re.findall(r"\d+", str(v))
            if digits:
                try:
                    return int(digits[0])
                except ValueError:
                    continue
    return None


# ------------------ STREAMLIT HELPERS ------------------
@st.cache_data(show_spinner=False)
def cached_load_documents(path: str):
    return load_documents(path)


@st.cache_data(show_spinner=False)
def cached_load_definitions(path: str = "definitions.json"):
    return load_definitions(path)


# ------------------ STREAMLIT APP ------------------
def streamlit_main():
    st.set_page_config(page_title="ER Triage Prototype", page_icon="🩺", layout="wide")

    st.title("The ER Triage Assistant Prototype")
    st.write(
        "This prototype uses symptom templates and trusted content (like NHS) to provide "
        "**general information** and **things to watch for**. It does **not** replace a real medical assessment."
    )

    # Safety consent
    st.markdown("### Safety and Consent")
    consent = st.checkbox(
        "I understand this is for information only and **does not** replace medical care.",
        key="consent_checkbox",
    )

    if not consent:
        st.info("Please tick the box above to continue.")
        st.stop()

    # Sidebar config
    st.sidebar.header("Prototype Settings")
    docs_path = st.sidebar.text_input("Documents folder", value="documents")
    use_nhs = st.sidebar.checkbox("Use NHS fallback if no local doc", value=True)

    # User condition input
    st.markdown("### 1) Describe your injury or symptom")
    user_condition = st.text_area(
        "In your own words, what is going on?",
        placeholder="e.g. I cut my hand on a piece of glass and it is still bleeding after 20 minutes...",
        height=100,
        key="user_condition_text",
    )

    if not user_condition.strip():
        st.stop()

    # Load data
    docs = cached_load_documents(docs_path)
    glossary = cached_load_definitions("definitions.json")

    doc = None
    nhs_info = None

    # 1) Local match
    if docs:
        doc = local_match_document(user_condition, docs)

    # 2) Gemini match
    if not doc and docs:
        with st.spinner("Matching your description to a condition template..."):
            selection = gemini_select_document(user_condition, docs)
        if selection and selection.get("condition"):
            doc = next((d for d in docs if d.get("condition") == selection["condition"]), None)

    # Show match (if any)
    if doc:
        st.success(f"Matched condition template: **{doc.get('condition', 'Unknown')}**")
    else:
        st.warning("No local condition template matched your description.")

    # 3) NHS fallback if no doc
    if not doc and use_nhs:
        st.info("Trying NHS content based on your description...")
        with st.spinner("Contacting NHS sandbox API..."):
            nhs_slug = gemini_suggest_nhs_slug(user_condition) or user_condition.strip().lower().replace(" ", "-")
            nhs_info = fetch_nhs_summary(nhs_slug)

        if nhs_info:
            st.success(f"NHS information retrieved for slug: `{nhs_slug}`")
        else:
            st.error("No NHS information could be retrieved for this description.")

    # 4) Triage questions (multi-flow, one-by-one) – aligned with age router idea in main.py
    user_answers = {}

    if doc:
        triage_qs = doc.get("triage_questions", []) or []
        named_flows = {k: v for k, v in doc.items() if isinstance(v, list)}
        router_node = None
        router_idx = None
        for i, n in enumerate(triage_qs):
            if n.get("type") == "system":
                router_node = n
                router_idx = i
                break

        # --- Initialise triage state once per condition ---
        doc_id = doc.get("condition") or id(doc)
        if st.session_state.get("triage_doc_id") != doc_id:
            # new condition → reset
            st.session_state.triage_doc_id = doc_id
            st.session_state.triage_mode = None     # "simple" or "router"
            st.session_state.triage_stage = None    # "pre_router", "main_flow"
            st.session_state.triage_pre_index = 0
            st.session_state.triage_flow_name = None
            st.session_state.triage_flow_index = 0
            st.session_state.triage_done = False
            st.session_state.triage_answers = {}
            st.session_state.router_idx = router_idx

            if named_flows and router_node is not None:
                # CASE A — router-based docs (age routing)
                st.session_state.triage_mode = "router"
                st.session_state.triage_stage = "pre_router"
            else:
                # CASE B — simple docs with no routing
                st.session_state.triage_mode = "simple"
                st.session_state.triage_stage = "main_flow"
                st.session_state.triage_flow_name = "triage_questions"  # treat triage_qs as main flow

        # ------------------ ROUTER-BASED MODE ------------------
        if st.session_state.triage_mode == "router" and not st.session_state.triage_done:
            st.markdown("### 2) Answer a few quick questions (one at a time)")
            st.caption("These questions help tailor the information to your situation.")

            # PRE-ROUTER STAGE: ask questions in triage_questions before the system router
            if st.session_state.triage_stage == "pre_router":
                pre_index = st.session_state.triage_pre_index
                router_idx = st.session_state.router_idx

                # Safety if router_idx is missing
                if router_idx is None:
                    # fallback to simple mode
                    st.session_state.triage_mode = "simple"
                    st.session_state.triage_stage = "main_flow"
                else:
                    # still have pre-router questions to ask?
                    if pre_index < router_idx:
                        node = triage_qs[pre_index]
                        q = node.get("question", "(no question)")
                        options = node.get("options", [])
                        answer_key = f"triage_pre_answer_{pre_index}"

                        with st.form(f"triage_pre_q_{pre_index}"):
                            if options:
                                answer = st.radio(q, options=options, key=answer_key)
                            else:
                                answer = st.text_input(q, key=answer_key)

                            next_clicked = st.form_submit_button("Next")

                        if next_clicked:
                            ans_key = node.get("id") or q
                            st.session_state.triage_answers[ans_key] = answer
                            st.session_state.triage_pre_index = pre_index + 1
                            st.rerun()

                        # Wait until "Next" is clicked
                        st.stop()

                    # finished all pre-router questions → compute age and pick flow
                    age_val = extract_age_from_answers(st.session_state.triage_answers)
                    # fallback: if no age extracted, default to adult flow (same spirit as main.py)
                    if age_val is not None and age_val < 16:
                        chosen_flow = router_node.get("route_under_16", "child_flow")
                    else:
                        chosen_flow = router_node.get("route_16_or_older", "adult_flow")

                    # safety fallback if chosen flow missing
                    if chosen_flow not in doc:
                        if "adult_flow" in doc:
                            chosen_flow = "adult_flow"
                        elif "child_flow" in doc:
                            chosen_flow = "child_flow"

                    st.session_state.triage_flow_name = chosen_flow
                    st.session_state.triage_flow_index = 0
                    st.session_state.triage_stage = "main_flow"
                    st.rerun()

            # MAIN_FLOW STAGE: walk through chosen flow one-by-one
            if st.session_state.triage_stage == "main_flow" and not st.session_state.triage_done:
                flow_name = st.session_state.triage_flow_name
                if not flow_name or flow_name not in doc:
                    # no valid flow; end triage
                    st.session_state.triage_done = True
                else:
                    nodes = doc[flow_name]
                    idx = st.session_state.triage_flow_index

                    # skip any 'system' nodes inside the flow
                    while idx < len(nodes) and nodes[idx].get("type") == "system":
                        idx += 1

                    if idx >= len(nodes):
                        # finished this flow; triage done
                        st.session_state.triage_done = True
                    else:
                        node = nodes[idx]
                        q = node.get("question", "(no question)")
                        options = node.get("options", [])
                        answer_key = f"triage_flow_{flow_name}_idx_{idx}"

                        with st.form(f"triage_flow_q_{flow_name}_{idx}"):
                            if options:
                                answer = st.radio(q, options=options, key=answer_key)
                            else:
                                answer = st.text_input(q, key=answer_key)

                            next_clicked = st.form_submit_button("Next")

                        if next_clicked:
                            ans_key = node.get("id") or q
                            st.session_state.triage_answers[ans_key] = answer

                            next_index = get_next_index(nodes, idx, answer)
                            if next_index is None:
                                st.session_state.triage_done = True
                            else:
                                st.session_state.triage_flow_index = next_index

                            st.rerun()

                        st.stop()

        # ------------------ SIMPLE MODE (NO ROUTER) ------------------
        if st.session_state.triage_mode == "simple" and not st.session_state.triage_done:
            st.markdown("### 2) Answer a few quick questions (one at a time)")
            st.caption("These questions help tailor the information to your situation.")

            nodes = triage_qs
            idx = st.session_state.triage_flow_index

            # skip system nodes, if any
            while idx < len(nodes) and nodes[idx].get("type") == "system":
                idx += 1

            if idx >= len(nodes):
                st.session_state.triage_done = True
            else:
                node = nodes[idx]
                q = node.get("question", "(no question)")
                options = node.get("options", [])
                answer_key = f"triage_simple_idx_{idx}"

                with st.form(f"triage_simple_q_{idx}"):
                    if options:
                        answer = st.radio(q, options=options, key=answer_key)
                    else:
                        answer = st.text_input(q, key=answer_key)

                    next_clicked = st.form_submit_button("Next")

                if next_clicked:
                    ans_key = node.get("id") or q
                    st.session_state.triage_answers[ans_key] = answer

                    next_index = get_next_index(nodes, idx, answer)
                    if next_index is None:
                        st.session_state.triage_done = True
                    else:
                        st.session_state.triage_flow_index = next_index

                    st.rerun()

                st.stop()

        # once triage is done, pull answers from session
        if st.session_state.triage_done:
            user_answers = st.session_state.triage_answers or {}

    # 5) Gemini summary
    st.markdown("### 3) Personalized information")
    with st.spinner("Generating a summary based on your description and answers..."):
        prompt = build_gemini_prompt(user_condition, user_answers if doc else None, doc, nhs_info)
        output = query_gemini(prompt)

    if output:
        st.write(output)
    else:
        st.error("No output could be generated from Gemini.")

    # 6) Follow-up Q&A
    st.markdown("### 4) Follow-up questions (information only)")
    st.caption(
        "You can ask short clarification questions about terms or risk factors mentioned above. "
        "This will not give diagnoses or treatment advice."
    )

    follow_q = st.text_input(
        "Type a follow-up question (optional):",
        placeholder="e.g. What does 'red flag' mean here?",
        key="follow_up_question",
    )

    if st.button("Ask follow-up") and follow_q.strip():
        with st.spinner("Generating follow-up explanation..."):
            follow_prompt = build_followup_prompt(
                user_condition=user_condition,
                user_answers=user_answers,
                doc=doc,
                nhs_info=nhs_info,
                follow_question=follow_q,
                glossary=glossary,
            )
            follow_answer = query_gemini(follow_prompt)

        if follow_answer:
            st.markdown("#### Follow-up explanation")
            st.write(follow_answer)

            SAFETY_PHRASE = "This question is outside what this prototype can safely answer."
            if SAFETY_PHRASE in follow_answer:
                st.warning(
                    "Your question goes beyond what this prototype can safely answer. "
                    "In a real system this would be sent to a clinician."
                )
                if st.button("Simulate adding this case to a clinician review queue"):
                    enqueue_for_clinician(
                        user_condition=user_condition,
                        doc=doc,
                        user_answers=user_answers,
                        nhs_info=nhs_info,
                        reason="follow_up_out_of_scope",
                    )
                    st.success("Case added to `escalation_queue.jsonl` (local demo queue).")
        else:
            st.error("There was a problem generating the follow-up explanation.")


if __name__ == "__main__":
    streamlit_main()
