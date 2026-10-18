import os
import json
import re
import requests
import google.generativeai as genai

# ------------------ CONFIG ------------------
GENAI_API_KEY = os.getenv("GENAI_API_KEY") or "YOUR_API_KEY_HERE"
NHS_API_KEY = os.getenv("NHS_API_KEY") or "YOUR_NHS_KEY_HERE"

os.environ["GENAI_API_KEY"] = GENAI_API_KEY
genai.configure(api_key=GENAI_API_KEY)

MODEL_NAME = "gemini-2.5-pro"   # adjust if needed
NHS_BASE = "https://sandbox.api.service.nhs.uk/health-a-to-z/conditions"

# ------------------ UTILITIES ------------------
def safe_json_extract(text):
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except:
        return None

# ------------------ DOCUMENT HANDLING ------------------
def load_documents(folder_path="documents"):
    docs = []
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
                # synonyms/variant names might be under different keys in your files
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
    doc_info = [{"condition": d.get("condition"), "keywords": d.get("keywords", []) + d.get("synonyms", [])} for d in docs]
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

# ------------------ TRIAGE RUNNER (generic) ------------------
def ask_question(node):
    q = node.get("question", "(no question)")
    options = node.get("options", [])
    if options:
        ans = input(f"{q} Options: {', '.join(options)}\n").strip()
    else:
        ans = input(f"{q}\n").strip()
    return ans

def run_nodes(nodes, doc=None, answers=None):
    """
    Run a linear/branched list of nodes (a flow). Returns collected answers.
    nodes: list of node dicts with fields like id, question, type, options, always_next, next_if_yes, next_if_<option>
    """
    if answers is None:
        answers = {}

    # Build id->index map to allow jumps by id
    index_by_id = {n.get("id"): i for i, n in enumerate(nodes) if n.get("id")}
    i = 0
    while i < len(nodes):
        node = nodes[i]
        ntype = node.get("type", "choice")
        # skip system nodes here (they belong in top-level triage router logic)
        if ntype == "system":
            i += 1
            continue

        ans = ask_question(node)
        key = node.get("id") or node.get("question")
        answers[key] = ans

        # determine next node id
        next_id = None

        #define q for branching logic
        q = node
        # Normalize answer once
        low = str(ans).strip().lower()

        # 1) Exact branching based on normalized answer
        if "next_if_yes" in q and low == "yes":
            next_id = q["next_if_yes"]
        elif "next_if_no" in q and low == "no":
            next_id = q["next_if_no"]
        elif "next_if_not_sure" in q and low == "not sure":
            next_id = q["next_if_not_sure"]
        elif "next_if_not_applicable" in q and low == "not applicable":
            next_id = q["next_if_not_applicable"]
        elif "always_next" in q:
            # only used if no exact branch matched
            next_id = q["always_next"]

        # 2) Fuzzy yes/no/not sure matching, only if nothing chosen yet
        if next_id is None:
            if "yes" in low and q.get("next_if_yes"):
                next_id = q.get("next_if_yes")
            elif "no" in low and q.get("next_if_no"):
                next_id = q.get("next_if_no")
            elif "not sure" in low and q.get("next_if_not_sure"):
                next_id = q.get("next_if_not_sure")

        # 3) Generic next_if_... (next_if_male, next_if_female, etc.), if still None
        if next_id is None:
            for k, v in q.items():
                if k.startswith("next_if_") and k not in ("next_if_yes", "next_if_no", "next_if_not_sure"):
                    option_key = k.replace("next_if_", "")
                    # match as substring in the answer
                    if option_key.lower() in low:
                        next_id = v
                        break


        # If next_id switches to another flow in doc (e.g., 'child_flow'), return special marker
        if next_id and doc and next_id in doc and isinstance(doc[next_id], list):
            # caller can detect this and handle switching flows
            return answers, next_id

        # If next_id is an id in same nodes list, jump
        if next_id and next_id in index_by_id:
            i = index_by_id[next_id]
            continue

        # Otherwise step forward
        i += 1

    return answers, None

# ------------------ HIGH-LEVEL TRIAGE (handles both doc styles) ------------------
def run_triage(doc):
    """
    Handles:
    - Age routing (route_under_12 / route_12_or_older)
    - Pregnancy routing after adult_preg_entry_flow
    - Flow switching via next_if_<flowname>
    """
    answers = {}

    triage_qs = doc.get("triage_questions", [])
    named_flows = {k: v for k, v in doc.items() if isinstance(v, list)}

    # Find the first router inside triage_questions
    router_node = None
    for n in triage_qs:
        if n.get("type") == "system":
            router_node = n
            break

    # ============================================================
    # CASE A — Document uses age routing with separate flows
    # ============================================================
    if named_flows and router_node:
        idx_router = triage_qs.index(router_node)
        pre_nodes = triage_qs[:idx_router]

        # --- Run questions BEFORE the age router (only age question)
        if pre_nodes:
            a, switch = run_nodes(pre_nodes, doc=doc, answers=answers)
            answers.update(a)

        # --- Extract Age
        def extract_age():
            for k, v in answers.items():
                if "age" in str(k).lower():
                    digits = re.findall(r"\d+", str(v))
                    if digits:
                        return int(digits[0])
            return None

        age_val = extract_age()

        # If missing or invalid ask again
        if age_val is None:
            try:
                age_val = int(input("Please enter the patient's age:\n").strip())
            except:
                age_val = None

        # --- Compute FIRST level routing (child OR adult entry flow)
        if age_val is not None and age_val < 16:
            chosen_flow_name = router_node.get("route_under_16", "child_flow")
        else:
            chosen_flow_name = router_node.get("route_16_or_older", "adult_flow")

        # Safety fallback
        if chosen_flow_name not in doc:
            chosen_flow_name = "adult_flow"

        # --- RUN first chosen flow (child_flow OR adult_preg_entry_flow)
        flow_nodes = doc[chosen_flow_name]
        a, switch = run_nodes(flow_nodes, doc=doc, answers=answers)
        answers.update(a)

        # Handle any automatic flow-switch signals
        while switch and switch in doc:
            flow_nodes = doc[switch]
            a, switch = run_nodes(flow_nodes, doc=doc, answers=answers)
            answers.update(a)

        # ============================================================
        # SECOND ROUTER: Pregnancy routing AFTER adult entry flow
        # ============================================================
        if "a0_preg_check" in answers:
            preg_val = answers["a0_preg_check"].lower()

            if preg_val in ("yes", "not sure"):
                chosen_flow_name = "pregnancy_flow"
            else:
                chosen_flow_name = "adult_flow"

            if chosen_flow_name in doc:
                flow_nodes = doc[chosen_flow_name]
                a, switch = run_nodes(flow_nodes, doc=doc, answers=answers)
                answers.update(a)

                while switch and switch in doc:
                    flow_nodes = doc[switch]
                    a, switch = run_nodes(flow_nodes, doc=doc, answers=answers)
                    answers.update(a)

        return answers

    # ============================================================
    # CASE B — Simple documents (no routers)
    # ============================================================
    a, switch = run_nodes(triage_qs, doc=doc, answers=answers)
    answers.update(a)

    while switch and switch in doc:
        flow_nodes = doc[switch]
        a, switch = run_nodes(flow_nodes, doc=doc, answers=answers)
        answers.update(a)

    return answers


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
    In a real system this would go to a secure database or task system.
    Here it's just a .jsonl file for demonstration / grading purposes.
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
        print("\n[Info] Your case has been added to the clinician review queue for this prototype.")
    except Exception as e:
        print(f"\n[Warning] Could not write to escalation queue: {e}")

# ------------------ MAIN ------------------
def main():
    print("=== ER TRIAGE (Prototype) ===\n")

    # --- Safety onboarding / consent ---
    consent = input(
        "This prototype can only provide general information about symptoms and things to watch for, "
        "based on trusted sources (like the NHS) and the answers you give.\n"
        "It does NOT replace a real medical assessment.\n\n"
        "Do you understand and agree to continue? (yes/no) "
    ).strip().lower()

    if consent not in ("yes", "y"):
        print("\nThank you for your understanding. This prototype cannot be used without your agreement.")
        print("Please seek medical advice from a clinician or your local emergency services if you are worried.\n")
        return
    # --- End safety onboarding ---

    glossary = load_definitions("definitions.json")
    user_condition = input("1) What is your injury or symptom? ").strip()

    docs = load_documents("documents")
    doc = None

    # 1) local match first
    doc = local_match_document(user_condition, docs)
    if doc:
        print(f"Local match found: {doc.get('condition')}")

    # 2) If not, ask Gemini to pick
    if not doc and docs:
        selection = gemini_select_document(user_condition, docs)
        if selection and selection.get("condition"):
            doc = next((d for d in docs if d.get("condition") == selection["condition"]), None)
            if doc:
                print(f"Gemini matched local doc: {doc.get('condition')}")

    # 3) NHS fallback if no doc
    nhs_info = None
    if not doc:
        print("No local document matched. Attempting NHS lookup...")
        nhs_slug = gemini_suggest_nhs_slug(user_condition) or user_condition.strip().lower().replace(" ", "-")
        nhs_info = fetch_nhs_summary(nhs_slug)
        if nhs_info:
            print(f"NHS content fetched for slug '{nhs_slug}'")
        else:
            print("No NHS information available for that slug.")

    # 4) Run triage if doc present
    user_answers = {}
    if doc:
        user_answers = run_triage(doc)

    # 5) Build prompt and ask Gemini
    prompt = build_gemini_prompt(user_condition, user_answers, doc, nhs_info)
    output = query_gemini(prompt)

    print("\n=== PERSONALIZED ER INFORMATION ===\n")
    print(output or "No Gemini output available.")
    
    # 6) Follow-up Q&A loop (definition / clarification only)
    while True:
        more = input("\nDo you have a follow-up question about this information? (yes/no) ").strip().lower()
        if more not in ("yes", "y"):
            break

        follow_q = input("Please type your follow-up question: ").strip()
        if not follow_q:
            print("No question entered.")
            continue

        follow_prompt = build_followup_prompt(
            user_condition=user_condition,
            user_answers=user_answers,
            doc=doc,
            nhs_info=nhs_info,
            follow_question=follow_q,
            glossary=glossary
        )

        follow_answer = query_gemini(follow_prompt)
        print("\n--- Follow-up explanation (information only) ---\n")
        print(follow_answer or "Sorry, there was a problem generating a follow-up explanation.")

        # --- Escalation path: detect fixed safety message and offer queue option ---
        SAFETY_PHRASE = "This question is outside what this prototype can safely answer."
        if follow_answer and SAFETY_PHRASE in follow_answer:
            want_queue = input(
                "\nYour question is outside what this prototype can safely answer.\n"
                "In a real service, you could ask for a clinician to review your information.\n"
                "For this prototype, we can add your case to a 'clinician review queue' to show how this might work.\n"
                "Would you like this case to be added to the clinician review queue? (yes/no) "
            ).strip().lower()

            if want_queue in ("yes", "y"):
                enqueue_for_clinician(
                    user_condition=user_condition,
                    doc=doc,
                    user_answers=user_answers,
                    nhs_info=nhs_info,
                    reason="follow_up_out_of_scope"
                )
            else:
                print("\nOkay, we will not add this case to the queue.")

if __name__ == "__main__":
    main()
