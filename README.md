# ER Triage LLM Assistant

### An AI-Powered Emergency Room Intake Prototype

**Project Type:** Academic Group Project  
**Role:** Project Lead  
**Domain:** Generative AI | Healthcare | Natural Language Processing

## Project Overview

Emergency departments face significant operational pressures, including high patient volumes, lengthy waiting times, and the need to collect accurate information during patient intake.

This project explored how large language models could support the early stages of emergency room intake by collecting structured symptom information and generating accessible, non-diagnostic patient summaries.

As project lead, I led a team in developing a functional prototype that combines Google's Gemini model, structured symptom questionnaires, rule-based branching, and retrieval from medical information sources.

The solution was designed as an information-gathering and decision-support concept, not as a replacement for clinical judgment.

## The Problem

Emergency room intake involves collecting patient symptoms, identifying relevant risk factors, and documenting information that clinicians may need for further assessment.

We explored whether an AI-assisted intake workflow could help standardize symptom collection, improve information accessibility, and potentially reduce administrative burden.

Our central question was:

**How can generative AI support emergency room intake while maintaining clear boundaries around patient safety and clinical decision-making?**

## Our Solution

We developed a Python-based prototype with a Streamlit user interface.

The application allows users to describe their symptoms in natural language, matches their description against structured condition templates, and guides them through relevant follow-up questions.

It then uses Gemini to generate a patient-friendly summary based on the available information.

The prototype includes:

- Natural language symptom input
- Local keyword and synonym matching
- LLM-assisted semantic matching
- NHS information retrieval as a fallback
- Structured, branching symptom questionnaires
- Age-based routing and condition-specific question flows
- Non-diagnostic summary generation
- Restricted follow-up explanations
- A simulated clinician review queue for out-of-scope questions

## Technical Architecture

The prototype uses a hybrid approach combining deterministic logic with generative AI.

**1. Symptom intake:** Users describe their symptoms through a Streamlit interface.

**2. Condition matching:** The system first searches local condition templates using keywords and synonyms. If no match is found, Gemini attempts semantic matching.

**3. External information retrieval:** If no local template is matched, the system can attempt to retrieve relevant information through the NHS API.

**4. Structured questioning:** Rule-based flows collect additional information, with branching based on responses and patient characteristics.

**5. Summary generation:** Gemini generates a conversational summary using the available condition information and user responses.

**6. Follow-up and escalation:** Follow-up questions are restricted to the provided information. Questions outside the intended scope can trigger a simulated clinician review workflow.

## Tools and Technologies

| Technology | Application |
|---|---|
| Python | Core application logic |
| Google Gemini | Semantic matching and text generation |
| Streamlit | Interactive user interface |
| JSON | Structured condition templates and question flows |
| NHS API | External health information retrieval |
| Requests | HTTP API communication |

## Safety and Responsible AI

Because this prototype operates in a healthcare context, we incorporated safeguards intended to limit inappropriate medical guidance.

These include user consent, non-diagnostic messaging, restricted prompts, predefined condition information, and a simulated escalation mechanism.

These safeguards are design features rather than evidence of clinical safety. The prototype has not been clinically validated and should not be used for real medical decisions.

## Project Leadership

As project lead, I coordinated the team's development of the prototype, bringing together the technical architecture, user experience, and business considerations into a unified solution.

The project strengthened my experience in leading collaborative AI development, translating technical concepts into practical applications, and evaluating the limitations of generative AI in high-stakes environments.

## Limitations and Future Development

The prototype is an academic proof of concept. Future development would require:

- Clinical expert review and validation
- More extensive testing across diverse symptom scenarios
- Stronger safeguards and deterministic escalation controls
- Secure patient data handling and privacy protections
- Evaluation of reliability, accuracy, and usability
- Integration with clinical workflows and information systems

## Repository Contents

- `main.py`: Command-line prototype and core triage logic
- `er_triage_streamlit.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `.env.example`: Example environment variable configuration
- `.gitignore`: Files excluded from version control

The application also references structured condition documents and a definitions glossary, which are required for relevant functionality.

## Running the Prototype

Install the dependencies:

```bash
pip install -r requirements.txt
```

Configure the `GENAI_API_KEY` and `NHS_API_KEY` environment variables using your own credentials.

Run the Streamlit interface:

```bash
streamlit run er_triage_streamlit.py
```

Or run the command-line version:

```bash
python main.py
```

**Note:** The application requires the relevant condition templates and glossary files for full functionality. The NHS integration uses a sandbox endpoint and requires appropriate access.

## Disclaimer

This is an educational prototype developed for an academic group project. It does not provide medical diagnoses, treatment recommendations, or clinically validated triage decisions. It is not intended for use in actual patient care.
