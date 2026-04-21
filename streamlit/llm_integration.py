import os
from typing import Tuple, Optional
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ── Constants (mirrored from app.py) ──────────────────────────────────────────
CLASSES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]

INTERPRETATIONS = {
    0: "No significant structural impact detected. The building appears intact "
       "with no visible damage from the disaster event.",
    1: "Small, localized damage observed. Structural integrity is mostly "
       "maintained; repairs may be needed.",
    2: "Significant structural damage present. The building has sustained major "
       "harm and may be unsafe for occupancy.",
    3: "Severe or near-total destruction detected. The structure has collapsed "
       "or is critically compromised.",
}

SEVERITY_PROFILE = {
    0: {
        "level": "LOW",
        "response_type": "Precautionary",
        "urgency": "Non-urgent — monitoring and documentation phase",
    },
    1: {
        "level": "MODERATE",
        "response_type": "Active Response",
        "urgency": "Moderate — localized intervention required",
    },
    2: {
        "level": "HIGH",
        "response_type": "Emergency Response",
        "urgency": "High — immediate field teams required",
    },
    3: {
        "level": "CRITICAL",
        "response_type": "Mass Casualty Incident Protocol",
        "urgency": "Critical — all available resources must be deployed",
    },
}


# ── API Key ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Load Gemini API key from .env file."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file:\n"
        )
    return key.strip()


# ── Area Extraction ───────────────────────────────────────────────────────────

def extract_area_from_filename(filename: str) -> str:
    """
    Extract disaster area/location from image filename.

    Examples:
      nepal_kathmandu_pre_disaster.tif  ->  "Nepal, Kathmandu"
      turkey_istanbul_post.tif          ->  "Turkey, Istanbul"
      image.tif                         ->  "Unknown Location"
    """
    name = os.path.splitext(filename)[0]
    parts = name.lower().split("_")
    ignore = {"pre", "post", "disaster", "image", "satellite",
              "tif", "tiff", "img", "photo", "aerial", "scene"}
    location_parts = [p for p in parts if p and p not in ignore]
    if location_parts:
        return ", ".join(p.capitalize() for p in location_parts)
    return "Unknown Location"


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(
    damage_class: str,
    interpretation: str,
    area: str,
    confidence: float,
    pred_idx: int,
    all_probs: list,
) -> str:
    """
    Build a rich, detailed prompt for Gemini.
    Includes all class probabilities, severity profile, and specific
    instructions for a comprehensive, structured emergency response plan.
    """
    profile = SEVERITY_PROFILE[pred_idx]

    prob_breakdown = "\n".join(
        f"  - {CLASSES[i]:<16} {all_probs[i]*100:5.1f}%"
        for i in range(len(CLASSES))
    )

    return f"""You are a senior disaster response and emergency management expert with 20+ years of field experience coordinating relief operations in earthquake, flood, and conflict zones worldwide.

A Siamese CNN + Multi-Scale Vision Transformer deep learning architecture has analysed pre- and post-disaster satellite imagery and produced the following damage assessment. Your task is to generate a **comprehensive, detailed, field-ready emergency response plan** based on this output.

══════════════════════════════════════════════
  DAMAGE ASSESSMENT REPORT — DAMAGESCOPE AI
══════════════════════════════════════════════
  Disaster Location   : {area}
  Predicted Class     : {damage_class}
  Model Confidence    : {confidence*100:.1f}%
  Severity Level      : {profile["level"]}
  Response Type       : {profile["response_type"]}
  Urgency             : {profile["urgency"]}

  Full Class Probability Breakdown:
{prob_breakdown}

  Model Interpretation:
  {interpretation}
══════════════════════════════════════════════

INSTRUCTIONS:
Generate a highly detailed, structured emergency response plan for rescue and relief teams deployed to {area}.

Requirements:
- Each section must contain at least 5-6 specific, actionable bullet points
- Use precise language appropriate for field commanders and emergency coordinators
- Consider the geographic and cultural context of {area} when giving recommendations
- Scale the urgency and resource deployment proportional to severity level: {profile["level"]}
- Include specific numbers, timeframes, and measurable actions wherever possible
- Reference relevant international disaster response frameworks (SPHERE, ICS, INSARAG) where appropriate

Use EXACTLY this structure:

## Immediate Actions (0-6 Hours)
*First-response priorities the moment teams arrive on site.*
- ...

## Search and Rescue Operations
*Protocols for locating and extracting survivors.*
- ...

## Medical Aid and Triage
*On-site medical response, triage categories, and hospital coordination.*
- ...

## Structural and Infrastructure Assessment
*Engineering checks, hazard identification, utility management.*
- ...

## Evacuation, Displacement and Shelter
*Population movement, temporary shelter setup, and registration.*
- ...

## Resource and Logistics Deployment
*Equipment, personnel, supply chain, and inter-agency coordination.*
- ...

## Critical Safety Hazards to Avoid
*Specific risks teams must actively mitigate or stay clear of.*
- ...

## Communication and Command Structure
*Coordination protocols, reporting lines, and public communication.*
- ...

## 72-Hour Recovery Timeline
*Phased milestones: Hour 0-6, Hour 6-24, Hour 24-48, Hour 48-72.*
- ...

## Post-Incident Follow-Up and Documentation
*Secondary assessments, damage documentation, long-term recovery planning.*
- ...

End with a paragraph titled **Overall Assessment Summary** that synthesises the situation, the top 3 most critical priorities for the incident commander, and expected recovery timeline.
"""


# ── Gemini API Call ───────────────────────────────────────────────────────────

def call_gemini(prompt: str, api_key: str) -> Tuple[bool, str]:
    """Send prompt to Gemini 2.5 Flash and return (success, response_text)."""
    try:
        import google.generativeai as genai
    except ImportError:
        return False, (
            "Package missing. Run:  pip install google-generativeai python-dotenv"
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,       
                max_output_tokens=4096,
            ),
        )
        return True, response.text

    except Exception as exc:
        return False, f"Gemini API Error: {exc}"


# ── Main Entry Point ──────────────────────────────────────────────────────────

def get_emergency_measures(
    pred_idx: int,
    probs,
    pre_filename: str,
    post_filename: str,
) -> Tuple[bool, str]:
    """
    Main entry point -- call this from app.py after inference.

    Args:
        pred_idx      : Predicted class index (0-3)
        probs         : Probability array from model (numpy array or list)
        pre_filename  : Original pre-disaster filename  (for area extraction)
        post_filename : Original post-disaster filename (fallback area)

    Returns:
        (success: bool, response_text: str)
    """
    try:
        api_key = get_api_key()
    except EnvironmentError as e:
        return False, str(e)

    damage_class   = CLASSES[pred_idx]
    interpretation = INTERPRETATIONS[pred_idx]
    confidence     = float(probs[pred_idx])
    all_probs      = [float(p) for p in probs]

    # Extract area -- prefer pre-disaster filename
    area = extract_area_from_filename(pre_filename)
    if area == "Unknown Location":
        area = extract_area_from_filename(post_filename)

    prompt = build_prompt(
        damage_class   = damage_class,
        interpretation = interpretation,
        area           = area,
        confidence     = confidence,
        pred_idx       = pred_idx,
        all_probs      = all_probs,
    )

    return call_gemini(prompt, api_key)