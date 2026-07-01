import os
import re
import json
import httpx
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


load_dotenv()

app = FastAPI(title="Protein-Ligand Interaction Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"

# Correct replacement for deprecated llama-3.3-70b-versatile
GROQ_MODEL = "openai/gpt-oss-120b"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "groq",
        "ai": GROQ_MODEL,
    }


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def parse_model_json_object(text: str) -> Dict[str, Any]:
    """
    Parse JSON object from model response.
    Uses direct parsing first, with fallback extraction if extra text appears.
    """
    cleaned_text = text.strip()
    cleaned_text = re.sub(r"^```json\s*", "", cleaned_text)
    cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()

    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Model response was not a JSON object.")
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned_text)
        if not match:
            raise ValueError("Could not parse response as JSON object.")

        parsed = json.loads(match.group())

        if not isinstance(parsed, dict):
            raise ValueError("Parsed response was not a JSON object.")

        return parsed


@app.get("/protein-info")
async def protein_info(name: str):
    result: Dict[str, Any] = {
        "name": name,
        "uniprot_id": "",
        "full_name": "",
        "organism": "",
        "length": "",
        "function": "",
        "binding_sites": [],
        "disease": "",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": name,
                    "format": "json",
                    "size": 1,
                    "fields": (
                        "accession,id,protein_name,organism_name,length,"
                        "cc_function,ft_binding,cc_disease"
                    ),
                },
            )
            resp.raise_for_status()

            data = resp.json()
            results = data.get("results", [])

            if not results:
                result["error"] = "No UniProt entry found."
                return result

            entry = results[0]

            result["uniprot_id"] = safe_str(entry.get("primaryAccession"))

            protein_description = entry.get("proteinDescription") or {}
            recommended_name = protein_description.get("recommendedName") or {}
            full_name = recommended_name.get("fullName") or {}

            result["full_name"] = safe_str(full_name.get("value"), name)
            result["organism"] = safe_str(
                (entry.get("organism") or {}).get("scientificName")
            )
            result["length"] = safe_str((entry.get("sequence") or {}).get("length"))

            for comment in entry.get("comments", []):
                comment_type = comment.get("commentType")

                if comment_type == "FUNCTION":
                    texts = comment.get("texts") or []

                    if texts:
                        result["function"] = safe_str(texts[0].get("value"))[:500]

                elif comment_type == "DISEASE":
                    disease = comment.get("disease") or {}
                    disease_name = disease.get("diseaseName") or {}
                    result["disease"] = safe_str(disease_name.get("value"))

            for feature in entry.get("features", []):
                if feature.get("type") != "Binding site":
                    continue

                location = feature.get("location") or {}
                start = (location.get("start") or {}).get("value", "")
                end = (location.get("end") or {}).get("value", "")
                description = safe_str(feature.get("description"))

                if start or end or description:
                    result["binding_sites"].append(
                        f"Position {start}-{end}: {description}".strip()
                    )

    except httpx.HTTPStatusError as e:
        result["error"] = f"UniProt API HTTP error: {e.response.status_code}"
        result["details"] = e.response.text
    except Exception as e:
        result["error"] = f"Protein lookup failed: {str(e)}"

    return result


@app.get("/ligand-info")
async def ligand_info(name: str):
    result: Dict[str, Any] = {
        "name": name,
        "chembl_id": "",
        "molecular_formula": "",
        "molecular_weight": "",
        "alogp": "",
        "known_targets": [],
        "max_phase": "",
        "indication": "",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://www.ebi.ac.uk/chembl/api/data/molecule/search",
                params={
                    "q": name,
                    "format": "json",
                    "limit": 1,
                },
            )
            resp.raise_for_status()

            data = resp.json()
            molecules = data.get("molecules", [])

            if not molecules:
                result["error"] = "No ChEMBL molecule found."
                return result

            molecule = molecules[0]

            result["chembl_id"] = safe_str(molecule.get("molecule_chembl_id"))
            properties = molecule.get("molecule_properties") or {}

            result["molecular_formula"] = safe_str(properties.get("full_molformula"))
            result["molecular_weight"] = safe_str(properties.get("full_mwt"))
            result["alogp"] = safe_str(properties.get("alogp"))
            result["max_phase"] = safe_str(molecule.get("max_phase"))
            result["indication"] = safe_str(molecule.get("indication_class"))

            chembl_id = result["chembl_id"]

            if chembl_id:
                activity_resp = await client.get(
                    "https://www.ebi.ac.uk/chembl/api/data/activity",
                    params={
                        "molecule_chembl_id": chembl_id,
                        "format": "json",
                        "limit": 10,
                        "assay_type": "B",
                    },
                )
                activity_resp.raise_for_status()

                seen_targets = set()

                for activity in activity_resp.json().get("activities", []):
                    target = safe_str(activity.get("target_pref_name"))

                    if target and target not in seen_targets:
                        seen_targets.add(target)

                        result["known_targets"].append(
                            {
                                "target": target,
                                "standard_type": safe_str(
                                    activity.get("standard_type")
                                ),
                                "standard_value": safe_str(
                                    activity.get("standard_value")
                                ),
                                "standard_units": safe_str(
                                    activity.get("standard_units")
                                ),
                            }
                        )

                    if len(result["known_targets"]) >= 5:
                        break

    except httpx.HTTPStatusError as e:
        result["error"] = f"ChEMBL API HTTP error: {e.response.status_code}"
        result["details"] = e.response.text
    except Exception as e:
        result["error"] = f"Ligand lookup failed: {str(e)}"

    return result


class PredictRequest(BaseModel):
    protein: Dict[str, Any]
    ligand: Dict[str, Any]
    context: str = ""


@app.post("/predict")
async def predict(req: PredictRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on server."}

    protein_str = json.dumps(req.protein, indent=2)
    ligand_str = json.dumps(req.ligand, indent=2)

    prompt = f"""
You are an expert computational chemist and structural biologist.

Predict the likely binding interaction between this protein and ligand based on known biochemistry, structural biology principles, and available public data.

Important:
- This is a research-oriented computational prediction.
- Do not present the result as experimentally validated unless the provided data supports it.
- Do not provide clinical or treatment advice.

Protein Data:
{protein_str}

Ligand Data:
{ligand_str}

Disease/Research Context:
{req.context or "General research"}

Return ONLY a valid JSON object with exactly these keys:
{{
  "binding_affinity": "Strong OR Moderate OR Weak OR Unlikely",
  "confidence": 0,
  "mechanism": "2-3 sentences explaining the molecular interaction mechanism",
  "key_interactions": [
    "interaction 1",
    "interaction 2",
    "interaction 3"
  ],
  "selectivity": "1-2 sentences on selectivity versus other targets",
  "druggability": "High OR Moderate OR Low",
  "druggability_reason": "1-2 sentences explaining druggability",
  "similar_known_binders": [
    "known binder 1",
    "known binder 2"
  ],
  "clinical_relevance": "1-2 sentences on therapeutic or research relevance",
  "limitations": "1 sentence on limitations of this prediction"
}}

Rules:
- confidence must be an integer from 0 to 100.
- binding_affinity must be exactly one of: "Strong", "Moderate", "Weak", "Unlikely".
- druggability must be exactly one of: "High", "Moderate", "Low".
- key_interactions must contain 3-4 strings.
- similar_known_binders must contain 2-3 strings. Use "Unknown" if insufficient data is available.
- Do not include markdown.
- Do not include backticks.
- Do not include text before or after the JSON object.
""".strip()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                GROQ_BASE,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a computational chemistry JSON API. "
                                "Output only valid JSON objects. Do not output markdown."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1500,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()

            data = resp.json()

            if "error" in data:
                return {
                    "error": data["error"].get("message", "Groq API error")
                }

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not text:
                return {"error": "Empty response from model."}

            parsed = parse_model_json_object(text)

            confidence = parsed.get("confidence", 0)

            try:
                confidence = int(confidence)
            except Exception:
                confidence = 0

            binding_affinity = safe_str(parsed.get("binding_affinity"))
            if binding_affinity not in {"Strong", "Moderate", "Weak", "Unlikely"}:
                binding_affinity = "Unlikely"

            druggability = safe_str(parsed.get("druggability"))
            if druggability not in {"High", "Moderate", "Low"}:
                druggability = "Low"

            key_interactions = parsed.get("key_interactions", [])
            if not isinstance(key_interactions, list):
                key_interactions = [safe_str(key_interactions)] if key_interactions else []

            similar_known_binders = parsed.get("similar_known_binders", [])
            if not isinstance(similar_known_binders, list):
                similar_known_binders = (
                    [safe_str(similar_known_binders)]
                    if similar_known_binders
                    else []
                )

            return {
                "binding_affinity": binding_affinity,
                "confidence": max(0, min(confidence, 100)),
                "mechanism": safe_str(parsed.get("mechanism")),
                "key_interactions": [
                    safe_str(item) for item in key_interactions[:4]
                ],
                "selectivity": safe_str(parsed.get("selectivity")),
                "druggability": druggability,
                "druggability_reason": safe_str(
                    parsed.get("druggability_reason")
                ),
                "similar_known_binders": [
                    safe_str(item) for item in similar_known_binders[:3]
                ],
                "clinical_relevance": safe_str(
                    parsed.get("clinical_relevance")
                ),
                "limitations": safe_str(parsed.get("limitations")),
            }

    except httpx.HTTPStatusError as e:
        return {
            "error": f"Groq API HTTP error: {e.response.status_code}",
            "details": e.response.text,
        }
    except json.JSONDecodeError:
        return {"error": "Model returned invalid JSON."}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# Mount static frontend only if the folder exists.
# This prevents deployment crash when static/ is missing.
static_dir = Path("static")

if static_dir.exists() and static_dir.is_dir():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
