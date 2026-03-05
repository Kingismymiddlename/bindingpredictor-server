import os
import re
import json
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


@app.get("/health")
def health():
    return {"status": "ok", "ai": "groq/llama-3.3-70b"}


@app.get("/protein-info")
async def protein_info(name: str):
    result = {"name": name, "uniprot_id": "", "full_name": "", "organism": "",
              "length": "", "function": "", "binding_sites": [], "disease": ""}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={"query": name, "format": "json", "size": 1,
                        "fields": "id,protein_name,organism_name,length,cc_function,ft_binding,cc_disease"}
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    entry = results[0]
                    result["uniprot_id"] = entry.get("primaryAccession", "")
                    pnames = entry.get("proteinDescription", {})
                    rec = pnames.get("recommendedName", {})
                    result["full_name"] = rec.get("fullName", {}).get("value", name)
                    result["organism"] = entry.get("organism", {}).get("scientificName", "")
                    result["length"] = entry.get("sequence", {}).get("length", "")
                    for c in entry.get("comments", []):
                        if c.get("commentType") == "FUNCTION":
                            texts = c.get("texts", [])
                            if texts:
                                result["function"] = texts[0].get("value", "")[:500]
                        if c.get("commentType") == "DISEASE":
                            result["disease"] = c.get("disease", {}).get("diseaseName", {}).get("value", "")
                    for f in entry.get("features", []):
                        if f.get("type") == "Binding site":
                            loc = f.get("location", {})
                            start = loc.get("start", {}).get("value", "")
                            end = loc.get("end", {}).get("value", "")
                            result["binding_sites"].append(f"Position {start}-{end}: {f.get('description','')}")
    except Exception as e:
        result["error"] = str(e)
    return result


@app.get("/ligand-info")
async def ligand_info(name: str):
    result = {"name": name, "chembl_id": "", "molecular_formula": "",
              "molecular_weight": "", "alogp": "", "known_targets": [],
              "max_phase": "", "indication": ""}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://www.ebi.ac.uk/chembl/api/data/molecule/search",
                params={"q": name, "format": "json", "limit": 1}
            )
            if resp.status_code == 200:
                data = resp.json()
                mols = data.get("molecules", [])
                if mols:
                    mol = mols[0]
                    result["chembl_id"] = mol.get("molecule_chembl_id", "")
                    props = mol.get("molecule_properties", {}) or {}
                    result["molecular_formula"] = props.get("full_molformula", "")
                    result["molecular_weight"] = props.get("full_mwt", "")
                    result["alogp"] = props.get("alogp", "")
                    result["max_phase"] = str(mol.get("max_phase", ""))
                    result["indication"] = mol.get("indication_class", "") or ""
                    chembl_id = result["chembl_id"]
                    if chembl_id:
                        act_resp = await client.get(
                            "https://www.ebi.ac.uk/chembl/api/data/activity",
                            params={"molecule_chembl_id": chembl_id, "format": "json",
                                    "limit": 5, "assay_type": "B"}
                        )
                        if act_resp.status_code == 200:
                            seen = set()
                            for a in act_resp.json().get("activities", []):
                                target = a.get("target_pref_name", "")
                                if target and target not in seen:
                                    seen.add(target)
                                    result["known_targets"].append({
                                        "target": target,
                                        "standard_type": a.get("standard_type", ""),
                                        "standard_value": a.get("standard_value", ""),
                                        "standard_units": a.get("standard_units", "")
                                    })
    except Exception as e:
        result["error"] = str(e)
    return result


class PredictRequest(BaseModel):
    protein: dict
    ligand: dict
    context: str = ""


@app.post("/predict")
async def predict(req: PredictRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on server."}

    protein_str = json.dumps(req.protein, indent=2)
    ligand_str = json.dumps(req.ligand, indent=2)

    prompt = f"""You are an expert computational chemist and structural biologist. Predict the binding interaction between this protein and ligand based on known biochemistry and published data.

Protein Data:
{protein_str}

Ligand Data:
{ligand_str}

Disease/Research Context: {req.context or "General research"}

Respond with ONLY a raw JSON object. No markdown. No code fences. Start with {{ end with }}.

Required keys:
- binding_affinity: one of "Strong", "Moderate", "Weak", "Unlikely"
- confidence: integer 0-100
- mechanism: string (2-3 sentences explaining the molecular interaction mechanism)
- key_interactions: array of 3-4 strings (specific interaction types e.g. "Hydrogen bond with Lys745")
- selectivity: string (1-2 sentences on selectivity vs other targets)
- druggability: one of "High", "Moderate", "Low"
- druggability_reason: string (1-2 sentences)
- similar_known_binders: array of 2-3 strings (known drugs that bind this target)
- clinical_relevance: string (1-2 sentences on therapeutic relevance)
- limitations: string (1 sentence on limitations of this prediction)
"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a JSON API for computational chemistry. Output only raw JSON objects. No markdown. No explanation. Start with { end with }."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1500,
                "response_format": {"type": "json_object"}
            }
        )
        data = resp.json()
        if "error" in data:
            return {"error": data["error"].get("message", "Groq API error")}

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()

        try:
            return json.loads(text)
        except Exception:
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                try:
                    return json.loads(m.group())
                except Exception:
                    pass
        return {"error": f"Could not parse response: {text[:300]}"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
