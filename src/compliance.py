import json
from openai import OpenAI
from src.config import Config

class ComplianceAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # ── UPDATED STANDARDS: Tailored specifically for Kaggle Dataset ──
        # Expected Classes: 'zone' (or 'room'), 'door', 'window'
        self.brand_standards = """
        ARCHITECTURAL LAYOUT STANDARDS v2.0
        1. Egress/Accessibility: Every detected 'zone' (or 'room') must have at least one 'door' located near or intersecting its boundaries to allow entry/exit.
        2. Natural Light: Every detected 'zone' (or 'room') must have at least one 'window' located near or intersecting its boundaries.
        3. Entryway Minimums: The overall floor plan must contain a minimum of 2 'door' elements total to account for a main entrance and interior flow.
        4. Spatial Logic: 'window' elements should typically reside on the outer edges of a 'zone', while 'door' elements act as bridges between 'zones' or the exterior.
        """

    def analyze(self, detections):
        # ── INJECT THE ID INTO THE CONTEXT STRING ──
        context_string = ", ".join(
            [f"[{d['id']}] {d['class_name']} at (X:{d['coordinates']['x']}, Y:{d['coordinates']['y']}, Width:{d['coordinates']['w']}, Height:{d['coordinates']['h']})" for d in detections]
        )

        prompt = f"""
        You are a senior floor plan compliance analyst.
        Based strictly on the following structural elements and their coordinates detected by our computer vision model, 
        analyze the floor plan against the architectural standards.

        DETECTED ELEMENTS:
        {context_string}

        {self.brand_standards}

        Use spatial reasoning based on the X, Y coordinates and Width/Height to determine if the detected layout violates any standards. 
        For example, if a 'window' shares similar X or Y coordinates with a 'zone', it belongs to that zone. 
        CRITICAL: Do not penalize the floor plan for missing 'walls' or other elements not explicitly listed in the standards.

        Return ONLY a raw JSON object. No markdown formatting.
        {{
            "compliance_score": <integer 0-100>,
            "status": "Pass" | "Review" | "Fail",
            "explainability_matrix": [
                {{
                    "feature": "<STRICTLY USE THE ID HERE e.g., ZONE-1. NEVER use X/Y coordinates here.>",
                    "issue": "<e.g., Lacks a window>",
                    "point_impact": <integer, e.g., -15>,
                    "reasoning": "<brief explanation of why this penalty was applied>"
                }}
            ],
            "summary": "<one sentence assessment>"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=800
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}