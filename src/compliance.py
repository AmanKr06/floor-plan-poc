import json
from openai import OpenAI
from src.config import Config

class ComplianceAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # ── UPDATED STANDARDS: Tailored specifically for CubiCasa classes ──
        # Classes available: 'room', 'wall', 'door', 'window'
        self.brand_standards = """
        ARCHITECTURAL LAYOUT STANDARDS v1.0
        1. Egress/Accessibility: Every 'room' must have at least one 'door' located near or intersecting its boundaries.
        2. Natural Light: Every 'room' must have at least one 'window' located near or intersecting its boundaries.
        3. Structural Integrity: The floor plan must contain 'wall' elements to separate spaces.
        4. Entryway: The total layout must contain at least 2 'door' elements (accounting for a main entrance and interior rooms).
        """

    def analyze(self, detections):
        # Translate the Python list into a readable string for GPT
        context_string = ", ".join(
            [f"1x {d['class_name']} at (X:{d['coordinates']['x']}, Y:{d['coordinates']['y']}, Width:{d['coordinates']['w']}, Height:{d['coordinates']['h']})" for d in detections]
        )

        prompt = f"""
        You are a senior floor plan compliance analyst.
        Based strictly on the following structural elements and their coordinates detected by our computer vision model, 
        analyze the floor plan against the architectural standards.

        DETECTED ELEMENTS:
        {context_string}

        {self.brand_standards}

        Use spatial reasoning based on the X, Y coordinates and Width/Height to determine if the detected layout violates any standards. For example, if a 'window' shares similar X or Y coordinates with a 'room', it belongs to that room.

        Return ONLY a raw JSON object. No markdown formatting.
        {{
            "compliance_score": <integer 0-100>,
            "status": "Pass" | "Review" | "Fail",
            "issues": ["<specific violation based on coordinate data>"],
            "summary": "<one sentence assessment>"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=500
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}