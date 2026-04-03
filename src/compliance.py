import json
from openai import OpenAI
from src.config import Config

class ComplianceAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # ── UPDATED STANDARDS: Tailored specifically for CubiCasa classes ──
        # Classes available: 'room', 'wall', 'door', 'window'
        self.brand_standards = """
        
        Detected element classes: 'zone', 'door', 'window', 'wall'

        RULE 1 — Minimum zones required:
        The floor plan must contain at least 4 distinct 'zone' elements.
        These represent: kitchen, dining, counter/service, and restroom areas.
        Flag if fewer than 5 zones are detected.

        RULE 2 — Exit accessibility (Critical):
        Every zone must have at least one 'door' element whose X,Y coordinates
        are within 150 pixels of the zone boundary. A zone with no nearby door
        is an accessibility violation. Minimum 2 doors required for the full layout.

        RULE 3 — Natural light / ventilation:
        At least 40% of detected zones must have a 'window' element within
        200 pixels of their boundary. Zones with no nearby window should be flagged
        as potentially non-compliant with ventilation requirements.

        RULE 4 — Structural separation:
        'wall' elements must be present to separate zones. If the total count of
        wall elements is fewer than 3, flag as insufficient structural separation.

        RULE 5 — Zone size distribution:
        Using the Width and Height of each zone, check for balance.
        One zone should not occupy more than 50% of the total detected zone area
        (sum of all zone Width x Height). If it does, flag as disproportionate layout.

        RULE 6 — Service access:
        At least one 'door' must be positioned on the outer boundary of the layout
        (X coordinate below 100 or above 90% of max X, or Y below 100 or above 90%
        of max Y). This represents a public entrance. If absent, flag it.

        VERDICT LOGIC:
        - score 80-100 → status: "Pass"
        - score 50-79  → status: "Review"
        - score 0-49   → status: "Fail"
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