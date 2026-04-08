import pandas as pd
import numpy as np
import random

def generate_realistic_floor_plan_data(num_rows=500):
    data = []
    
    for _ in range(num_rows):
        # ─── 1. SIMULATE YOLO DETECTIONS (The Math) ───
        # Most floor plans have 1-5 zones. We occasionally force a 0 to simulate a terrible upload.
        zone_count = random.choices([0, 1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.3, 0.3, 0.15, 0.1])[0]
        
        # Doors and windows usually scale with zones, but YOLO sometimes misses them
        door_count = max(0, zone_count + random.randint(-1, 2))
        window_count = max(0, zone_count + random.randint(-1, 2))
        
        # Calculate missing elements (cannot be greater than total zones)
        if zone_count == 0:
            zones_missing_door = 0
            zones_missing_window = 0
        else:
            # Most of the time (70%), no missing doors. Sometimes 1 or 2.
            zones_missing_door = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]
            zones_missing_door = min(zones_missing_door, zone_count)
            
            zones_missing_window = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
            zones_missing_window = min(zones_missing_window, zone_count)

        # ─── 2. SIMULATE GPT-4o VISION (The Semantic Engine) ───
        # We simulate the exact mathematical rules we gave GPT-4o in compliance.py
        gpt_base_score = 100
        if zone_count == 0: gpt_base_score -= 30
        if door_count == 0: gpt_base_score -= 20
        if window_count == 0: gpt_base_score -= 20
        
        gpt_base_score -= (zones_missing_door * 20)
        gpt_base_score -= (zones_missing_window * 15)
        
        # Add a tiny bit of AI "fuzziness" (1-3 points of variance)
        gpt_compliance_score = max(0, min(100, gpt_base_score + random.randint(-3, 3)))
        
        # Encode GPT Status (Pass = 2, Review = 1, Fail = 0)
        if gpt_compliance_score >= 80:
            gpt_status_encoded = 2
        elif gpt_compliance_score >= 50:
            gpt_status_encoded = 1
        else:
            gpt_status_encoded = 0

        # ─── 3. SIMULATE HUMAN GROUND TRUTH (The Target) ───
        # Human architects usually agree with GPT, but sometimes they are stricter or more lenient.
        # We introduce "Human Disagreement" (-8 to +8 points) so XGBoost has to learn to balance YOLO and GPT.
        human_disagreement = random.randint(-8, 8)
        final_approved_score = max(0, min(100, gpt_compliance_score + human_disagreement))
        
        if final_approved_score >= 80:
            final_status = "Pass"
        elif final_approved_score >= 50:
            final_status = "Review"
        else:
            final_status = "Reject"

        # Append the row
        data.append({
            "zone_count": zone_count,
            "door_count": door_count,
            "window_count": window_count,
            "zones_missing_door": zones_missing_door,
            "zones_missing_window": zones_missing_window,
            "gpt_compliance_score": gpt_compliance_score,
            "gpt_status_encoded": gpt_status_encoded,
            "Final_Approved_Score": final_approved_score,
            "Final_Status": final_status
        })
        
    df = pd.DataFrame(data)
    df.to_csv("dummy_training_data.csv", index=False)
    print(f"✅ Successfully generated 'dummy_training_data.csv' with {num_rows} rows.")

if __name__ == "__main__":
    generate_realistic_floor_plan_data(500)