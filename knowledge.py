"""
Condition knowledge base for MedicalAI - Light Weight.

Maps a predicted diagnosis label to plain-language, general-education
context: what the finding means, symptoms it's commonly associated with,
how urgent it typically is, and a suggested next step. This is shown
alongside model predictions in the CLI, web UI, batch output, and API so
a bare label + confidence score isn't the only thing a user sees.

This is general medical education content, not a diagnosis or treatment
plan, and every entry below carries that caveat: the app doesn't practice
medicine, it flags what a finding usually means and tells the user to
involve a clinician.
"""

# Urgency buckets, roughly in order of how quickly a clinician should be
# involved. Purely informational — the app never blocks or gates on these.
URGENCY_ROUTINE = "routine"
URGENCY_FOLLOW_UP = "follow-up"
URGENCY_URGENT = "urgent"
URGENCY_EMERGENCY = "emergency"

CONDITION_INFO = {
    "No Finding": {
        "description": "No acute abnormality was identified on this chest X-ray.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "No follow-up needed based on imaging alone. Continue routine care.",
    },
    "Atelectasis": {
        "description": "Partial or complete collapse of part of a lung, often from mucus plugging, post-surgical shallow breathing, or external compression.",
        "common_symptoms": ["shortness of breath", "shallow breathing", "cough"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Usually managed with breathing exercises, incentive spirometry, or treating the underlying cause; discuss with a clinician if new or worsening.",
    },
    "Cardiomegaly": {
        "description": "An enlarged cardiac silhouette, which can reflect heart failure, valve disease, cardiomyopathy, or a pericardial effusion.",
        "common_symptoms": ["shortness of breath", "leg swelling", "fatigue", "orthopnea"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Recommend echocardiogram and clinical cardiac evaluation to determine the cause.",
    },
    "Effusion": {
        "description": "Fluid collecting in the pleural space around a lung, which can be caused by heart failure, infection, malignancy, or other conditions.",
        "common_symptoms": ["shortness of breath", "chest pain", "cough"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Clinical correlation and, if large or symptomatic, consider thoracentesis to identify the cause.",
    },
    "Infiltration": {
        "description": "An area of increased density in the lung, commonly from infection, inflammation, fluid, or less commonly, tumor.",
        "common_symptoms": ["cough", "fever", "shortness of breath"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Correlate with clinical symptoms; persistent or worsening infiltrates warrant follow-up imaging.",
    },
    "Mass": {
        "description": "A lesion larger than 3 cm, which requires further characterization to distinguish benign from malignant causes.",
        "common_symptoms": ["cough", "weight loss", "hemoptysis"],
        "urgency": URGENCY_URGENT,
        "follow_up": "CT chest and specialist referral are typically recommended for further work-up.",
    },
    "Nodule": {
        "description": "A lesion smaller than 3 cm; most are benign (old scarring, granulomas) but size, growth, and risk factors determine next steps.",
        "common_symptoms": [],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Compare with prior imaging if available; CT chest is often recommended to characterize the nodule.",
    },
    "Pneumonia": {
        "description": "Infection of the lung tissue causing airspace consolidation, from bacterial, viral, or atypical organisms.",
        "common_symptoms": ["fever", "productive cough", "shortness of breath", "chest pain"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Clinical evaluation for antibiotic or antiviral therapy; seek prompt care if fever is high or breathing is labored.",
    },
    "Pneumothorax": {
        "description": "Air trapped in the pleural space causing partial or complete lung collapse.",
        "common_symptoms": ["sudden chest pain", "shortness of breath"],
        "urgency": URGENCY_EMERGENCY,
        "follow_up": "Small pneumothoraces may be observed; large or tension pneumothorax requires emergency treatment.",
    },
    "Consolidation": {
        "description": "Airspaces filled with fluid, pus, or blood instead of air — commonly from pneumonia, but also hemorrhage or other causes.",
        "common_symptoms": ["cough", "fever", "shortness of breath"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Clinical correlation needed to determine the cause and appropriate treatment.",
    },
    "Edema": {
        "description": "Fluid accumulation in the lung tissue, most often from heart failure (cardiogenic) but sometimes from other causes.",
        "common_symptoms": ["shortness of breath", "orthopnea", "leg swelling"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Prompt clinical evaluation recommended, especially if breathing difficulty is significant.",
    },
    "Pulmonary_Edema": {
        "description": "Fluid accumulation specifically within the lungs, either cardiogenic (heart failure) or non-cardiogenic (e.g., ARDS, high altitude).",
        "common_symptoms": ["shortness of breath", "orthopnea", "frothy sputum"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Prompt clinical evaluation recommended; identify whether the cause is cardiac or non-cardiac.",
    },
    "Emphysema": {
        "description": "Destruction of lung air sacs (alveoli), most commonly from smoking, leading to chronic airflow limitation.",
        "common_symptoms": ["chronic shortness of breath", "cough"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonary function testing and smoking cessation counseling if applicable.",
    },
    "Fibrosis": {
        "description": "Scarring of lung tissue that stiffens the lungs and reduces their ability to exchange oxygen.",
        "common_symptoms": ["progressive shortness of breath", "dry cough"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Referral to pulmonology for further evaluation, including possible high-resolution CT.",
    },
    "Pleural_Thickening": {
        "description": "Thickening of the lining around the lung, often from prior infection, inflammation, or asbestos exposure.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Usually benign and stable; note any occupational exposure history for the record.",
    },
    "Hernia": {
        "description": "Abdominal contents protruding into the chest cavity, most commonly a hiatal hernia.",
        "common_symptoms": ["heartburn", "regurgitation", "chest discomfort"],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Often incidental; symptomatic cases may warrant gastroenterology referral.",
    },
    "Calcified_Granuloma": {
        "description": "A small calcified nodule representing healed prior infection or inflammation, almost always benign.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "No action typically needed; useful as a stable reference point on future imaging.",
    },
    "Granulomatous_Disease": {
        "description": "Evidence of prior granuloma-forming infection or inflammation (e.g., histoplasmosis, TB, sarcoidosis).",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Usually reflects old, healed disease; correlate with exposure history and prior imaging.",
    },
    "Bronchiectasis": {
        "description": "Irreversible widening of the airways, often from chronic infection or inflammation, leading to mucus buildup.",
        "common_symptoms": ["chronic cough", "sputum production", "recurrent infections"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonology referral for airway clearance strategies and infection prevention.",
    },
    "Hiatal_Hernia": {
        "description": "A portion of the stomach protrudes through the diaphragm into the chest cavity.",
        "common_symptoms": ["heartburn", "regurgitation", "chest discomfort"],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Often incidental and manageable with lifestyle/medical therapy; large or symptomatic cases may need GI referral.",
    },
    "Pneumoperitoneum": {
        "description": "Free air beneath the diaphragm, which is a strong indicator of a perforated hollow organ in the abdomen.",
        "common_symptoms": ["severe abdominal pain", "rigid abdomen"],
        "urgency": URGENCY_EMERGENCY,
        "follow_up": "Requires urgent surgical evaluation.",
    },
    "Pericardial_Effusion": {
        "description": "Fluid accumulating in the sac surrounding the heart, which can impair heart function if large or rapid.",
        "common_symptoms": ["shortness of breath", "chest discomfort", "lightheadedness"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Echocardiogram recommended to assess size and hemodynamic impact.",
    },
    "Rib_Fracture": {
        "description": "A break in one or more ribs, usually from trauma.",
        "common_symptoms": ["localized chest pain", "pain with breathing"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pain control and monitoring for complications like pneumothorax or hemothorax; seek care if breathing worsens.",
    },
    "COPD": {
        "description": "Chronic Obstructive Pulmonary Disease — long-term airflow limitation, commonly from smoking, encompassing emphysema and chronic bronchitis.",
        "common_symptoms": ["chronic cough", "shortness of breath", "wheezing"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonary function testing, inhaler therapy, and smoking cessation support as applicable.",
    },
    "Interstitial_Lung_Disease": {
        "description": "A group of disorders causing progressive scarring of lung tissue, impairing oxygen exchange.",
        "common_symptoms": ["progressive shortness of breath", "dry cough"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonology referral for further work-up, including high-resolution CT.",
    },
    "Sarcoidosis": {
        "description": "An inflammatory disease causing granulomas to form in the lungs and lymph nodes, and sometimes other organs.",
        "common_symptoms": ["cough", "fatigue", "shortness of breath"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonology referral; many cases are mild and self-limited but need monitoring.",
    },
    "Lung_Cancer": {
        "description": "A malignant tumor of the lung, often first identified as a mass or nodule on imaging.",
        "common_symptoms": ["persistent cough", "weight loss", "hemoptysis", "chest pain"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Prompt oncology/pulmonology referral for staging and biopsy.",
    },
    "Metastatic_Disease": {
        "description": "Spread of cancer from another site in the body to the lungs, typically seen as multiple nodules or masses.",
        "common_symptoms": ["cough", "weight loss", "shortness of breath"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Oncology referral to identify the primary tumor and plan treatment.",
    },
    "Mediastinal_Mass": {
        "description": "A mass in the central chest compartment between the lungs, with causes ranging from benign cysts to lymphoma or thymoma.",
        "common_symptoms": ["chest pain", "cough", "difficulty swallowing"],
        "urgency": URGENCY_URGENT,
        "follow_up": "CT chest and specialist referral for further characterization.",
    },
    "Aortic_Aneurysm": {
        "description": "Abnormal widening of the aorta, which carries a risk of rupture if it enlarges further.",
        "common_symptoms": ["chest or back pain (often asymptomatic until rupture)"],
        "urgency": URGENCY_URGENT,
        "follow_up": "CT angiography and vascular surgery referral for sizing and monitoring.",
    },
    "Aortic_Dissection": {
        "description": "A tear in the inner layer of the aorta allowing blood to flow between the layers of the vessel wall — a life-threatening emergency.",
        "common_symptoms": ["sudden severe tearing chest or back pain"],
        "urgency": URGENCY_EMERGENCY,
        "follow_up": "Requires immediate emergency evaluation and treatment.",
    },
    "Pulmonary_Embolism": {
        "description": "A blood clot blocking one or more pulmonary arteries, which can be life-threatening.",
        "common_symptoms": ["sudden shortness of breath", "chest pain", "rapid heart rate"],
        "urgency": URGENCY_EMERGENCY,
        "follow_up": "Requires urgent evaluation, typically with CT pulmonary angiography.",
    },
    "Pulmonary_Hypertension": {
        "description": "Elevated blood pressure in the pulmonary arteries, which strains the right side of the heart over time.",
        "common_symptoms": ["shortness of breath on exertion", "fatigue", "leg swelling"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Echocardiogram and cardiology/pulmonology referral for further evaluation.",
    },
    "Tuberculosis": {
        "description": "A bacterial infection that most commonly affects the lungs and can be contagious when active.",
        "common_symptoms": ["chronic cough", "fever", "night sweats", "weight loss"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Sputum testing and isolation precautions as appropriate; prompt infectious disease evaluation.",
    },
    "Aspergillosis": {
        "description": "A fungal infection or reaction to the Aspergillus mold, ranging from an allergic response to invasive infection depending on the host.",
        "common_symptoms": ["cough", "hemoptysis", "wheezing"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonology referral; presentation and treatment vary widely by form (ABPA, aspergilloma, invasive).",
    },
    "ARDS": {
        "description": "Acute Respiratory Distress Syndrome — severe, widespread lung inflammation causing fluid buildup and impaired oxygenation.",
        "common_symptoms": ["severe shortness of breath", "rapid breathing"],
        "urgency": URGENCY_EMERGENCY,
        "follow_up": "Requires intensive care management.",
    },
    "Alveolar_Hemorrhage": {
        "description": "Bleeding into the lung's air sacs, from causes ranging from autoimmune disease to infection or trauma.",
        "common_symptoms": ["hemoptysis", "shortness of breath"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Prompt work-up to identify the underlying cause.",
    },
    "Organizing_Pneumonia": {
        "description": "A pattern of lung inflammation and healing that can follow infection, autoimmune disease, or certain medications.",
        "common_symptoms": ["cough", "low-grade fever", "shortness of breath"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Often responds well to corticosteroids after other causes are excluded; pulmonology follow-up recommended.",
    },
    "Hypersensitivity_Pneumonitis": {
        "description": "Lung inflammation from an allergic reaction to inhaled organic dust, mold, or bird proteins.",
        "common_symptoms": ["cough", "shortness of breath", "flu-like symptoms after exposure"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Identify and avoid the triggering exposure; pulmonology referral for further evaluation.",
    },
    "Lymphangitic_Carcinomatosis": {
        "description": "Spread of cancer through the lymphatic channels of the lung, usually from breast, lung, stomach, or pancreatic primaries.",
        "common_symptoms": ["progressive shortness of breath", "dry cough"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Oncology referral for evaluation and management of the underlying malignancy.",
    },
    "Lymphadenopathy": {
        "description": "Enlarged lymph nodes in the chest, which can reflect infection, inflammation, or malignancy.",
        "common_symptoms": [],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Clinical correlation and, if persistent or enlarging, further imaging or biopsy.",
    },
    "Pleural_Plaque": {
        "description": "Localized areas of pleural thickening, classically associated with prior asbestos exposure.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Usually benign; document occupational/exposure history.",
    },
    "Pneumoconiosis": {
        "description": "Lung disease caused by inhaling occupational dust (e.g., coal, silica, asbestos) over years.",
        "common_symptoms": ["chronic cough", "shortness of breath"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonology referral; document occupational exposure history.",
    },
    "Silicosis": {
        "description": "A form of pneumoconiosis from inhaling silica dust, common in mining and construction work.",
        "common_symptoms": ["chronic cough", "shortness of breath"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonology referral and occupational exposure counseling.",
    },
    "Asbestosis": {
        "description": "Lung scarring from inhaling asbestos fibers, typically after long-term occupational exposure.",
        "common_symptoms": ["chronic cough", "shortness of breath"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonology referral; monitor for associated pleural disease or malignancy risk.",
    },
    "Empyema": {
        "description": "A collection of pus in the pleural space, usually a complication of pneumonia.",
        "common_symptoms": ["fever", "chest pain", "shortness of breath"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Typically requires drainage and antibiotic therapy.",
    },
    "Hemothorax": {
        "description": "Blood collecting in the pleural space, most often from trauma or a procedural complication.",
        "common_symptoms": ["chest pain", "shortness of breath"],
        "urgency": URGENCY_EMERGENCY,
        "follow_up": "Requires prompt drainage and evaluation for the bleeding source.",
    },
    "Chylothorax": {
        "description": "Lymphatic fluid (chyle) collecting in the pleural space, often from thoracic duct injury or obstruction.",
        "common_symptoms": ["shortness of breath", "chest discomfort"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Evaluation for the underlying cause (surgical, traumatic, or malignant) and dietary/drainage management.",
    },
    "Round_Atelectasis": {
        "description": "A rounded area of collapsed lung adjacent to thickened pleura, a benign mimic of a tumor, often linked to asbestos exposure.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Usually stable and benign; comparison with prior imaging helps confirm the diagnosis.",
    },
    "Mucus_Plugging": {
        "description": "Airways blocked by thick mucus, which can cause partial lung collapse.",
        "common_symptoms": ["cough", "shortness of breath"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Airway clearance therapy and treatment of the underlying condition (e.g., asthma, bronchiectasis).",
    },
    "Tree_in_Bud": {
        "description": "A pattern of small nodules connected to branching lines, suggesting small airway disease, often infectious (including TB).",
        "common_symptoms": ["cough", "fever"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Further work-up for infectious causes recommended.",
    },
    "Cavitary_Lesion": {
        "description": "A gas-filled cavity within a lung lesion, seen in infections (TB, fungal, abscess) or malignancy.",
        "common_symptoms": ["cough", "fever", "weight loss"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Further work-up needed to distinguish infectious from malignant causes.",
    },
    "Cystic_Lung_Disease": {
        "description": "Multiple thin-walled air-filled cysts in the lung, seen in conditions like LAM or LCH.",
        "common_symptoms": ["shortness of breath", "recurrent pneumothorax"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Pulmonology referral for further characterization.",
    },
    "Post_Surgical_Changes": {
        "description": "Expected findings after thoracic surgery, such as staple lines, volume loss, or resected lung tissue.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Expected post-operative finding; compare with surgical history and prior imaging.",
    },
    "Support_Devices": {
        "description": "Medical hardware visible on the X-ray, such as central lines, pacemakers, or endotracheal tubes.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Confirm correct positioning of the device as part of clinical care.",
    },
    "Pancoast_Tumor": {
        "description": "A lung tumor at the apex of the lung that can invade nearby nerves, ribs, and blood vessels.",
        "common_symptoms": ["shoulder/arm pain", "hand weakness", "Horner's syndrome"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Prompt oncology/thoracic surgery referral for staging and treatment planning.",
    },
    "Diaphragmatic_Hernia": {
        "description": "Abdominal organs protruding into the chest through a defect in the diaphragm.",
        "common_symptoms": ["shortness of breath", "abdominal discomfort"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Surgical evaluation recommended, especially if symptomatic.",
    },
    "Eventration": {
        "description": "Abnormal elevation of part or all of the diaphragm due to weakness, rather than a structural defect.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Often incidental and asymptomatic; correlate clinically if breathing symptoms are present.",
    },
    "Subcutaneous_Emphysema": {
        "description": "Air trapped under the skin of the chest wall or neck, usually from an underlying pneumothorax or recent procedure/surgery.",
        "common_symptoms": ["chest wall swelling", "crackling sensation under the skin"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Identify and address the source of the air leak (e.g., pneumothorax).",
    },
    "Ankylosing_Spondylitis": {
        "description": "A chronic inflammatory arthritis affecting the spine, which can restrict chest wall movement and cause upper-lobe lung changes.",
        "common_symptoms": ["back pain and stiffness", "reduced chest expansion"],
        "urgency": URGENCY_FOLLOW_UP,
        "follow_up": "Rheumatology referral for management of the underlying condition.",
    },
    "Kyphoscoliosis": {
        "description": "Abnormal curvature of the spine that can restrict lung expansion when severe.",
        "common_symptoms": ["back deformity", "shortness of breath if severe"],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Orthopedic evaluation if progressive; pulmonary function testing if breathing is affected.",
    },
    "Thoracic_Spine_Fracture": {
        "description": "A break in one or more thoracic vertebrae, from trauma or weakened bone (e.g., osteoporosis).",
        "common_symptoms": ["back pain", "reduced mobility"],
        "urgency": URGENCY_URGENT,
        "follow_up": "CT or MRI to assess stability and spinal canal involvement; orthopedic/spine referral.",
    },

    # ── Musculoskeletal / bone X-ray (bone_xray_training.py, separate model) ──
    "Fracture": {
        "description": "A break in a bone, ranging from a hairline crack to a full break, identified on an extremity X-ray.",
        "common_symptoms": ["localized pain", "swelling", "reduced range of motion", "visible deformity"],
        "urgency": URGENCY_URGENT,
        "follow_up": "Orthopedic evaluation for immobilization/reduction; urgent care if displaced, open, or neurovascular compromise is suspected.",
    },
    "Hardware": {
        "description": "Surgical hardware (plates, screws, pins, or rods) visible on the X-ray from a prior orthopedic procedure.",
        "common_symptoms": [],
        "urgency": URGENCY_ROUTINE,
        "follow_up": "Expected post-operative finding; compare with surgical history and check for hardware loosening or failure.",
    },
}

_GENERIC_INFO = {
    "description": "This finding was flagged by the model but doesn't have a detailed entry yet in the local knowledge base.",
    "common_symptoms": [],
    "urgency": URGENCY_FOLLOW_UP,
    "follow_up": "Discuss this result with a qualified clinician for interpretation.",
}

DISCLAIMER = (
    "This is general educational information, not a medical diagnosis. "
    "Always confirm findings with a qualified healthcare professional."
)


def _normalize(label):
    return label.strip().lower().replace(" ", "_").replace("-", "_")


_NORMALIZED_LOOKUP = {_normalize(k): k for k in CONDITION_INFO}


def get_condition_info(label):
    """Look up knowledge-base info for a (possibly free-text) diagnosis label.

    Falls back to a generic entry for anything unrecognized — including
    full radiology-report-style strings from a trained model, where we
    scan for any known condition name inside the text — so callers never
    have to special-case a missing lookup.
    """
    if not label:
        return dict(_GENERIC_INFO)

    n = _normalize(label)
    if n in _NORMALIZED_LOOKUP:
        return dict(CONDITION_INFO[_NORMALIZED_LOOKUP[n]])

    # Free-text diagnosis (e.g. a full report string): check for any known
    # condition name mentioned inside it, preferring the longest match.
    best_key, best_len = None, -1
    for norm_key, original_key in _NORMALIZED_LOOKUP.items():
        if norm_key in n and len(norm_key) > best_len:
            best_key, best_len = original_key, len(norm_key)
    if best_key:
        return dict(CONDITION_INFO[best_key])

    return dict(_GENERIC_INFO)


def format_condition_info(label, confidence=None):
    """Human-readable multi-line summary for CLI / batch output."""
    info = get_condition_info(label)
    lines = [info["description"]]
    if info["common_symptoms"]:
        lines.append("Commonly associated with: " + ", ".join(info["common_symptoms"]) + ".")
    lines.append(f"Urgency: {info['urgency']}. {info['follow_up']}")
    if confidence is not None:
        lines.append(f"(Model confidence: {confidence:.1%})")
    lines.append(DISCLAIMER)
    return "\n".join(lines)
