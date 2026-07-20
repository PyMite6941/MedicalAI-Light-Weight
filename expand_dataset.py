import csv
import os
import random

CSV_PATH = "./data/dataset.csv"

def count_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))

# Symptom templates (much more varied than just the default one)
SYMPTOM_TEMPLATES = [
    "What abnormality is present in this chest X-ray?",
    "Patient presents with shortness of breath and cough. What is the diagnosis?",
    "Routine pre-operative chest X-ray. Any abnormal findings?",
    "Patient with history of smoking. Evaluate for lung pathology.",
    "Fever and productive cough for 5 days. Assess for pneumonia.",
    "Chest pain and dyspnea on exertion. Cardiac or pulmonary cause?",
    "Post-surgical follow-up. Evaluate lung expansion and complications.",
    "Patient with known COPD. Assess for acute changes or infection.",
    "Trauma patient. Evaluate for pneumothorax, hemothorax, or fractures.",
    "Immunocompromised patient with fever. Opportunistic infection?",
    "Patient with weight loss and night sweats. Evaluate for TB or malignancy.",
    "Dysphagia and regurgitation. Evaluate for hiatal hernia or mediastinal mass.",
    "Pre-employment screening chest X-ray.",
    "Congestive heart failure follow-up. Evaluate for pulmonary edema.",
    "Patient with known interstitial lung disease. Assess for progression.",
    "Central cyanosis and clubbing. Evaluate for congenital heart disease.",
    "Hemoptysis for 2 weeks. Evaluate for bronchiectasis or mass.",
    "Contact TB patient. Screening chest X-ray.",
    "Pre-operative clearance for knee replacement surgery.",
    "Rheumatoid arthritis patient with new dyspnea. Interstitial lung disease?",
    "HIV positive patient with cough and fever.",
    "Post-chemotherapy evaluation. Neutropenic fever.",
    "Patient with asbestos exposure history. Routine surveillance.",
    "Hoarseness and cough. Evaluate for mediastinal mass or recurrent laryngeal nerve involvement.",
    "Chest trauma after MVA. Evaluate for aortic injury.",
    "Patient on chronic steroids. Evaluate for opportunistic infection.",
    "Pre-renal transplant evaluation chest X-ray.",
    "Suspected foreign body aspiration in elderly patient.",
    "Evaluation for pulmonary metastasis in patient with known primary malignancy.",
    "Post-operative CABG. Evaluate for complications, effusion, or pneumothorax.",
    "Rule out tuberculosis in patient with positive PPD.",
    "Dyspnea and orthopnea. Evaluate for heart failure.",
    "Chest wall deformity. Evaluate spine and thoracic cage.",
    "Liver cirrhosis patient with dyspnea. Hepatic hydrothorax?",
    "Pancreatitis patient with respiratory distress.",
    "ARDS follow-up chest X-ray.",
    "Ventilator-associated pneumonia surveillance.",
    "Neonatal respiratory distress. Evaluate for congenital anomalies.",
    "Suspected pulmonary embolism. Evaluate for Hampton's hump or Westermark sign.",
    "Chronic cough with mucus production. Bronchiectasis evaluation.",
]

# List of image paths to reuse (cycling through existing images)
def get_existing_images():
    images = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row.get("image_path", "").strip()
            if p and p not in images:
                images.append(p)
    if not images:
        images = [f"./data\\images\\iu_xray_{i}.jpg" for i in range(1500)]
    return images

IMAGES = get_existing_images()

# Massive list of FINDINGS + IMPRESSION entries covering common AND obscure conditions
DIAGNOSIS_ENTRIES = [
    # ── Normal / No Finding ──
    {
        "diagnosis": "FINDINGS: The cardiomediastinal silhouette is within normal limits. The lungs are clear without focal consolidation, pleural effusion, or pneumothorax. The bony thorax is intact. IMPRESSION: Normal chest radiograph. No acute cardiopulmonary abnormality.",
        "labels": "No Finding",
    },
    {
        "diagnosis": "FINDINGS: Heart size normal. Mediastinal contours normal. Lungs clear bilaterally. Pulmonary vascularity normal. No pleural effusions or pneumothoraces. IMPRESSION: Normal chest examination. No active disease.",
        "labels": "No Finding",
    },
    {
        "diagnosis": "FINDINGS: Clear lungs with no infiltrates, masses, or nodules. Normal cardiac silhouette. No pleural effusion or pneumothorax. IMPRESSION: No acute cardiopulmonary disease.",
        "labels": "No Finding",
    },

    # ── Cardiomegaly / Heart Failure ──
    {
        "diagnosis": "FINDINGS: The cardiac silhouette is moderately enlarged. Pulmonary vascularity is increased with cephalization of the upper lobe vessels. Small bilateral pleural effusions. No pneumothorax. IMPRESSION: Cardiomegaly with signs of congestive heart failure and pulmonary vascular congestion.",
        "labels": "Cardiomegaly|Edema|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Severe cardiomegaly. Diffuse bilateral interstitial opacities with Kerley B lines in the costophrenic angles. Bilateral pleural effusions, right greater than left. No pneumothorax. IMPRESSION: Severe cardiomegaly with interstitial pulmonary edema and bilateral pleural effusions consistent with congestive heart failure.",
        "labels": "Cardiomegaly|Edema|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Moderate cardiomegaly. Prominent pulmonary vasculature with perihilar bat-wing distribution of airspace opacities. Small bilateral pleural effusions. No pneumothorax. IMPRESSION: Acute pulmonary edema superimposed on chronic cardiomegaly.",
        "labels": "Cardiomegaly|Edema",
    },
    {
        "diagnosis": "FINDINGS: Borderline cardiomegaly. Redistribution of pulmonary blood flow to the upper lobes. No frank pulmonary edema. No pleural effusion. IMPRESSION: Mild cardiomegaly with early pulmonary venous hypertension.",
        "labels": "Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: The cardiac silhouette is severely enlarged with a globular configuration. The pulmonary vascularity is within normal limits. No pleural effusion or pneumothorax. The lungs are clear. IMPRESSION: Severe cardiomegaly, possibly pericardial effusion. Correlation with echocardiogram recommended.",
        "labels": "Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Moderate cardiomegaly with a prominent left atrial appendage. Double density sign present. Splayed carina. No pulmonary edema. No pleural effusion. IMPRESSION: Cardiomegaly with left atrial enlargement, consider mitral valve disease.",
        "labels": "Cardiomegaly",
    },

    # ── Pneumonia (various types) ──
    {
        "diagnosis": "FINDINGS: Dense airspace consolidation in the right lower lobe with obscuration of the right hemidiaphragm. Air bronchograms present. No pleural effusion. No pneumothorax. IMPRESSION: Right lower lobe pneumonia.",
        "labels": "Consolidation|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Patchy airspace opacities in the left upper lobe with ill-defined margins. No cavitation. No effusion. No pneumothorax. IMPRESSION: Left upper lobe pneumonia. Recommend follow-up to document resolution.",
        "labels": "Consolidation|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Multifocal bilateral patchy and confluent airspace opacities in a peribronchovascular distribution. No definite cavitation. Small bilateral effusions. No pneumothorax. IMPRESSION: Multifocal pneumonia, consider atypical or viral etiology.",
        "labels": "Infiltration|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Rounded opacity in the right middle lobe with a positive silhouette sign against the right heart border. Air bronchograms visible. No effusion. IMPRESSION: Right middle lobe pneumonia (silhouette sign present).",
        "labels": "Consolidation|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Extensive left lower lobe consolidation obscuring the left hemidiaphragm and descending aorta. Air bronchograms are present. Small left pleural effusion. No pneumothorax. IMPRESSION: Left lower lobe pneumonia with parapneumonic effusion.",
        "labels": "Consolidation|Effusion|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Bilateral perihilar interstitial and airspace opacities with central distribution. No pleural effusion. No pneumothorax. IMPRESSION: Interstitial pneumonia, favor atypical/viral etiology.",
        "labels": "Infiltration|Consolidation",
    },
    {
        "diagnosis": "FINDINGS: Dense consolidation in the right upper lobe with air bronchograms and associated volume loss. Right tracheal shift. No cavitation. No effusion. IMPRESSION: Right upper lobe pneumonia with volume loss. Recommend follow-up to exclude underlying mass.",
        "labels": "Consolidation|Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Round pneumonia presenting as a spherical opacity in the left lower lobe. Surrounding ground-glass opacity. No effusion. No pneumothorax. IMPRESSION: Round pneumonia in the left lower lobe. Clinical correlation recommended.",
        "labels": "Consolidation",
    },
    {
        "diagnosis": "FINDINGS: Cavitary lesion in the right upper lobe with thick irregular wall and surrounding consolidation. Air-fluid level present. No definite pleural effusion. IMPRESSION: Cavitary pneumonia, consider TB, fungal, or necrotizing bacterial infection.",
        "labels": "Infiltration|Consolidation",
    },
    {
        "diagnosis": "FINDINGS: Bilateral lower lobe consolidations with air bronchograms. Small bilateral pleural effusions. No pneumothorax. Cardiomegaly noted. IMPRESSION: Bilateral lower lobe pneumonias superimposed on cardiomegaly.",
        "labels": "Consolidation|Infiltration|Cardiomegaly|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Segmental airspace opacity in the lingula with obscuration of the left heart border (silhouette sign). No effusion. No pneumothorax. IMPRESSION: Lingular pneumonia.",
        "labels": "Consolidation|Infiltration",
    },

    # ── Atelectasis ──
    {
        "diagnosis": "FINDINGS: Linear opacity in the right lower lobe extending to the pleura. Minor fissure elevation. No pleural effusion. No pneumothorax. IMPRESSION: Platelike/segmental atelectasis in the right lower lobe.",
        "labels": "Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Bandlike opacity in the left base with elevation of the left hemidiaphragm. Compensatory hyperinflation of the remaining lung. No effusion. IMPRESSION: Left basilar subsegmental atelectasis.",
        "labels": "Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Triangular opacity in the right base with the apex pointing toward the hilum. Right hemidiaphragm is elevated. No pleural effusion. IMPRESSION: Right lower lobe atelectasis.",
        "labels": "Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Golden S sign visible in the right upper lobe with an S-shaped curve of the minor fissure. A central hilar mass is suspected underlying the lobar atelectasis. No effusion. IMPRESSION: Right upper lobe atelectasis. The Golden S sign suggests an underlying central mass. CT chest recommended.",
        "labels": "Atelectasis|Mass",
    },
    {
        "diagnosis": "FINDINGS: Widespread platelike atelectasis in both lower lobes. Low lung volumes. Hemidiaphragms are elevated. No effusion or pneumothorax. IMPRESSION: Bilateral basilar platelike atelectasis due to hypoventilation.",
        "labels": "Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Complete opacification of the left hemithorax with mediastinal shift to the left. Compensatory hyperinflation of the right lung. No pneumothorax. IMPRESSION: Complete left lung atelectasis. Underlying obstructing lesion suspected. Urgent CT and bronchoscopy recommended.",
        "labels": "Atelectasis|Mass",
    },
    {
        "diagnosis": "FINDINGS: Rounded atelectasis in the right lower lobe presenting as a rounded opacity with comet-tail sign. Pleural thickening adjacent. No change from prior study. IMPRESSION: Rounded atelectasis, stable. No evidence of active disease.",
        "labels": "Atelectasis|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Discoid atelectasis in the right midlung. Minimal volume loss. No pleural effusion. Heart size normal. IMPRESSION: Discoid atelectasis. Otherwise unremarkable chest.",
        "labels": "Atelectasis",
    },

    # ── Pleural Effusion ──
    {
        "diagnosis": "FINDINGS: Moderate right pleural effusion with blunting of the right costophrenic angle and a meniscus sign. Underlying compressive atelectasis. No pneumothorax. IMPRESSION: Moderate right pleural effusion with adjacent atelectasis.",
        "labels": "Effusion|Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Large left pleural effusion causing near-complete opacification of the left hemithorax with mediastinal shift to the right. No pneumothorax. IMPRESSION: Large left pleural effusion with mass effect. Thoracentesis recommended.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Bilateral small pleural effusions with blunting of both costophrenic angles. No layering on lateral decubitus. No pneumothorax. IMPRESSION: Small bilateral pleural effusions.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Loculated right pleural effusion with biconvex opacity along the lateral chest wall extending into the fissure. No free-flowing component. No pneumothorax. IMPRESSION: Loculated right pleural effusion, consider empyema or hemothorax.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Massive right pleural effusion with complete opacification of the right hemithorax and contralateral mediastinal shift. Left lung clear. No pneumothorax. IMPRESSION: Massive right pleural effusion. Therapeutic thoracentesis indicated.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Small right pleural effusion with intact meniscus sign. No pleural thickening. No loculation. Lungs are clear. IMPRESSION: Small right pleural effusion. Clinical correlation for etiology recommended.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Bilateral pleural effusions with associated basal atelectasis. No pneumothorax. Cardiomegaly noted. Pulmonary vascular congestion present. IMPRESSION: Bilateral pleural effusions in the setting of congestive heart failure.",
        "labels": "Effusion|Cardiomegaly|Atelectasis",
    },

    # ── Pneumothorax ──
    {
        "diagnosis": "FINDINGS: Right apical pneumothorax with visceral pleural line visible approximately 2 cm from the chest wall. No mediastinal shift. The underlying lung is clear. No effusion. IMPRESSION: Small right apical pneumothorax.",
        "labels": "Pneumothorax",
    },
    {
        "diagnosis": "FINDINGS: Moderate left pneumothorax with the lung edge seen at 3 cm from the lateral chest wall. Mild mediastinal shift to the right. No pleural effusion. IMPRESSION: Moderate left pneumothorax with early tension physiology.",
        "labels": "Pneumothorax",
    },
    {
        "diagnosis": "FINDINGS: Large right pneumothorax with complete collapse of the right lung (lung appears as a dense hilar mass). Significant mediastinal shift to the left. Depressed right hemidiaphragm. Deep sulcus sign present. IMPRESSION: Large tension pneumothorax on the right. Requires immediate decompression.",
        "labels": "Pneumothorax",
    },
    {
        "diagnosis": "FINDINGS: Small left apical pneumothorax. No mediastinal shift. No pleural effusion. Remainder of lung is clear. Heart size normal. IMPRESSION: Small spontaneous pneumothorax.",
        "labels": "Pneumothorax",
    },
    {
        "diagnosis": "FINDINGS: Hydropneumothorax on the left with visible air-fluid level extending across the hemithorax. Partially collapsed left lung. Mediastinal shift to the right. IMPRESSION: Left hydropneumothorax. Consider empyema or bronchopleural fistula.",
        "labels": "Pneumothorax|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Tension pneumothorax on the right with deep sulcus sign, mediastinal shift to the left, and flattening of the right heart border. Complete right lung collapse. IMPRESSION: Large right tension pneumothorax, life-threatening. STAT decompression indicated.",
        "labels": "Pneumothorax",
    },
    {
        "diagnosis": "FINDINGS: Tiny right apical pneumothorax barely visible. Visceral pleural line seen only on inspiratory film. Less than 1 cm from chest wall. No mediastinal shift. IMPRESSION: Very small right apical pneumothorax, likely will resolve spontaneously.",
        "labels": "Pneumothorax",
    },

    # ── COPD / Emphysema ──
    {
        "diagnosis": "FINDINGS: Hyperinflated lungs with flattened hemidiaphragms. Increased AP chest diameter. Bullous changes at the apices. Heart size is normal. No pneumothorax or effusion. IMPRESSION: Chronic obstructive pulmonary disease with emphysematous changes.",
        "labels": "Emphysema",
    },
    {
        "diagnosis": "FINDINGS: Severe hyperinflation with flattened and depressed hemidiaphragms. Widened intercostal spaces. Increased retrosternal clear space. Small heart. No focal consolidation. No pneumothorax. IMPRESSION: Severe emphysema. No acute infiltrate.",
        "labels": "Emphysema",
    },
    {
        "diagnosis": "FINDINGS: Hyperlucent lungs with attenuation of peripheral vascular markings. Large central pulmonary arteries. Flattened hemidiaphragms. No bullae. No pneumothorax. IMPRESSION: Emphysema with pulmonary hypertension signs. Cor pulmonale suspected.",
        "labels": "Emphysema",
    },
    {
        "diagnosis": "FINDINGS: Severe bilateral bullous emphysema. Large thin-walled bullae occupy more than one-third of both hemithoraces. No pneumothorax. No pleural effusion. Cardiac silhouette is vertically oriented and small. IMPRESSION: Severe bilateral bullous emphysema.",
        "labels": "Emphysema",
    },
    {
        "diagnosis": "FINDINGS: Moderate hyperinflation of the lungs with flattened hemidiaphragms. Subtle reticular opacities in the lung bases. Heart size normal. No effusion or pneumothorax. IMPRESSION: COPD with mild interstitial changes.",
        "labels": "Emphysema|Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Hyperexpanded lungs with increased retrosternal airspace. Mild flattening of the diaphragms. No focal consolidation. Heart size is normal. No pneumothorax. IMPRESSION: Mild hyperinflation, consistent with early COPD.",
        "labels": "Emphysema",
    },

    # ── Nodules and Masses ──
    {
        "diagnosis": "FINDINGS: Solitary pulmonary nodule in the right upper lobe measuring approximately 1.5 cm. Margins are smooth. No calcification. No associated adenopathy. No pleural effusion. IMPRESSION: Solitary pulmonary nodule. Recommend CT chest for further characterization.",
        "labels": "Nodule|Mass",
    },
    {
        "diagnosis": "FINDINGS: Spiculated mass in the left upper lobe measuring 3.2 x 2.8 cm. Associated pleural tail. No cavitation. No calcification. No hilar or mediastinal adenopathy. No pleural effusion. IMPRESSION: Spiculated left upper lobe mass suspicious for primary lung malignancy. CT and tissue sampling recommended.",
        "labels": "Mass|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Multiple bilateral pulmonary nodules of varying sizes, ranging from 0.5 to 2.0 cm. Random distribution. No cavitation. No calcification. Small right pleural effusion. No pneumothorax. IMPRESSION: Numerous bilateral pulmonary nodules consistent with metastatic disease.",
        "labels": "Nodule|Mass|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Cavitating mass in the right upper lobe with thick, irregular walls measuring up to 1.5 cm in thickness. Air-fluid level present. Surrounding ground-glass opacity. No effusion. IMPRESSION: Cavitating lung mass, differential includes primary lung carcinoma, abscess, or fungal infection.",
        "labels": "Mass|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Pancoast tumor in the right apex with associated apical cap thickening and destruction of the posterior right first and second ribs. No mediastinal widening. No effusion. IMPRESSION: Right apical mass (Pancoast tumor) with chest wall invasion. Urgent CT and biopsy recommended.",
        "labels": "Mass",
    },
    {
        "diagnosis": "FINDINGS: Well-defined, smoothly marginated nodule in the left lower lobe with popcorn calcification. No growth compared to prior study. No effusion. No pneumothorax. IMPRESSION: Hamartoma with characteristic popcorn calcification. Benign appearance.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Subsolid nodule with ground-glass and solid components (part-solid nodule) in the right middle lobe. About 1.2 cm. No calcification. No pleural effusion. IMPRESSION: Part-solid nodule suspicious for adenocarcinoma spectrum. CT follow-up recommended.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Numerous small, well-defined nodules in a perilymphatic distribution with fissural nodularity and right paratracheal adenopathy. No pleural effusion. No pneumothorax. IMPRESSION: Perilymphatic nodules and mediastinal adenopathy, consider sarcoidosis or lymphangitic carcinomatosis.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Solitary pulmonary nodule in the right lower lobe measuring 8 mm. Contains central calcification. No spiculation. No adenopathy. No effusion. IMPRESSION: Calcified granuloma. Benign appearance, no further action required.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Large 6 cm mass in the left lower lobe with irregular borders and central necrosis. Left hilar adenopathy present. No pleural effusion. No pneumothorax. IMPRESSION: Large left lower lobe mass with hilar adenopathy, highly suspicious for malignancy.",
        "labels": "Mass|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Anterior mediastinal mass with lobulated contours and no calcification. Trachea is midline. No pleural or pericardial effusion. Lungs are clear. IMPRESSION: Anterior mediastinal mass, differential includes thymoma, lymphoma, or germ cell tumor. CT with contrast recommended.",
        "labels": "Mass",
    },
    {
        "diagnosis": "FINDINGS: Middle mediastinal mass causing splaying of the carina. No calcification. No pleural effusion. Lungs are clear. IMPRESSION: Subcarinal mediastinal mass/massive lymphadenopathy. CT chest recommended for further evaluation.",
        "labels": "Mass",
    },

    # ── Interstitial Lung Disease / Fibrosis ──
    {
        "diagnosis": "FINDINGS: Diffuse bilateral reticular opacities with honeycombing in the lung bases. Traction bronchiectasis present. No pneumothorax. No pleural effusion. Normal heart size. IMPRESSION: Usual interstitial pneumonia (UIP) pattern with honeycombing and traction bronchiectasis. Consistent with idiopathic pulmonary fibrosis.",
        "labels": "Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral ground-glass opacities in the lower lobes with reticular superimposed opacities. Minimal honeycombing. No pleural effusion. No pneumothorax. IMPRESSION: Nonspecific interstitial pneumonia (NSIP) pattern. Clinical correlation recommended.",
        "labels": "Fibrosis|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Extensive bilateral interstitial opacities with a perilymphatic distribution. Septal thickening and subpleural nodules. Bilateral hilar and right paratracheal lymphadenopathy. No effusion. IMPRESSION: Pulmonary sarcoidosis with bilateral hilar adenopathy and interstitial lung disease (stage II sarcoidosis).",
        "labels": "Fibrosis|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Bilateral upper lobe predominant fibrotic changes with volume loss, hilar retraction, and architectural distortion. Traction bronchiectasis. No pneumothorax. IMPRESSION: Chronic upper lobe fibrotic changes, consider post-TB sequelae or radiation fibrosis.",
        "labels": "Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral ground-glass opacities with superimposed reticular opacities and traction bronchiectasis in the lung bases. No honeycombing. No effusion. IMPRESSION: Probable interstitial lung disease. HRCT recommended for further characterization.",
        "labels": "Fibrosis|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Bilateral apical pleural thickening with subpleural fibrotic bands. Upper lobe volume loss with upward hilar retraction. No cavitation. No pneumothorax. IMPRESSION: Chronic apical fibrosis with pleural thickening, likely post-inflammatory.",
        "labels": "Fibrosis|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Crazy-paving pattern with scattered ground-glass opacities superimposed on interlobular septal thickening in the bilateral lower lobes. No pleural effusion. No pneumothorax. IMPRESSION: Crazy-paving pattern. Differential includes alveolar proteinosis, lipoid pneumonia, or cardiogenic pulmonary edema. Clinical correlation recommended.",
        "labels": "Infiltration|Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral reticulonodular opacities in a mid-to-upper lung distribution. Eggshell calcifications in bilateral hilar lymph nodes. No pleural effusion. IMPRESSION: Silicosis with eggshell calcification of hilar lymph nodes and reticulonodular interstitial lung disease.",
        "labels": "Fibrosis|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Bilateral pleural plaques with calcification along the diaphragmatic pleura and lateral chest wall. No pleural effusion. Lungs are otherwise clear. No pneumothorax. IMPRESSION: Bilateral pleural plaques in a patient with asbestos exposure history. No evidence of asbestosis or mesothelioma.",
        "labels": "Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral fine reticulonodular opacities predominantly in the lower lobes. No honeycombing. No pleural effusion. Normal heart size. IMPRESSION: Early interstitial lung disease, consider connective tissue disease-related ILD or hypersensitivity pneumonitis.",
        "labels": "Fibrosis|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Lymphangitic carcinomatosis presenting with unilateral right-sided septal thickening and peribronchial cuffing. Right hilar adenopathy. Small right pleural effusion. No pneumothorax. IMPRESSION: Right-sided lymphangitic carcinomatosis, consistent with known primary malignancy.",
        "labels": "Infiltration|Effusion|Nodule",
    },

    # ── Pulmonary Edema ──
    {
        "diagnosis": "FINDINGS: Bilateral perihilar airspace opacities with a butterfly-wing pattern. Upper lobe pulmonary vascular redistribution. Kerley B lines at the lung bases. Cardiomegaly. Small bilateral pleural effusions. No pneumothorax. IMPRESSION: Acute pulmonary edema due to congestive heart failure.",
        "labels": "Edema|Cardiomegaly|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral airspace opacities that are more confluent centrally with peripheral sparing. No cardiomegaly. No pleural effusion. Normal heart size. No pneumothorax. IMPRESSION: Noncardiogenic pulmonary edema (ARDS pattern). Clinical correlation needed.",
        "labels": "Edema|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Mild interstitial pulmonary edema with peribronchial cuffing, indistinct vascular margins, and Kerley A and B lines. Cardiomegaly is mild. No pleural effusion. No pneumothorax. IMPRESSION: Mild interstitial pulmonary edema, early congestive heart failure.",
        "labels": "Edema|Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Asymmetric left-sided pulmonary edema with a perihilar distribution. Patchy airspace opacities in the left upper and lower lobes. No pleural effusion. Normal heart size. No pneumothorax. IMPRESSION: Asymmetric pulmonary edema. Consider acute mitral regurgitation or localized lung pathology.",
        "labels": "Edema|Infiltration",
    },

    # ── Tuberculosis ──
    {
        "diagnosis": "FINDINGS: Right apical fibronodular opacities with associated volume loss and upward hilar retraction. Small calcified granuloma in the right apex. No cavitation. No pleural effusion. No pneumothorax. IMPRESSION: Right apical fibronodular changes consistent with prior granulomatous disease, likely old TB. No evidence of active disease.",
        "labels": "Fibrosis|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Cavitary lesion in the right upper lobe with surrounding airspace disease. Tree-in-bud opacities in the right upper lobe. Right paratracheal adenopathy. No pleural effusion. IMPRESSION: Right upper lobe cavitary lesion with tree-in-bud opacities suspicious for active pulmonary tuberculosis. Sputum AFB recommended.",
        "labels": "Infiltration|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Military pattern with innumerable tiny 1-2 mm nodules diffusely throughout both lungs. No consolidation. No pleural effusion. Heart size normal. No pneumothorax. IMPRESSION: Military nodules, highly suspicious for miliary tuberculosis. Urgent clinical correlation and AFB cultures recommended.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Left apical pleural thickening with an associated fibrotic band extending to the hilum. Calcified left hilar lymph node. Surrounding parenchyma shows traction bronchiectasis. No effusion. IMPRESSION: Old tuberculosis with apical pleural thickening and fibrotic scarring. Stable appearance.",
        "labels": "Fibrosis|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Bilateral upper lobe fibrocavitary disease with thick-walled cavities and surrounding retraction. Elevated hila. Compensatory hyperinflation of lower lobes. No pneumothorax. Minimal pleural thickening. IMPRESSION: Chronic fibrocavitary TB with bilateral upper lobe involvement. Active disease cannot be excluded.",
        "labels": "Fibrosis|Infiltration",
    },

    # ── Bronchiectasis ──
    {
        "diagnosis": "FINDINGS: Dilated, thickened bronchi seen end-on as tram-track opacities in the bilateral lower lobes. Bronchial wall thickening. Some mucus plugging. No consolidation. No pleural effusion. No pneumothorax. IMPRESSION: Bilateral lower lobe bronchiectasis with peribronchial thickening and mucus plugging.",
        "labels": "Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Severe cystic bronchiectasis in both lower lobes with thin-walled cysts and air-fluid levels. Signet-ring sign present. Scattered tree-in-bud opacities. No pneumothorax. No pleural effusion. IMPRESSION: Cystic bronchiectasis with superinfection. Clinical correlation and sputum culture recommended.",
        "labels": "Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Tram-track and ring-shadow opacities in the right middle lobe and lingula. Volume loss in the right middle lobe. No pleural effusion. No pneumothorax. IMPRESSION: Right middle lobe and lingular bronchiectasis with volume loss.",
        "labels": "Infiltration|Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Central bronchiectasis with dilated, mucus-filled bronchi appearing as glove-finger opacities radiating from the hila. Tree-in-bud opacities in the surrounding parenchyma. No effusion. No pneumothorax. IMPRESSION: Central bronchiectasis with mucoid impaction. Consider allergic bronchopulmonary aspergillosis (ABPA) if asthmatic.",
        "labels": "Infiltration",
    },

    # ── Rib Fractures / Trauma ──
    {
        "diagnosis": "FINDINGS: Minimally displaced fracture of the right lateral 7th rib. Small associated right pleural effusion. No pneumothorax. Lungs are otherwise clear. IMPRESSION: Right 7th rib fracture with small adjacent pleural effusion.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Multiple left-sided rib fractures involving ribs 4-8 with a flail segment. Large left hemothorax with nearly complete opacification of the left hemithorax. Mediastinal shift to the right. No pneumothorax. IMPRESSION: Left flail chest with massive hemothorax. Cardiac contusion may be present.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Fracture of the left clavicle with inferior displacement. No pneumothorax. No hemothorax. Lungs are clear. Heart and mediastinum normal. IMPRESSION: Isolated left clavicular fracture. No thoracic injury.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Multiple bilateral old rib fractures with callus formation and cortical irregularity. No acute fracture line identified. No pneumothorax. No pleural effusion. Lungs clear. IMPRESSION: Multiple old healed rib fractures. No acute traumatic findings.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Superior sulcus opacity on the right with thickened pleura and erosion of the posterior right first rib. Correlate clinically for Pancoast syndrome. No effusion. No pneumothorax. IMPRESSION: Right apical mass with rib destruction concerning for Pancoast tumor.",
        "labels": "Mass",
    },
    {
        "diagnosis": "FINDINGS: Sternal fracture with mild displacement noted on lateral view. No pneumothorax. No hemothorax. No mediastinal widening. Heart size normal. IMPRESSION: Sternal fracture without underlying cardiac or aortic injury.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Wide mediastinum measuring >8 cm on AP view. Loss of aortic knob contour. Left apical cap. Left pleural effusion. No pneumothorax. IMPRESSION: Traumatic aortic injury suggested by mediastinal widening, apical cap, and left pleural effusion. Urgent CT aortogram and surgical consultation required.",
        "labels": "Effusion",
    },

    # ── Hiatal Hernia ──
    {
        "diagnosis": "FINDINGS: Large retrocardiac air-fluid level behind the heart. Visible herniated stomach contents above the diaphragm. Mild compressive atelectasis of the left lower lobe. Heart size normal. No pneumothorax. IMPRESSION: Large hiatal hernia with stomach herniated into the chest. No acute obstructive findings.",
        "labels": "Hernia",
    },
    {
        "diagnosis": "FINDINGS: Moderate hiatal hernia seen on the lateral view as a retrocardiac opacity with air-fluid level. The hernia contains both stomach and colon. No evidence of obstruction. No pleural effusion. IMPRESSION: Large hiatal hernia containing stomach and colon, likely chronic.",
        "labels": "Hernia",
    },
    {
        "diagnosis": "FINDINGS: Small hiatal hernia noted incidentally. No air-fluid level. No associated atelectasis. Normal cardiomediastinal silhouette. IMPRESSION: Small hiatal hernia, otherwise normal chest.",
        "labels": "Hernia",
    },
    {
        "diagnosis": "FINDINGS: Large hiatal hernia with a mixed gas and soft tissue density in the retrocardiac region. The herniated stomach shows evidence of volvulus with two air-fluid levels at different heights. No pneumothorax. No pleural effusion. IMPRESSION: Large hiatal hernia with gastric volvulus. Urgent surgical evaluation recommended.",
        "labels": "Hernia",
    },

    # ── Pneumoperitoneum / Free Air ──
    {
        "diagnosis": "FINDINGS: Free air under the right hemidiaphragm on upright PA view. Air outlining the liver. No pneumothorax. Lungs are clear. Heart size normal. IMPRESSION: Pneumoperitoneum suggesting hollow viscus perforation. Urgent surgical evaluation recommended.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Rigler sign visible with air outlining both sides of the bowel wall. Free air under both hemidiaphragms. Visible falciform ligament. No pneumothorax. IMPRESSION: Massive pneumoperitoneum. Emergency surgical consultation required.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Football sign in the supine AP view with large oval lucency over the upper abdomen. Air outlining the peritoneal cavity. No pneumothorax. IMPRESSION: Large pneumoperitoneum (football sign). Suspect perforated viscus.",
        "labels": "",
    },

    # ── Aortic Aneurysm / Dissection ──
    {
        "diagnosis": "FINDINGS: Prominent thoracic aortic knob with calcification. Mediastinal width is at the upper limits of normal. No definite widening. Lungs are clear. IMPRESSION: Tortuous calcified aorta without definite aneurysm.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Wide mediastinum with loss of the aortic knob contour. Tracheal deviation to the right. Left pleural effusion. No pneumothorax. Heart size normal. IMPRESSION: Wide mediastinum concerning for aortic dissection. CT aortogram recommended urgently.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Prominent descending thoracic aorta with calcified walls. Measured 4.5 cm in diameter. No dissection flap visible on X-ray. No pleural effusion. Lungs clear. IMPRESSION: Descending thoracic aortic aneurysm measuring 4.5 cm. CT with contrast recommended for further characterization.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Calcified aortic arch aneurysm projecting to the right of the trachea. No definite dissection. No pleural effusion. No pneumothorax. IMPRESSION: Aortic arch aneurysm. Echocardiogram or CT for further evaluation recommended.",
        "labels": "",
    },

    # ── Pericardial Effusion ──
    {
        "diagnosis": "FINDINGS: Globular enlargement of the cardiac silhouette with a water-bottle configuration. Clear sharp borders. Lungs are clear. No pleural effusion. Pulmonary vascularity is normal. No pneumothorax. IMPRESSION: Large pericardial effusion giving a water-bottle heart appearance. Echocardiogram recommended.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Mild enlargement of the cardiac silhouette. Epicardial fat pad sign visible with a lucent line separating the heart from the pericardium. Lungs clear. No pleural effusion. IMPRESSION: Small to moderate pericardial effusion. Echocardiogram for confirmation.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Massive globular cardiomegaly with clear lungs. No pulmonary vascular congestion. No pleural effusion. No pneumothorax. IMPRESSION: Large pericardial effusion. Rule out pericardial tamponade. Urgent echocardiogram indicated.",
        "labels": "",
    },

    # ── Pulmonary Embolism ──
    {
        "diagnosis": "FINDINGS: Hampton's hump: wedge-shaped pleural-based opacity in the right costophrenic angle. Small right pleural effusion. No pneumothorax. Heart size normal. IMPRESSION: Wedge-shaped opacity consistent with pulmonary infarct, likely secondary to pulmonary embolism. CT pulmonary angiogram recommended.",
        "labels": "Effusion|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Westermark sign: oligemia of the left lung with decreased vascular markings. Prominent central pulmonary artery on the left (Fleischner sign). No pleural effusion. No pneumothorax. IMPRESSION: Westermark sign suggesting pulmonary embolism in the left pulmonary artery. Urgent CT pulmonary angiography recommended.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Enlarged right descending pulmonary artery (Palla sign). Right lower lobe opacity consistent with infarct. Small right pleural effusion. No pneumothorax. IMPRESSION: Right pulmonary artery enlargement with adjacent infarct, concerning for pulmonary embolism. CTPA recommended.",
        "labels": "Effusion|Infiltration",
    },

    # ── Pediatric / Congenital ──
    {
        "diagnosis": "FINDINGS: Thymic sail sign present with a triangular soft tissue density projecting from the right superior mediastinum. Lungs clear. Heart size normal. No pneumothorax. IMPRESSION: Normal thymus with sail sign. No abnormality in this pediatric chest.",
        "labels": "No Finding",
    },
    {
        "diagnosis": "FINDINGS: Scimitar sign: curved tubular opacity in the right lower lobe coursing toward the right cardiophrenic angle. Right lung is hypoplastic. Mediastinal shift to the right. Heart is dextroposed. IMPRESSION: Scimitar syndrome (hypogenetic right lung syndrome with partial anomalous pulmonary venous return).",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Boot-shaped heart with upturned cardiac apex, prominent right ventricle, and concave pulmonary artery segment (coeur en sabot). Decreased pulmonary vascularity. No pleural effusion. IMPRESSION: Tetralogy of Fallot with classic boot-shaped heart. Surgical evaluation recommended.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Egg-on-side cardiac silhouette with narrow vascular pedicle. Increased pulmonary vascularity. No pleural effusion. No pneumothorax. IMPRESSION: Transposition of the great arteries with egg-on-side heart. Neonatal cardiology evaluation required.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Figure-of-3 sign in the aortic knob with rib notching of the posterior inferior aspects of ribs 3-8 bilaterally. Heart size normal. Lungs clear. No pleural effusion. IMPRESSION: Coarctation of the aorta with classic figure-of-3 sign and rib notching due to collateral circulation.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Dilated azygos vein seen as a comma-shaped density at the right tracheobronchial angle. No mediastinal mass. Lungs are clear. Heart size normal. IMPRESSION: Prominent azygos vein, likely due to azygos continuation of the IVC or congenital anomaly.",
        "labels": "",
    },

    # ── Pneumoconiosis ──
    {
        "diagnosis": "FINDINGS: Small rounded nodular opacities (p, q, r type) in the upper and mid lung zones bilaterally. Eggshell calcifications of hilar lymph nodes. No pleural plaques. No pneumothorax. No pleural effusion. IMPRESSION: Simple silicosis with eggshell calcification of hilar nodes.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Bilateral upper lobe predominantly large opacities (progressive massive fibrosis) with surrounding emphysema. Hilar retraction and architectural distortion. Eggshell calcification of hilar nodes. No pneumothorax. IMPRESSION: Complicated silicosis with progressive massive fibrosis (PMF).",
        "labels": "Fibrosis|Mass",
    },
    {
        "diagnosis": "FINDINGS: Bilateral diaphragmatic and lateral pleural plaques with calcification. No pleural effusion. Lungs are clear. No pneumothorax. Heart size normal. IMPRESSION: Bilateral pleural plaques due to asbestos exposure. No evidence of asbestosis or mesothelioma at this time.",
        "labels": "Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Bilateral lower lobe interstitial fibrosis with subpleural lines and honeycombing. Bilateral calcified pleural plaques. No pleural effusion. No pneumothorax. IMPRESSION: Asbestosis with parenchymal fibrosis and bilateral pleural plaques.",
        "labels": "Fibrosis|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral fine nodular opacities in a mid-to-upper lung zone predominance. Minimal hilar adenopathy. No pleural involvement. No pneumothorax. IMPRESSION: Simple coal workers pneumoconiosis (CWP) with upper lobe predominant nodular opacities.",
        "labels": "Nodule",
    },

    # ── Rare/Obscure Conditions ──
    {
        "diagnosis": "FINDINGS: Extensive bilateral ground-glass opacities with interlobular septal thickening (crazy-paving pattern) in a geographic distribution. No pleural effusion. No pneumothorax. IMPRESSION: Crazy-paving pattern, suspicious for pulmonary alveolar proteinosis. BAL and HRCT recommended.",
        "labels": "Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Bilateral diffuse micronodular opacities with a mid-to-upper lung predominance. Multiple thin-walled cysts of varying sizes. No pneumothorax. No pleural effusion. IMPRESSION: Langerhans cell histiocytosis (LCH) with characteristic cysts and nodules in a smoker. HRCT for confirmation.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Bilateral large thin-walled cysts with normal intervening lung parenchyma. Predominantly lower lobe distribution. No pneumothorax. No pleural effusion. No nodules. IMPRESSION: Lymphangioleiomyomatosis (LAM) with bilateral thin-walled cysts. HRCT recommended. Rule out tuberous sclerosis.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Bilateral perihilar and lower zone opacities with consolidation and air bronchograms. Spontaneous pneumothorax noted on the right. No pleural effusion. IMPRESSION: Spontaneous pneumothorax with underlying consolidation. Consider pulmonary contusion or alveolar hemorrhage.",
        "labels": "Pneumothorax|Consolidation",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral airspace opacities with central predominance and air bronchograms. Normal heart size. No pleural effusion. No pneumothorax. IMPRESSION: Acute respiratory distress syndrome (ARDS) with bilateral diffuse airspace disease.",
        "labels": "Infiltration|Consolidation",
    },
    {
        "diagnosis": "FINDINGS: Swyer-James syndrome: hyperlucent left lung with diminished vascular markings. Expiratory film shows air trapping on the left. Left lung is smaller than the right. Heart shifted to the left. No pneumothorax. IMPRESSION: Unilateral hyperlucent lung (Swyer-James/MacLeod syndrome), likely post-infectious in childhood.",
        "labels": "Emphysema",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral ground-glass opacities and consolidation with a peripheral predominance, sparing the costophrenic angles. Reverse halo sign (atoll sign) visible. No pleural effusion. No pneumothorax. IMPRESSION: Organizing pneumonia pattern. Clinical and histopathological correlation recommended.",
        "labels": "Consolidation|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Multiple cavitary nodules of varying sizes in both lungs with thick walls. Some nodules show feeding vessel signs. No pleural effusion. No pneumothorax. IMPRESSION: Septic pulmonary emboli with cavitation. Sought for source of infection.",
        "labels": "Nodule|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Bilateral basal and peripheral ground-glass opacities with consolidation. Areas of sparing. Normal heart size. Small left pleural effusion. No pneumothorax. IMPRESSION: Acute eosinophilic pneumonia or cryptogenic organizing pneumonia. Clinical history of eosinophilia needed.",
        "labels": "Consolidation|Effusion|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Extensive bilateral reticulonodular opacities with a basal predominance. Bronchiectasis and architectural distortion. No pneumothorax. No pleural effusion. IMPRESSION: Rheumatoid arthritis-associated interstitial lung disease with a usual interstitial pneumonia (UIP) pattern.",
        "labels": "Fibrosis|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Scleroderma lung: bilateral lower lobe ground-glass and reticular opacities with early honeycombing. Dilated esophagus with air-fluid level. No pleural effusion. No pneumothorax. IMPRESSION: Systemic sclerosis-associated interstitial lung disease with NSIP pattern and esophageal dilatation.",
        "labels": "Fibrosis|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Bilateral symmetric lower lobe consolidation with air bronchograms. Kerley B lines and small bilateral pleural effusions present. Heart size normal. No pneumothorax. IMPRESSION: Acute interstitial pneumonia (AIP/Hamman-Rich syndrome). Rapid progression suggests this entity.",
        "labels": "Consolidation|Effusion|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Multiple subcentimeter nodules with a perilymphatic distribution and bilateral hilar adenopathy. No pleural effusion. No pneumothorax. Lungs otherwise clear. IMPRESSION: Stage I pulmonary sarcoidosis with bilateral hilar lymphadenopathy (1-2-3 sign). No parenchymal involvement.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Bilateral hilar and right paratracheal lymphadenopathy with associated upper lobe reticulonodular opacities. No honeycombing. No pleural effusion. No pneumothorax. IMPRESSION: Stage II pulmonary sarcoidosis with bilateral hilar adenopathy and parenchymal involvement.",
        "labels": "Nodule|Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral fine nodules with upper lobe predominance. Mild hilar adenopathy. No fibrosis. No pleural effusion. IMPRESSION: Hypersensitivity pneumonitis presenting with subacute stage. Exposure history correlation recommended.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Extensive bilateral consolidations and ground-glass opacities with air bronchograms and air-fluid levels. No definite pleural effusion. No pneumothorax. IMPRESSION: Diffuse alveolar hemorrhage. Consider vasculitis or coagulopathy. Clinical correlation with hemoptysis status urgent.",
        "labels": "Consolidation",
    },
    {
        "diagnosis": "FINDINGS: Unilateral right perihilar mass with obstructive pneumonitis and Golden S sign. Right upper lobe volume loss. No pleural effusion. No pneumothorax. IMPRESSION: Central right lung mass with obstructive atelectasis (Golden S sign suspicious for lung carcinoma). Bronchoscopy and CT recommended.",
        "labels": "Mass|Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Multiple randomly distributed nodules throughout both lungs with some showing halo sign (ground-glass surrounding nodule). No cavitation. No pleural effusion. No pneumothorax. IMPRESSION: Random nodules with halo sign. Consider fungal infection (aspergillosis, mucormycosis) in immunocompromised patient.",
        "labels": "Nodule|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Air crescent sign in a preexisting cavity with a round opacity inside. Right upper lobe cavity with intracavitary mass. No pleural effusion. No pneumothorax. IMPRESSION: Aspergilloma (fungus ball) in a preexisting cavity with air crescent sign.",
        "labels": "Nodule|Mass",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral fine miliary nodules with a random distribution. Bilateral hilar adenopathy. No pleural effusion. No pneumothorax. IMPRESSION: Miliary tuberculosis versus metastatic disease. Clinical history and sputum studies recommended.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Situs inversus with cardiac apex, aortic knob, and stomach bubble on the right. Dextrocardia with normal cardiac situs. Lungs clear. No pneumothorax. IMPRESSION: Dextrocardia with situs inversus totalis. Consider Kartagener syndrome if bronchiectasis present.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Pectus excavatum with depressed sternum, right-sided cardiac displacement, and increased retrosternal airspace. Heart size normal. Lungs clear. No pneumothorax. No pleural effusion. IMPRESSION: Pectus excavatum deformity with cardiac displacement. Otherwise normal chest.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Absence of the right pectoralis muscle with hyperlucency of the right hemithorax. No mediastinal shift. No lung herniation. Lungs clear. No pneumothorax. IMPRESSION: Poland syndrome with absent right pectoralis muscle and right-sided hyperlucency.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Cervical rib arising from the C7 vertebra bilaterally. No associated thoracic outlet syndrome findings. Lungs clear. Heart size normal. No pneumothorax. IMPRESSION: Bilateral cervical ribs, an incidental finding.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Azygos lobe with a visible azygos fissure forming a tear-drop shaped opacity at the right apex. The azygos vein is seen at the base of the fissure. Lungs clear. No other abnormality. IMPRESSION: Azygos lobe, a normal variant. No pathological significance.",
        "labels": "No Finding",
    },
    {
        "diagnosis": "FINDINGS: Diffuse interstitial pulmonary calcification with bilateral dense nodular and conglomerate opacities. No pleural effusion. No pneumothorax. Heart size normal. IMPRESSION: Metastatic pulmonary calcification, likely secondary to chronic renal failure or hyperparathyroidism.",
        "labels": "Nodule|Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Unilateral right lung hyperlucency with diminished vascular markings. Expiratory views show air trapping. No mediastinal shift. No pneumothorax. IMPRESSION: Swyer-James syndrome (post-infectious obliterative bronchiolitis) causing right lung hyperlucency.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Bilateral symmetrical consolidation with air bronchograms in the lower lobes. Rapid progression over 24 hours. No pleural effusion. No pneumothorax. Normal heart size. IMPRESSION: Rapid progression of bilateral airspace disease suspicious for ARDS or diffuse alveolar hemorrhage. Clinical history critical.",
        "labels": "Consolidation",
    },
    {
        "diagnosis": "FINDINGS: Large bulla in the right upper lobe occupying more than 30% of the hemithorax. Compressed adjacent lung. No pneumothorax. No pleural effusion. Remaining lung is hyperexpanded and shows emphysematous change. IMPRESSION: Giant bulla in the right upper lobe. Consider bullectomy if symptomatic.",
        "labels": "Emphysema",
    },
    {
        "diagnosis": "FINDINGS: Bilateral apical pleural thickening with associated fibrotic band. No definite mass. No cavitation. No calcification. No pleural effusion. No pneumothorax. IMPRESSION: Biapical pleural thickening, likely post-inflammatory. If new or progressive, consider Pancoast tumor and CT correlation.",
        "labels": "Pleural_Thickening|Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Extensive bilateral nodular and reticular opacities in a perihilar distribution with traction bronchiectasis. No honeycombing. No pleural effusion. No pneumothorax. IMPRESSION: Lymphangitic carcinomatosis. Known history of malignancy correlates with this appearance.",
        "labels": "Nodule|Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Air-fluid level in a preexisting cystic lesion in the right lower lobe. Surrounding consolidation. The cyst appears to have an intracavitary mass. No pneumothorax. Small pleural effusion on the right. IMPRESSION: Infected lung cyst with possible intracavitary mass, likely an aspergilloma or lung abscess.",
        "labels": "Consolidation|Effusion|Mass",
    },
    {
        "diagnosis": "FINDINGS: Bilateral subpleural reticular opacities with basal honeycombing and traction bronchiectasis. Progressive from prior exam. No pneumothorax. No pleural effusion. Normal heart size. IMPRESSION: Idiopathic pulmonary fibrosis with decline and honeycombing progression. Pulmonary function correlation recommended.",
        "labels": "Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Fat-fluid level and air-fluid level within a large loculated pleural collection on the left. Pneumothorax is absent. The underlying lung is partially compressed. IMPRESSION: Empyema necessitans or complicated parapneumonic effusion with air-fluid level. Diagnostic thoracentesis recommended.",
        "labels": "Effusion|Pneumothorax",
    },
    {
        "diagnosis": "FINDINGS: Bilaterally enlarged pulmonary arteries with pruning of peripheral vessels. Right ventricular enlargement. No pleural effusion. No pneumothorax. IMPRESSION: Pulmonary arterial hypertension with enlarged central pulmonary arteries and right ventricular enlargement.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Spontaneous pneumomediastinum with air outlining the mediastinal structures, including the thymus (spinnaker sail sign), heart border, and great vessels. Subcutaneous emphysema in the neck. No pneumothorax. No pleural effusion. IMPRESSION: Pneumomediastinum with subcutaneous emphysema. Likely due to airway rupture or Valsalva.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Extensive subcutaneous emphysema in the chest wall extending to the neck. No pneumothorax. No pneumomediastinum. Multiple chest tubes in place. Lungs are partially aerated. IMPRESSION: Subcutaneous emphysema, likely from chest tube or traumatic air leak.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Bochdalek hernia defect on the left with herniated abdominal contents (stomach and bowel) visible in the left hemithorax. Mediastinal shift to the right. No pneumothorax. No pleural effusion. IMPRESSION: Bochdalek hernia containing abdominal viscera. Congenital or acquired diaphragmatic hernia.",
        "labels": "Hernia",
    },
    {
        "diagnosis": "FINDINGS: Morgagni hernia in the right anterior cardiophrenic angle with omental fat herniating through the foramen of Morgagni. No bowel obstruction. No pleural effusion. No pneumothorax. IMPRESSION: Morgagni hernia containing omental fat. Usually asymptomatic and benign.",
        "labels": "Hernia",
    },
    {
        "diagnosis": "FINDINGS: Eventration of the right hemidiaphragm with marked elevation. The underlying lung shows compressive atelectasis. No pneumothorax. No pleural effusion. Heart shifted to the left. IMPRESSION: Right hemidiaphragm eventration with adjacent atelectasis. May mimic diaphragmatic rupture.",
        "labels": "Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Diaphragmatic rupture on the left with stomach herniated into the left hemithorax. The nasogastric tube is coiled above the diaphragm. Mediastinal shift to the right. No pneumothorax. Left pleural effusion. IMPRESSION: Left diaphragmatic rupture with gastric herniation. Emergency surgical repair needed.",
        "labels": "Effusion|Hernia",
    },
]

# Add more rows with varied symptoms cycling through existing images
def generate_extended_rows():
    rows = []
    img_idx = 0
    num_images = len(IMAGES)

    symptom_idx = 0
    for diag in DIAGNOSIS_ENTRIES:
        img_path = IMAGES[img_idx % num_images]
        img_idx += 1

        symptom = SYMPTOM_TEMPLATES[symptom_idx % len(SYMPTOM_TEMPLATES)]
        symptom_idx += 1

        labels = diag["labels"]
        diagnosis = diag["diagnosis"]

        rows.append([img_path, "augmented", symptom, diagnosis, labels])

    return rows


if __name__ == "__main__":
    initial_count = count_rows()
    print(f"Current dataset rows: {initial_count}")

    new_rows = generate_extended_rows()
    print(f"Adding {len(new_rows)} new rows...")

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    final_count = count_rows()
    print(f"New dataset rows: {final_count}")
    print(f"Added {final_count - initial_count} rows")
    print("Done!")
