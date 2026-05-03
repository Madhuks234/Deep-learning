"""
AyuLeafNet Configuration
========================
Central configuration for the hybrid CNN-based Ayurvedic leaf classification system.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")
RESULT_DIR  = os.path.join(BASE_DIR, "results")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
IMAGE_SIZE      = 224          # Input image size (224x224)
BATCH_SIZE      = 32
NUM_WORKERS     = 4
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
AUGMENT_TRAIN   = True

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
BACKBONE        = "mobilenet_v2"   # Options: mobilenet_v2 | efficientnet_b0 | resnet50
DROPOUT_RATE    = 0.4
NUM_CLASSES     = 10               # Number of Ayurvedic leaf classes

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
EPOCHS          = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
LR_STEP_SIZE    = 10
LR_GAMMA        = 0.5
EARLY_STOP_PAT  = 7               # Early stopping patience
PRETRAINED      = True            # Use ImageNet pretrained weights

# ─────────────────────────────────────────────
# AYURVEDIC LEAF CLASSES
# ─────────────────────────────────────────────
CLASSES = [
    "Tulsi",          # Ocimum tenuiflorum
    "Neem",           # Azadirachta indica
    "Aloe Vera",      # Aloe barbadensis
    "Turmeric",       # Curcuma longa
    "Ashwagandha",    # Withania somnifera
    "Brahmi",         # Bacopa monnieri
    "Amla",           # Phyllanthus emblica
    "Giloy",          # Tinospora cordifolia
    "Mint",           # Mentha
    "Curry Leaf",     # Murraya koenigii
]

# ─────────────────────────────────────────────
# MEDICINAL INTELLIGENCE DATABASE
# ─────────────────────────────────────────────
MEDICINAL_DB = {
    "Tulsi": {
        "scientific_name"  : "Ocimum tenuiflorum",
        "family"           : "Lamiaceae",
        "rasa"             : "Pungent, Bitter",
        "virya"            : "Hot",
        "vipaka"           : "Pungent",
        "dosha_effect"     : "Balances Vata & Kapha, slightly increases Pitta",
        "properties"       : ["Antibacterial", "Antiviral", "Adaptogenic", "Expectorant"],
        "uses"             : "Respiratory disorders, fever, stress, diabetes management",
        "parts_used"       : "Leaves, seeds, roots",
        "caution"          : "Avoid large doses during pregnancy",
        "color"            : "#2d6a4f",
    },
    "Neem": {
        "scientific_name"  : "Azadirachta indica",
        "family"           : "Meliaceae",
        "rasa"             : "Bitter, Astringent",
        "virya"            : "Cold",
        "vipaka"           : "Pungent",
        "dosha_effect"     : "Pacifies Pitta and Kapha",
        "properties"       : ["Antibacterial", "Antifungal", "Anti-inflammatory", "Antipyretic"],
        "uses"             : "Skin disorders, dental care, blood purification, malaria",
        "parts_used"       : "Leaves, bark, seeds, oil",
        "caution"          : "Avoid during pregnancy; toxic in large amounts",
        "color"            : "#40916c",
    },
    "Aloe Vera": {
        "scientific_name"  : "Aloe barbadensis miller",
        "family"           : "Asphodelaceae",
        "rasa"             : "Bitter, Sweet",
        "virya"            : "Cold",
        "vipaka"           : "Sweet",
        "dosha_effect"     : "Pacifies all three Doshas (Tridoshic)",
        "properties"       : ["Anti-inflammatory", "Wound healing", "Laxative", "Immunostimulant"],
        "uses"             : "Skin burns, digestive disorders, constipation, wound healing",
        "parts_used"       : "Gel, latex",
        "caution"          : "Avoid internal use during pregnancy",
        "color"            : "#52b788",
    },
    "Turmeric": {
        "scientific_name"  : "Curcuma longa",
        "family"           : "Zingiberaceae",
        "rasa"             : "Bitter, Pungent, Astringent",
        "virya"            : "Hot",
        "vipaka"           : "Pungent",
        "dosha_effect"     : "Balances all three Doshas",
        "properties"       : ["Anti-inflammatory", "Antioxidant", "Antibacterial", "Hepatoprotective"],
        "uses"             : "Arthritis, digestive disorders, liver health, wound healing",
        "parts_used"       : "Rhizome",
        "caution"          : "High doses may cause stomach upset",
        "color"            : "#d4a017",
    },
    "Ashwagandha": {
        "scientific_name"  : "Withania somnifera",
        "family"           : "Solanaceae",
        "rasa"             : "Bitter, Astringent, Sweet",
        "virya"            : "Hot",
        "vipaka"           : "Sweet",
        "dosha_effect"     : "Balances Vata and Kapha",
        "properties"       : ["Adaptogenic", "Immunomodulatory", "Anti-stress", "Anabolic"],
        "uses"             : "Stress, anxiety, fatigue, sexual vitality, neurological disorders",
        "parts_used"       : "Roots, leaves",
        "caution"          : "Avoid during pregnancy and hyperthyroidism",
        "color"            : "#b5838d",
    },
    "Brahmi": {
        "scientific_name"  : "Bacopa monnieri",
        "family"           : "Plantaginaceae",
        "rasa"             : "Bitter, Sweet, Astringent",
        "virya"            : "Cold",
        "vipaka"           : "Sweet",
        "dosha_effect"     : "Pacifies all three Doshas",
        "properties"       : ["Nootropic", "Anxiolytic", "Antioxidant", "Neuroprotective"],
        "uses"             : "Memory enhancement, anxiety, epilepsy, insomnia",
        "parts_used"       : "Whole plant",
        "caution"          : "May cause digestive upset if taken without food",
        "color"            : "#74c69d",
    },
    "Amla": {
        "scientific_name"  : "Phyllanthus emblica",
        "family"           : "Phyllanthaceae",
        "rasa"             : "All six tastes (predominant: Sour)",
        "virya"            : "Cold",
        "vipaka"           : "Sweet",
        "dosha_effect"     : "Balances all three Doshas",
        "properties"       : ["Antioxidant", "Immunomodulatory", "Rasayana", "Digestive"],
        "uses"             : "Immunity, hair health, digestion, anti-aging, diabetes",
        "parts_used"       : "Fruit",
        "caution"          : "May interact with blood thinners",
        "color"            : "#95d5b2",
    },
    "Giloy": {
        "scientific_name"  : "Tinospora cordifolia",
        "family"           : "Menispermaceae",
        "rasa"             : "Bitter, Astringent, Pungent",
        "virya"            : "Hot",
        "vipaka"           : "Sweet",
        "dosha_effect"     : "Balances all three Doshas",
        "properties"       : ["Immunomodulatory", "Antipyretic", "Anti-inflammatory", "Hepatoprotective"],
        "uses"             : "Fever, diabetes, liver disorders, dengue, immunity boost",
        "parts_used"       : "Stem, roots, leaves",
        "caution"          : "May lower blood sugar; monitor in diabetics",
        "color"            : "#1b4332",
    },
    "Mint": {
        "scientific_name"  : "Mentha spicata / Mentha piperita",
        "family"           : "Lamiaceae",
        "rasa"             : "Pungent, Sweet",
        "virya"            : "Cold",
        "vipaka"           : "Pungent",
        "dosha_effect"     : "Pacifies Pitta and Kapha",
        "properties"       : ["Carminative", "Analgesic", "Antimicrobial", "Antispasmodic"],
        "uses"             : "Digestive disorders, headaches, nausea, respiratory issues",
        "parts_used"       : "Leaves, oil",
        "caution"          : "Avoid in infants; may worsen acid reflux",
        "color"            : "#52b788",
    },
    "Curry Leaf": {
        "scientific_name"  : "Murraya koenigii",
        "family"           : "Rutaceae",
        "rasa"             : "Pungent, Bitter",
        "virya"            : "Hot",
        "vipaka"           : "Pungent",
        "dosha_effect"     : "Balances Kapha and Vata",
        "properties"       : ["Antidiabetic", "Antioxidant", "Anti-inflammatory", "Carminative"],
        "uses"             : "Diabetes, hair growth, digestive issues, weight management",
        "parts_used"       : "Leaves, bark, roots",
        "caution"          : "Generally safe in culinary amounts",
        "color"            : "#2d6a4f",
    },
}

# ─────────────────────────────────────────────
# NORMALIZATION (ImageNet statistics)
# ─────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
