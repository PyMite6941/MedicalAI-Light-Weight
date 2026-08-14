"""
Offline drug-drug interaction checker for MedicalAI - Light Weight.

Two tiers, both fully offline once set up:

1. A small built-in seed set (~35 classic, extremely well-established
   interactions taught in every pharmacology course — warfarin+NSAIDs,
   MAOIs+SSRIs, ACE inhibitors+potassium-sparing diuretics, etc.). Zero
   download required; this is what a fresh install has on day one.

2. An optional local SQLite database built from DDInter 2.0
   (http://ddinter.scbdd.com), a free, no-login academic database of
   ~160,000 deduplicated drug-drug interaction pairs across ~1,900 drugs,
   released under CC BY-NC-SA 4.0 (non-commercial, share-alike,
   attribution required). Run `python drug_interactions.py --download`
   once while online; after that, every lookup is a local, offline
   indexed SQLite query (~5-6 MB on disk).

This module never phones home on its own — downloading is an explicit,
one-time, user-initiated action, consistent with the app being
offline-first with online mode only for fetching material.
"""

import csv
import os
import re
import sqlite3
from itertools import combinations

DATA_DIR = "./data"
DB_PATH = os.path.join(DATA_DIR, "drug_interactions.db")

DDINTER_BASE_URL = "http://ddinter.scbdd.com/static/media/download"
DDINTER_ATC_CODES = ["A", "B", "D", "H", "L", "P", "R", "V"]

LEVEL_UNKNOWN, LEVEL_MINOR, LEVEL_MODERATE, LEVEL_MAJOR = 0, 1, 2, 3
LEVEL_NAMES = {LEVEL_UNKNOWN: "Unknown", LEVEL_MINOR: "Minor", LEVEL_MODERATE: "Moderate", LEVEL_MAJOR: "Major"}
LEVEL_FROM_STR = {"Unknown": LEVEL_UNKNOWN, "Minor": LEVEL_MINOR, "Moderate": LEVEL_MODERATE, "Major": LEVEL_MAJOR}

DISCLAIMER = (
    "This is a general interaction screen, not a substitute for a pharmacist or "
    "clinician review. Severity labels are approximate; always verify with a "
    "qualified professional before making treatment decisions."
)

ATTRIBUTION = (
    "Extended interaction data from DDInter 2.0 (http://ddinter.scbdd.com), "
    "CC BY-NC-SA 4.0 — non-commercial use only, shared here for offline lookup, "
    "not redistributed as part of this repository."
)

# ── Built-in seed set: works with zero download ────────────────────────
# Keyed by a frozenset of two normalized drug names. These are standard,
# high-confidence interactions found in any major pharmacology reference
# (e.g. Beers Criteria, standard drug interaction teaching) — not edge
# cases, so they're safe to hand-curate rather than requiring a download.

BUILT_IN_SEED = [
    ("warfarin", "aspirin", LEVEL_MAJOR, "Additive bleeding risk (anticoagulant + antiplatelet)."),
    ("warfarin", "ibuprofen", LEVEL_MAJOR, "NSAIDs increase bleeding risk and can displace warfarin from protein binding."),
    ("warfarin", "amiodarone", LEVEL_MAJOR, "Amiodarone inhibits warfarin metabolism, raising INR/bleeding risk."),
    ("warfarin", "rifampin", LEVEL_MAJOR, "Rifampin strongly induces warfarin metabolism, reducing anticoagulant effect."),
    ("warfarin", "fluconazole", LEVEL_MAJOR, "Fluconazole inhibits warfarin metabolism (CYP2C9), raising bleeding risk."),
    ("phenelzine", "sertraline", LEVEL_MAJOR, "MAOI + SSRI: risk of life-threatening serotonin syndrome."),
    ("tranylcypromine", "fluoxetine", LEVEL_MAJOR, "MAOI + SSRI: risk of life-threatening serotonin syndrome."),
    ("tramadol", "sertraline", LEVEL_MAJOR, "Both raise serotonin; combination risks serotonin syndrome."),
    ("linezolid", "fluoxetine", LEVEL_MAJOR, "Linezolid has MAOI-like activity; combining with SSRIs risks serotonin syndrome."),
    ("lisinopril", "spironolactone", LEVEL_MODERATE, "ACE inhibitor + potassium-sparing diuretic: risk of hyperkalemia."),
    ("lisinopril", "potassium chloride", LEVEL_MODERATE, "ACE inhibitors reduce potassium excretion; added potassium risks hyperkalemia."),
    ("lisinopril", "losartan", LEVEL_MODERATE, "Dual renin-angiotensin system blockade increases hyperkalemia/renal risk without added benefit."),
    ("simvastatin", "clarithromycin", LEVEL_MAJOR, "Macrolide inhibits statin metabolism (CYP3A4), raising rhabdomyolysis risk."),
    ("simvastatin", "erythromycin", LEVEL_MAJOR, "Macrolide inhibits statin metabolism (CYP3A4), raising rhabdomyolysis risk."),
    ("simvastatin", "itraconazole", LEVEL_MAJOR, "Azole antifungal strongly inhibits statin metabolism, raising rhabdomyolysis risk."),
    ("sildenafil", "nitroglycerin", LEVEL_MAJOR, "PDE5 inhibitor + nitrate: risk of severe, life-threatening hypotension."),
    ("sildenafil", "isosorbide dinitrate", LEVEL_MAJOR, "PDE5 inhibitor + nitrate: risk of severe, life-threatening hypotension."),
    ("theophylline", "ciprofloxacin", LEVEL_MAJOR, "Fluoroquinolone inhibits theophylline clearance, risking toxicity (seizures, arrhythmia)."),
    ("digoxin", "amiodarone", LEVEL_MAJOR, "Amiodarone raises digoxin levels; risk of digoxin toxicity."),
    ("digoxin", "verapamil", LEVEL_MODERATE, "Verapamil raises digoxin levels and adds AV-nodal blockade; risk of bradycardia/toxicity."),
    ("metoprolol", "verapamil", LEVEL_MAJOR, "Beta-blocker + non-dihydropyridine CCB: additive AV-node/heart-rate suppression, risk of bradycardia/heart block."),
    ("metoprolol", "diltiazem", LEVEL_MAJOR, "Beta-blocker + non-dihydropyridine CCB: additive AV-node/heart-rate suppression, risk of bradycardia/heart block."),
    ("metformin", "iodinated contrast media", LEVEL_MODERATE, "Contrast-induced renal impairment can cause metformin accumulation and lactic acidosis."),
    ("methotrexate", "ibuprofen", LEVEL_MAJOR, "NSAIDs reduce methotrexate renal clearance, raising toxicity risk (especially at high-dose methotrexate)."),
    ("methotrexate", "trimethoprim-sulfamethoxazole", LEVEL_MAJOR, "Both are folate antagonists; combination risks severe bone marrow suppression."),
    ("lithium", "ibuprofen", LEVEL_MAJOR, "NSAIDs reduce lithium renal clearance, raising lithium toxicity risk."),
    ("lithium", "hydrochlorothiazide", LEVEL_MAJOR, "Thiazide diuretics reduce lithium clearance, raising lithium toxicity risk."),
    ("lithium", "lisinopril", LEVEL_MODERATE, "ACE inhibitors can reduce lithium clearance, raising lithium levels."),
    ("glyburide", "fluconazole", LEVEL_MODERATE, "Fluconazole inhibits sulfonylurea metabolism, raising hypoglycemia risk."),
    ("clopidogrel", "omeprazole", LEVEL_MODERATE, "Omeprazole may reduce clopidogrel activation (CYP2C19), lowering antiplatelet effect."),
    ("citalopram", "tramadol", LEVEL_MODERATE, "Additive serotonergic effect; combined with QT-prolonging potential of citalopram, use with caution."),
    ("azithromycin", "amiodarone", LEVEL_MODERATE, "Both can prolong the QT interval; combination raises risk of dangerous arrhythmia."),
    ("ciprofloxacin", "amiodarone", LEVEL_MODERATE, "Both can prolong the QT interval; combination raises risk of dangerous arrhythmia."),
    ("prednisone", "ibuprofen", LEVEL_MODERATE, "Corticosteroid + NSAID: additive GI ulceration/bleeding risk."),
    ("warfarin", "acetaminophen", LEVEL_MODERATE, "Regular high-dose acetaminophen can potentiate warfarin's anticoagulant effect."),
    ("spironolactone", "potassium chloride", LEVEL_MAJOR, "Potassium-sparing diuretic + potassium supplement: high risk of hyperkalemia."),
]


def normalize_name(name):
    n = name.strip().lower()
    n = re.sub(r"\s+", " ", n)
    return n


def _seed_lookup():
    """Build the {frozenset({a, b}): (level, note)} map once."""
    return {
        frozenset((normalize_name(a), normalize_name(b))): (level, note)
        for a, b, level, note in BUILT_IN_SEED
    }


_SEED_LOOKUP = _seed_lookup()


# ── SQLite-backed full database (optional, built via --download) ───────

def db_available():
    return os.path.exists(DB_PATH)


def _connect():
    return sqlite3.connect(DB_PATH)


def _find_drug_id(conn, name):
    n = normalize_name(name)
    row = conn.execute("SELECT id FROM drugs WHERE name = ?", (n,)).fetchone()
    if row:
        return row[0]
    # Fall back to a prefix/substring match for slightly-off spellings
    # (e.g. "amoxicillin" vs "amoxicillin trihydrate").
    row = conn.execute(
        "SELECT id FROM drugs WHERE name LIKE ? ORDER BY length(name) ASC LIMIT 1",
        (f"%{n}%",),
    ).fetchone()
    return row[0] if row else None


def _db_check_interaction(conn, drug_a, drug_b):
    ida = _find_drug_id(conn, drug_a)
    idb = _find_drug_id(conn, drug_b)
    if ida is None or idb is None:
        return None
    lo, hi = (ida, idb) if ida <= idb else (idb, ida)
    row = conn.execute(
        "SELECT level FROM interactions WHERE drug_a = ? AND drug_b = ?", (lo, hi)
    ).fetchone()
    if row is None:
        return None
    return {"level": LEVEL_NAMES[row[0]], "note": None, "source": "DDInter 2.0"}


# ── Public lookup API ───────────────────────────────────────────────────

def check_interaction(drug_a, drug_b):
    """Check a single drug pair. Checks the downloaded database first (if
    present) since it's far more complete, then falls back to the
    built-in seed set. Returns None if nothing is known about the pair
    (this is NOT the same as "confirmed safe" — see DISCLAIMER)."""
    if db_available():
        try:
            conn = _connect()
            try:
                result = _db_check_interaction(conn, drug_a, drug_b)
                if result:
                    return result
            finally:
                conn.close()
        except sqlite3.Error:
            pass

    key = frozenset((normalize_name(drug_a), normalize_name(drug_b)))
    seed_hit = _SEED_LOOKUP.get(key)
    if seed_hit:
        level, note = seed_hit
        return {"level": LEVEL_NAMES[level], "note": note, "source": "built-in reference set"}

    return None


def check_medication_list(medications):
    """Check every pairwise combination in a list of medication names.

    Returns (interactions, unrecognized) where `interactions` is a list of
    dicts {drug_a, drug_b, level, note, source} for every pair with a
    known interaction, and `unrecognized` lists names not found in either
    the downloaded database or the built-in set (not necessarily unsafe —
    just nothing on file, e.g. a misspelling or a very new drug).
    """
    names = [m.strip() for m in medications if m and m.strip()]
    interactions = []
    known_names = set()

    conn = _connect() if db_available() else None
    try:
        for a, b in combinations(names, 2):
            result = None
            if conn is not None:
                try:
                    result = _db_check_interaction(conn, a, b)
                except sqlite3.Error:
                    result = None
            if result is None:
                key = frozenset((normalize_name(a), normalize_name(b)))
                seed_hit = _SEED_LOOKUP.get(key)
                if seed_hit:
                    level, note = seed_hit
                    result = {"level": LEVEL_NAMES[level], "note": note, "source": "built-in reference set"}
            if result:
                interactions.append({"drug_a": a, "drug_b": b, **result})
                known_names.add(normalize_name(a))
                known_names.add(normalize_name(b))
    finally:
        if conn is not None:
            conn.close()

    # A name counts as "recognized" if it appears in the seed set at all,
    # even without a hit against these specific companions.
    seed_names = {n for pair in _SEED_LOOKUP for n in pair}
    unrecognized = [
        m for m in names
        if normalize_name(m) not in known_names and normalize_name(m) not in seed_names
        and not (conn_names_has(m))
    ]
    return interactions, unrecognized


def conn_names_has(name):
    """Best-effort check of whether a name exists in the downloaded DB at
    all (even with zero interactions on file), so it isn't wrongly
    flagged as 'unrecognized'."""
    if not db_available():
        return False
    try:
        conn = _connect()
        try:
            return _find_drug_id(conn, name) is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def format_interaction(result):
    level = result["level"]
    line = f"{result['drug_a']} + {result['drug_b']}: {level}"
    if result.get("note"):
        line += f" — {result['note']}"
    line += f" (source: {result['source']})"
    return line


# ── Download + build the full offline database ─────────────────────────

def download_and_build_db(progress=True):
    """Fetch DDInter 2.0's CSVs and build a compact, indexed local SQLite
    database. One-time, explicit, online action — everything after this
    runs fully offline. Safe to re-run (rebuilds from scratch)."""
    try:
        from rich.console import Console
        console = Console()
        p = console.print
    except ImportError:
        p = print

    try:
        import requests
    except ImportError:
        p("[red]'requests' is required to download the interaction database.[/red]" if progress else
          "'requests' is required. Install with: pip install requests")
        p("Install with: pip install requests")
        return False

    os.makedirs(DATA_DIR, exist_ok=True)
    p(f"[cyan]Downloading DDInter 2.0 drug-interaction data ({len(DDINTER_ATC_CODES)} files, ~13 MB)...[/cyan]")
    p(f"[dim]{ATTRIBUTION}[/dim]")

    rows = []
    for code in DDINTER_ATC_CODES:
        url = f"{DDINTER_BASE_URL}/ddinter_downloads_code_{code}.csv"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            p(f"[yellow]  Skipped code {code}: {e}[/yellow]")
            continue

        text = resp.content.decode("utf-8", errors="replace")
        reader = csv.DictReader(text.splitlines())
        count = 0
        for row in reader:
            a = normalize_name(row.get("Drug_A", ""))
            b = normalize_name(row.get("Drug_B", ""))
            level = row.get("Level", "Unknown").strip()
            if not a or not b or a == b:
                continue
            rows.append((a, b, LEVEL_FROM_STR.get(level, LEVEL_UNKNOWN)))
            count += 1
        p(f"  [green]{code}: {count} rows[/green]")

    if not rows:
        p("[red]No data downloaded. Check your network connection and try again.[/red]")
        return False

    p("[cyan]Building indexed local database...[/cyan]")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("CREATE TABLE drugs (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL)")
    conn.execute("CREATE TABLE interactions (drug_a INTEGER NOT NULL, drug_b INTEGER NOT NULL, level INTEGER NOT NULL)")

    drug_ids = {}
    dedup_pairs = {}
    for a, b, level in rows:
        ida = drug_ids.setdefault(a, len(drug_ids) + 1)
        idb = drug_ids.setdefault(b, len(drug_ids) + 1)
        lo, hi = (ida, idb) if ida <= idb else (idb, ida)
        # Keep the highest-severity level if the same pair appears in
        # more than one ATC-code file.
        existing = dedup_pairs.get((lo, hi))
        if existing is None or level > existing:
            dedup_pairs[(lo, hi)] = level

    conn.executemany("INSERT INTO drugs (id, name) VALUES (?, ?)", [(v, k) for k, v in drug_ids.items()])
    conn.executemany(
        "INSERT INTO interactions (drug_a, drug_b, level) VALUES (?, ?, ?)",
        [(a, b, level) for (a, b), level in dedup_pairs.items()],
    )
    conn.execute("CREATE INDEX idx_interactions_a ON interactions(drug_a)")
    conn.execute("CREATE INDEX idx_interactions_b ON interactions(drug_b)")
    conn.execute("CREATE UNIQUE INDEX idx_drugs_name ON drugs(name)")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    size_mb = os.path.getsize(DB_PATH) / 1024 / 1024
    p(f"[green]Done. {len(drug_ids)} drugs, {len(dedup_pairs)} interaction pairs, {size_mb:.1f} MB at {DB_PATH}[/green]")
    p("[dim]All future lookups run fully offline against this file.[/dim]")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline drug-drug interaction database")
    parser.add_argument("--download", action="store_true", help="Download + build the full local database (one-time, needs network)")
    parser.add_argument("--check", nargs="+", metavar="DRUG", help="Check all pairwise interactions among the given drug names")
    args = parser.parse_args()

    if args.download:
        download_and_build_db()
    elif args.check:
        interactions, unrecognized = check_medication_list(args.check)
        if interactions:
            for r in interactions:
                print(format_interaction(r))
        else:
            print("No known interactions found among the given medications.")
        if unrecognized:
            print(f"Not found in the interaction database: {', '.join(unrecognized)}")
        print()
        print(DISCLAIMER)
        if not db_available():
            print("(Using the small built-in reference set only — run --download for full coverage.)")
    else:
        parser.print_help()
