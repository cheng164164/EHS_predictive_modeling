from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import config as cfg
import numpy as np
import pandas as pd

from utils import clean_text_value, ensure_dir, load_table, normalize_embeddings, save_csv, save_json

# ---------------------------------------------------------------------------
# Step 02: safety tag extraction
# ---------------------------------------------------------------------------
# This script uses three layers:
#   Layer 1: deterministic regex rules for high-precision known safety terms.
#   Layer 2: optional embedding fallback for rows/categories with no rule match.
#   Layer 3: unmatched-text discovery files to show where the dictionary is blind.
#
# LLM extraction is still supported for optional experiments, but it is not the
# default. The default backend remains local rules + embedding fallback.
# ---------------------------------------------------------------------------


def log_progress(message: str) -> None:
    """Print lightweight phase-level progress updates for long Step 02 runs."""
    print(f"[Step 02] {message}", flush=True)


def normalize_for_rules(value: object) -> str:
    """Normalize text for robust regex matching.

    This strips accents so rules can match both Spanish words with accents and
    unaccented variants, e.g. inspeccion/inspeccion, electrico/electrico.
    """
    text = clean_text_value(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("/", " ").replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Rules are intentionally broad but still use word boundaries where practical.
# All patterns should be written against normalized lowercase text.
TAG_RULES: Dict[str, List[Tuple[str, str]]] = {
    "hazard_type": [
        (
            "mobile_equipment_pedestrian",
            r"\b(fork\s?lift|forklift|lift\s?truck|powered industrial truck|\bpit\b|pallet jack|walkie|tugger|yard dog|hostler|hilo|hi lo|montacargas|autoelevador|carretilla elevadora|vehiculo industrial|peaton|peatonal|pedestrian|traffic|trafico|walkway|crosswalk|loading dock|dock door|blind spot|backing|reverse|reversing|retroceso|struck by.*(forklift|truck|vehicle|montacargas))\b",
        ),
        (
            "vehicle_transport",
            r"\b(vehicle|truck|trailer|semi|tractor|yard truck|car|van|bus|driver|driving|parking lot|roadway|motor vehicle|fleet|transport|transporte|camion|remolque|vehiculo|conduccion|chofer)\b",
        ),
        (
            "material_handling_lifting",
            r"\b(manual handling|material handling|lift|lifting|carry|carrying|push|pull|pushing|pulling|load handling|pallet|stacking|unstacking|warehouse|almacen|carga|descarga|levantar|empujar|jalar)\b",
        ),
        (
            "slip_trip_fall",
            r"\b(slip|slipped|slippery|trip|tripped|fall|fell|falling|same level|walking surface|wet floor|icy|ice|snow|uneven|hole|cord|mat|stair|stairs|step|sidewalk|walkway|resbalon|resbalar|tropiezo|tropezar|caida|piso mojado)\b",
        ),
        (
            "fall_from_height",
            r"\b(fall from height|working at height|height|roof|mezzanine|platform|elevated|leading edge|open edge|floor opening|harness|lanyard|tie off|tieoff|guardrail|fall protection|fall arrest|altura|arnes)\b",
        ),
        (
            "ladder_scaffold_platform",
            r"\b(ladder|step ladder|extension ladder|scaffold|scaffolding|scissor lift|aerial lift|boom lift|manlift|elevating work platform|plataforma|andamio|escalera)\b",
        ),
        (
            "machine_guarding_pinch_point",
            r"\b(machine|machinery|conveyor|belt|roller|rotating|rotation|gear|sprocket|chain|press|saw|blade|cutter|pinch|nip point|caught|caught between|caught in|caught on|crush point|unguarded|guard|guarding|interlock|amputation|maquina|guardas|punto de atrapamiento|atrapado)\b",
        ),
        (
            "electrical_loto",
            r"\b(electrical|electric|energized|de energized|deenergized|lockout|tagout|lock out|tag out|loto|breaker|panel|voltage|volt|arc flash|shock|electrocution|disconnect|isolation|zero energy|live wire|electrico|energia electrica|bloqueo|etiquetado)\b",
        ),
        (
            "chemical_exposure_spill",
            r"\b(chemical|chemicals|spill|spilled|leak|leaking|acid|caustic|corrosive|solvent|fume|vapou?r|gas|dust|powder|sds|msds|exposure|splash|odor|odour|inhalation|irritation|container|drum|secondary containment|derrame|quimico|acido|fuga|exposicion|salpicadura)\b",
        ),
        (
            "fire_hot_work_explosion",
            r"\b(fire|smoke|spark|sparks|weld|welding|hot work|burn|flame|ignition|combustible|flammable|extinguisher|explosion|blast|bateria|battery fire|incendio|humo|soldadura|quemadura|explosion)\b",
        ),
        (
            "ergonomic_strain",
            r"\b(ergonomic|ergonomics|strain|sprain|overexertion|awkward posture|repetitive|repetition|twist|twisting|bend|bending|back pain|shoulder pain|wrist|hand pain|musculoskeletal|musc?ulo|ergonomia|sobreesfuerzo|dolor de espalda)\b",
        ),
        (
            "dropped_object_falling_material",
            r"\b(dropped|falling object|object fell|tool fell|material fell|overhead|falling material|unsecured load|load fell|racking|shelf|shelving|stack collapsed|dropped object|objeto caido|material caido)\b",
        ),
        (
            "housekeeping_obstruction",
            r"\b(housekeeping|clutter|debris|blocked|blockage|obstructed|obstruction|aisle|egress|exit blocked|storage|trash|garbage|cleanliness|5s|pallets in aisle|cables on floor|orden y limpieza|pasillo bloqueado|obstruido|basura)\b",
        ),
        (
            "ppe",
            r"\b(ppe|personal protective equipment|glove|gloves|cut glove|eye protection|safety glasses|goggles|face shield|hard hat|helmet|hearing protection|ear plug|earplug|respirator|mask|safety shoes|steel toe|chaleco|guantes|lentes de seguridad|casco|proteccion personal)\b",
        ),
        (
            "pressure_release",
            r"\b(pressure|pressurized|hydraulic|pneumatic|compressed air|air hose|hose|line rupture|burst|stored energy|steam line|cylinder|gas cylinder|presion|hidraulico|neumatico|aire comprimido|manguera)\b",
        ),
        (
            "confined_space",
            r"\b(confined space|permit required confined space|tank entry|vessel entry|manhole|oxygen deficient|atmosphere test|espacio confinado)\b",
        ),
        (
            "crane_hoist_rigging",
            r"\b(crane|hoist|rigging|sling|chain fall|overhead crane|lift plan|suspended load|gantry|bridge crane|grua|izaje|eslinga|polipasto|carga suspendida)\b",
        ),
        (
            "hand_power_tools",
            r"\b(hand tool|power tool|drill|grinder|sander|saw|knife|box cutter|utility knife|hammer|wrench|torque|herramienta|taladro|esmeril|cuchillo|cutters?)\b",
        ),
        (
            "sharp_object_cut",
            r"\b(cut|cuts|cutting|laceration|puncture|needle|sharp|blade|razor|glass|metal edge|burr|corte|cortadura|laceracion|punzon)\b",
        ),
        (
            "environmental_release",
            r"\b(environmental|release|spill to drain|storm drain|waste|hazardous waste|oil spill|fuel spill|contamination|contaminated|emission|ambiental|residuo|derrame de aceite)\b",
        ),
        (
            "biological_exposure",
            r"\b(blood|bodily fluid|biohazard|biological|needle stick|insect|bee|wasp|animal|mold|mould|bacteria|virus|biologico|sangre)\b",
        ),
        (
            "noise_hearing",
            r"\b(noise|loud|hearing|decibel|dba|earplug|audiogram|ruido|audicion)\b",
        ),
        (
            "thermal_stress",
            r"\b(heat stress|heat illness|heat exhaustion|cold stress|hypothermia|frostbite|hot environment|temperatura|calor|frio)\b",
        ),
        (
            "security_violence",
            r"\b(violence|threat|assault|security|trespass|intruder|aggressive|fight|weapon|amenaza|asalto|seguridad)\b",
        ),
        (
            "struck_by_caught_between",
            r"\b(struck by|hit by|impact|collision|caught between|caught in|pinched|crushed|golpeado|atrapado|aplastado)\b",
        ),
    ],
    "control_failure": [
        (
            "traffic_separation",
            r"\b(traffic|pedestrian|walkway|crosswalk|barrier|segregation|separation|blind spot|right of way|speed limit|one way|forklift lane|traffic plan|traffic control|peatonal|separacion|barrera|pasillo peatonal)\b",
        ),
        (
            "missing_or_poor_guarding",
            r"\b(missing guard|unguarded|guard removed|guarding|interlock|barrier guard|guard damaged|guard open|bypass|defeated|proteccion removida|sin guarda|guarda danada)\b",
        ),
        (
            "poor_housekeeping",
            r"\b(housekeeping|clutter|debris|blocked|obstructed|obstruction|spill|wet floor|aisle|storage|trash|5s|clear path|exit blocked|limpieza|pasillo bloqueado|obstruido|derrame)\b",
        ),
        (
            "loto_gap",
            r"\b(lockout|tagout|lock out|tag out|loto|energized|deenergize|de energized|isolation|zero energy|disconnect|try out|bloqueo|etiquetado|energia cero)\b",
        ),
        (
            "ppe_gap",
            r"\b(no ppe|not wearing|without gloves|no gloves|without eye|no eye protection|ppe|glove|goggles|respirator|hard hat|hearing protection|guantes|lentes|casco|proteccion personal)\b",
        ),
        (
            "procedure_training_gap",
            r"\b(procedure|sop|standard work|work instruction|training|trained|instruction|not followed|awareness|competency|unauthorized|did not follow|process not followed|procedimiento|capacitacion|entrenamiento|instruccion)\b",
        ),
        (
            "maintenance_repair_gap",
            r"\b(maintenance|repair|damaged|broken|worn|wear|defect|defective|failed|failure|malfunction|leaking|out of service|preventive maintenance|pm overdue|mantenimiento|reparacion|danado|roto|defecto)\b",
        ),
        (
            "signage_visibility_gap",
            r"\b(sign|signage|label|labeling|marking|floor marking|visibility|blind|lighting|line of sight|faded|not visible|no sign|senal|senalizacion|etiqueta|iluminacion|visible)\b",
        ),
        (
            "inspection_gap",
            r"\b(inspection|inspect|audit|checklist|pre use|preuse|pre shift|walkthrough|not inspected|missed inspection|inspeccion|auditoria|lista de verificacion)\b",
        ),
        (
            "supervision_planning_gap",
            r"\b(supervisor|supervision|planning|pre job|prejob|job plan|job hazard analysis|jha|jsa|communication|handoff|coordination|pre task|pretask|supervision|planificacion|comunicacion)\b",
        ),
        (
            "chemical_storage_handling_gap",
            r"\b(storage|container|label|sds|secondary containment|chemical handling|incompatible|unlabeled container|drum|closed container|almacenamiento|contenedor|etiqueta|contencion secundaria)\b",
        ),
        (
            "fall_protection_gap",
            r"\b(fall protection|harness|lanyard|tie off|tieoff|guardrail|ladder inspection|scaffold|anchor point|100 percent tie off|proteccion contra caidas|arnes|barandal)\b",
        ),
        (
            "permit_authorization_gap",
            r"\b(permit|authorization|hot work permit|confined space permit|work permit|permit required|approved|approval|permiso|autorizacion)\b",
        ),
        (
            "emergency_response_gap",
            r"\b(emergency|evacuation|alarm|first aid|eyewash|shower|spill kit|fire extinguisher|emergency exit|blocked exit|respuesta de emergencia|extintor|botiquin|lavaojos)\b",
        ),
        (
            "containment_spill_control_gap",
            r"\b(containment|secondary containment|spill kit|drip pan|absorbent|berm|drain cover|storm drain|contencion|kit de derrames|absorbente)\b",
        ),
        (
            "ergonomic_design_gap",
            r"\b(ergonomic|awkward|repetitive|workstation|height adjustment|manual handling|lift assist|tooling|ergonomia|estacion de trabajo)\b",
        ),
        (
            "tool_equipment_selection_gap",
            r"\b(wrong tool|improper tool|damaged tool|tool selection|equipment selection|not rated|capacity|load rating|herramienta incorrecta|capacidad)\b",
        ),
        (
            "load_securement_gap",
            r"\b(unsecured load|load securement|strap|tie down|chock|blocked wheels|stacking|racking|shelf|load shifted|carga no asegurada|estiba)\b",
        ),
        (
            "barricade_exclusion_zone_gap",
            r"\b(barricade|exclusion zone|restricted area|red tape|caution tape|cones|line of fire|drop zone|barricada|zona de exclusion|cinta de precaucion)\b",
        ),
        (
            "ventilation_exposure_control_gap",
            r"\b(ventilation|local exhaust|fume hood|respiratory protection|air monitoring|exposure control|ventilacion|extraccion|monitoreo de aire)\b",
        ),
    ],
    "energy_source": [
        ("vehicle_motion", r"\b(fork\s?lift|truck|vehicle|backing|reverse|reversing|moving equipment|pallet jack|montacargas|vehiculo|camion|retroceso)\b"),
        ("gravity", r"\b(fall|dropped|falling|height|overhead|ladder|scaffold|suspended load|caida|altura|carga suspendida)\b"),
        ("mechanical", r"\b(machine|conveyor|rotating|pinch|caught|gear|press|saw|blade|roller|mechanical|maquina|mecanico|atrapado)\b"),
        ("electrical", r"\b(electric|electrical|energized|voltage|breaker|arc flash|shock|electrocution|electrico|voltaje)\b"),
        ("chemical", r"\b(chemical|acid|caustic|solvent|fume|vapor|spill|exposure|gas|quimico|acido|derrame)\b"),
        ("thermal", r"\b(hot|heat|burn|weld|fire|flame|steam|cold|calor|quemadura|incendio|vapor|frio)\b"),
        ("pressure", r"\b(pressure|hydraulic|pneumatic|compressed|hose|line rupture|steam line|cylinder|presion|hidraulico|neumatico)\b"),
        ("human_motion", r"\b(lift|lifting|push|pull|twist|repetitive|strain|sprain|manual handling|levantar|empujar|jalar)\b"),
        ("biological", r"\b(blood|biohazard|biological|needle|insect|animal|mold|virus|bacteria|biologico|sangre)\b"),
        ("noise", r"\b(noise|loud|hearing|decibel|dba|ruido|audicion)\b"),
    ],
    "action_type": [
        ("repair", r"\b(repair|fix|correct|restore|service|troubleshoot|reparar|corregir|arreglar)\b"),
        ("replace", r"\b(replace|replacement|new part|swap|cambiar|reemplazar|sustituir)\b"),
        ("inspect", r"\b(inspect|inspection|audit|verify|check|review|evaluate|assess|inspeccionar|verificar|revisar|auditar)\b"),
        ("train", r"\b(train|training|coach|communicate|brief|toolbox|awareness|retrain|capacitar|entrenar|comunicar)\b"),
        ("clean", r"\b(clean|remove debris|housekeeping|sweep|clear|organize|5s|limpiar|ordenar|retirar)\b"),
        ("install_engineering_control", r"\b(install|barrier|guard|interlock|signage|marking|rail|guardrail|ventilation|light curtain|instalar|barrera|guarda|senalizacion)\b"),
        ("update_procedure", r"\b(update procedure|revise sop|procedure|standard work|work instruction|jsa|jha|actualizar procedimiento|revisar procedimiento)\b"),
        ("label_or_sign", r"\b(label|sign|signage|mark|floor marking|paint line|etiquetar|senal|senalizar|marcar)\b"),
        ("barricade_or_isolate", r"\b(barricade|isolate|lockout|block access|restrict access|cone off|caution tape|barricada|aislar|bloquear acceso)\b"),
        ("contain_spill", r"\b(contain|spill kit|absorbent|clean spill|secondary containment|contener|kit de derrames|absorbente)\b"),
        ("review_investigate", r"\b(investigate|root cause|rca|review incident|lessons learned|investigar|causa raiz)\b"),
        ("assign_close_followup", r"\b(assign|follow up|complete|close|closure|due date|owner|action item|asignar|seguimiento|completar|cerrar)\b"),
        ("ppe_provide_enforce", r"\b(provide ppe|wear ppe|enforce ppe|gloves|goggles|respirator|safety glasses|proveer epp|usar epp)\b"),
        ("maintenance_pm", r"\b(preventive maintenance|pm|maintain|lubricate|calibrate|mantenimiento preventivo|calibrar|lubricar)\b"),
    ],
}

# Definitions for semantic fallback. These are short domain descriptions, not rules.
TAG_DEFINITIONS: Dict[str, List[Tuple[str, str]]] = {
    "hazard_type": [
        ("mobile_equipment_pedestrian", "Forklifts, powered industrial trucks, pallet jacks, vehicles, pedestrians, loading docks, backing operations, traffic separation, struck-by mobile equipment."),
        ("vehicle_transport", "Road vehicles, trucks, trailers, driving, parking lots, yard trucks, transportation and fleet safety."),
        ("material_handling_lifting", "Manual material handling, lifting, carrying, pushing, pulling, pallets, loading and unloading materials."),
        ("slip_trip_fall", "Slips, trips, same-level falls, wet floors, stairs, uneven walking surfaces, cords, mats, ice, obstructions."),
        ("fall_from_height", "Work at height, roofs, platforms, mezzanines, open edges, fall protection, harnesses, guardrails."),
        ("ladder_scaffold_platform", "Ladders, scaffolds, aerial lifts, scissor lifts, boom lifts and elevated work platforms."),
        ("machine_guarding_pinch_point", "Machines, conveyors, rotating equipment, guards, interlocks, pinch points, caught-in and caught-between hazards."),
        ("electrical_loto", "Electrical hazards, energized equipment, lockout tagout, breakers, panels, voltage, arc flash and shock exposure."),
        ("chemical_exposure_spill", "Chemical spills, leaks, fumes, vapors, acids, caustics, solvents, SDS, exposure and splashes."),
        ("fire_hot_work_explosion", "Fire, smoke, sparks, welding, hot work, burns, ignition sources, flammables, explosions."),
        ("ergonomic_strain", "Ergonomics, strain, sprain, overexertion, awkward posture, repetitive motion and musculoskeletal issues."),
        ("dropped_object_falling_material", "Dropped objects, falling tools, overhead materials, unsecured loads, racking and shelving collapse."),
        ("housekeeping_obstruction", "Housekeeping, clutter, blocked aisles, debris, storage issues, trash, 5S and obstructed exits."),
        ("ppe", "Personal protective equipment such as gloves, safety glasses, goggles, hard hats, respirators, hearing protection and safety shoes."),
        ("pressure_release", "Hydraulic, pneumatic, compressed air, pressurized hoses, cylinders, steam lines and stored pressure releases."),
        ("confined_space", "Confined spaces, permit-required entries, tanks, vessels, manholes, oxygen deficiency and atmospheric testing."),
        ("crane_hoist_rigging", "Cranes, hoists, rigging, slings, suspended loads, lift plans, overhead cranes and gantries."),
        ("hand_power_tools", "Hand tools and power tools including drills, grinders, saws, knives, box cutters, hammers and wrenches."),
        ("sharp_object_cut", "Cuts, lacerations, punctures, sharp edges, blades, razors, glass and metal burrs."),
        ("environmental_release", "Environmental releases, oil or fuel spills, waste, storm drains, contamination and emissions."),
        ("biological_exposure", "Blood, bodily fluids, biological exposure, needle sticks, insects, animals, mold, bacteria and viruses."),
        ("noise_hearing", "Noise, loud equipment, hearing exposure, decibels, earplugs and hearing protection."),
        ("thermal_stress", "Heat stress, cold stress, burns from hot surfaces or steam, frostbite and temperature exposure."),
        ("security_violence", "Security incidents, threats, assault, workplace violence, trespassers, intruders and aggressive behavior."),
        ("struck_by_caught_between", "Struck-by, hit-by, impact, collision, caught-between, caught-in, pinched and crushed exposures."),
    ],
    "control_failure": [
        ("traffic_separation", "Missing or weak separation between pedestrians and mobile equipment, traffic routes, barriers, crosswalks, signs and blind spots."),
        ("missing_or_poor_guarding", "Missing, removed, damaged or bypassed machine guards, interlocks and barrier guards."),
        ("poor_housekeeping", "Poor housekeeping, clutter, debris, blocked aisles, spills, obstructions, storage problems and blocked exits."),
        ("loto_gap", "Lockout tagout or energy isolation gap, energized equipment, zero energy not verified, disconnect or tryout missing."),
        ("ppe_gap", "PPE not used, wrong PPE, missing gloves, eye protection, respirator, hard hat, hearing protection or safety shoes."),
        ("procedure_training_gap", "Procedure not followed, missing SOP, training gap, competency issue, awareness issue or unauthorized work."),
        ("maintenance_repair_gap", "Damaged, broken, worn, defective or malfunctioning equipment requiring repair, maintenance or preventive maintenance."),
        ("signage_visibility_gap", "Missing labels, signage, floor markings, visibility controls, lighting, line of sight or faded markings."),
        ("inspection_gap", "Inspection, audit, checklist, pre-use check or walkthrough gap; condition was not detected or verified."),
        ("supervision_planning_gap", "Supervision, planning, pre-job briefing, communication, coordination, JHA/JSA or job plan gap."),
        ("chemical_storage_handling_gap", "Chemical storage, labeling, SDS, secondary containment, incompatible storage, drum or container handling gap."),
        ("fall_protection_gap", "Fall protection gap including harness, lanyard, tie-off, guardrail, anchor, ladder inspection or scaffold controls."),
        ("permit_authorization_gap", "Permit or authorization gap for hot work, confined space, work permit or approval process."),
        ("emergency_response_gap", "Emergency response gap including eyewash, shower, spill kit, extinguisher, alarm, first aid or evacuation route."),
        ("containment_spill_control_gap", "Containment, spill control, secondary containment, absorbents, drip pans, drain covers or storm drain controls."),
        ("ergonomic_design_gap", "Workstation, tool, lift assist, reach, height adjustment or ergonomic design gap."),
        ("tool_equipment_selection_gap", "Wrong tool, improper tool, damaged tool, equipment capacity, load rating or selection issue."),
        ("load_securement_gap", "Unsecured load, straps, tie-downs, wheel chocks, stacking, racking, shelving or load shift controls."),
        ("barricade_exclusion_zone_gap", "Barricade, exclusion zone, caution tape, cones, red tape, restricted area, line-of-fire or drop-zone control gap."),
        ("ventilation_exposure_control_gap", "Ventilation, local exhaust, fume hood, respiratory protection, air monitoring or exposure control gap."),
    ],
    "energy_source": [
        ("vehicle_motion", "Motion of forklifts, trucks, vehicles, pallet jacks and mobile equipment."),
        ("gravity", "Gravity hazards including falls, dropped objects, overhead loads, suspended loads and elevated work."),
        ("mechanical", "Mechanical energy from machines, conveyors, rotating parts, gears, presses, saws and blades."),
        ("electrical", "Electrical energy, energized panels, voltage, arc flash, breakers, shock and electrocution."),
        ("chemical", "Chemical energy or exposure from spills, fumes, vapors, acids, caustics and solvents."),
        ("thermal", "Thermal energy from heat, hot surfaces, burns, welding, fire, steam or cold exposure."),
        ("pressure", "Pressure energy from hydraulic, pneumatic, compressed air, hoses, cylinders and steam lines."),
        ("human_motion", "Human motion, manual handling, lifting, pushing, pulling, twisting, repetitive movement and exertion."),
        ("biological", "Biological agents, blood, bodily fluids, needle sticks, insects, animals, mold, bacteria and viruses."),
        ("noise", "Noise and acoustic energy, loud equipment, hearing exposure and decibels."),
    ],
    "action_type": [
        ("repair", "Repair, fix, correct, restore, troubleshoot or service damaged equipment or conditions."),
        ("replace", "Replace, swap or install a new part, component or item."),
        ("inspect", "Inspect, audit, verify, check, review, assess or evaluate a condition."),
        ("train", "Train, coach, communicate, brief, toolbox talk, awareness or retraining action."),
        ("clean", "Clean, sweep, remove debris, clear obstruction, housekeeping or 5S action."),
        ("install_engineering_control", "Install barrier, guard, interlock, signage, floor marking, guardrail, ventilation or engineering control."),
        ("update_procedure", "Update procedure, revise SOP, standard work, work instruction, JSA or JHA."),
        ("label_or_sign", "Add label, sign, floor marking, paint line or visual control."),
        ("barricade_or_isolate", "Barricade, isolate, lockout, cone off, block access or restrict access."),
        ("contain_spill", "Contain a spill, use absorbent, spill kit, secondary containment or drain protection."),
        ("review_investigate", "Investigate, root cause analysis, incident review or lessons learned."),
        ("assign_close_followup", "Assign owner, follow up, complete, close, track due date or action item."),
        ("ppe_provide_enforce", "Provide, wear, enforce or replace PPE such as gloves, goggles, respirators or safety glasses."),
        ("maintenance_pm", "Preventive maintenance, lubrication, calibration, service or maintenance program action."),
    ],
}

CONSEQUENCE_RULES = [
    ("fatality_potential", 4, r"\b(fatal|fatality|death|kill|killed|crush|crushed|amputation|electrocution|explosion|confined space|asphyxiation|struck by forklift|struck by truck|fall from height|arc flash|muerte|fatalidad|aplastado|amputacion|electrocucion)\b"),
    ("serious_potential", 3, r"\b(fracture|broken bone|hospital|inpatient|lost time|restricted|caught between|forklift.*pedestrian|pedestrian.*forklift|chemical exposure|high potential|serious injury|major injury|head injury|unconscious|ambulance|fractura|hospital|lesion grave)\b"),
    ("moderate_potential", 2, r"\b(cut|laceration|burn|sprain|strain|bruise|contusion|first aid|medical treatment|near miss|near-miss|puncture|minor injury|corte|laceracion|quemadura|esguince|primeros auxilios)\b"),
    ("low_potential", 1, r"\b(minor|paper cut|discomfort|soreness|irritation|scratch|near hit|molestia|rasguno|menor)\b"),
]

COMPILED_TAG_RULES: Dict[str, List[Tuple[str, re.Pattern]]] = {
    category: [(tag, re.compile(pattern, flags=re.IGNORECASE)) for tag, pattern in rules]
    for category, rules in TAG_RULES.items()
}
COMPILED_CONSEQUENCE_RULES = [
    (label, score, re.compile(pattern, flags=re.IGNORECASE))
    for label, score, pattern in CONSEQUENCE_RULES
]


def find_tags(text: str, rules: Iterable[Tuple[str, re.Pattern]]) -> List[str]:
    tags: List[str] = []
    for tag, pattern in rules:
        if pattern.search(text):
            tags.append(tag)
    return tags


def consequence(text: str, severe_actual: bool = False, any_injury: bool = False) -> Tuple[str, int]:
    if severe_actual:
        return "actual_severe", 5
    for label, score, pattern in COMPILED_CONSEQUENCE_RULES:
        if pattern.search(text):
            return label, score
    if any_injury:
        return "actual_injury", 2
    return "unknown_or_low", 0


def choose_theme_candidate(hazard_tags: List[str], control_tags: List[str]) -> str:
    real_hazard = [x for x in hazard_tags if x and x.lower() != "unknown"]
    real_control = [x for x in control_tags if x and x.lower() != "unknown"]
    if real_hazard:
        if real_control:
            return f"{real_hazard[0]}__{real_control[0]}"
        return real_hazard[0]
    if real_control:
        return real_control[0]
    return "unclassified"


def pipe_join(values: Iterable[str]) -> str:
    seen = []
    for value in values:
        value = clean_text_value(value).strip().lower().replace(" ", "_")
        if value and value not in {"nan", "none", "unknown"} and value not in seen:
            seen.append(value)
    return "|".join(seen)


def split_pipe(value: object) -> List[str]:
    text = clean_text_value(value)
    if not text:
        return []
    out = []
    for token in text.replace(";", "|").split("|"):
        token = token.strip()
        if token and token.lower() not in {"nan", "none", "unknown"}:
            out.append(token)
    return out


def tag_count(series: pd.Series) -> Counter:
    c: Counter = Counter()
    for value in series.fillna("").astype(str):
        for token in split_pipe(value):
            c[token] += 1
    return c


TAG_OUTPUT_COLUMNS = [
    "hazard_tags",
    "control_failure_tags",
    "energy_source_tags",
    "action_type_tags",
    "consequence_potential",
    "consequence_score",
    "risk_theme_candidate",
]

METHOD_COLUMNS = [
    "hazard_tag_method",
    "hazard_tag_confidence",
    "hazard_tag_similarity",
    "control_failure_tag_method",
    "control_failure_tag_confidence",
    "control_failure_tag_similarity",
    "energy_source_tag_method",
    "energy_source_tag_confidence",
    "energy_source_tag_similarity",
    "action_type_tag_method",
    "action_type_tag_confidence",
    "action_type_tag_similarity",
]

CATEGORY_TO_OUTPUT = {
    "hazard_type": ("hazard_tags", "hazard_tag_method", "hazard_tag_confidence", "hazard_tag_similarity"),
    "control_failure": ("control_failure_tags", "control_failure_tag_method", "control_failure_tag_confidence", "control_failure_tag_similarity"),
    "energy_source": ("energy_source_tags", "energy_source_tag_method", "energy_source_tag_confidence", "energy_source_tag_similarity"),
    "action_type": ("action_type_tags", "action_type_tag_method", "action_type_tag_confidence", "action_type_tag_similarity"),
}


def get_category_threshold(category: str) -> float:
    defaults = {
        "hazard_type": getattr(cfg, "TAG_EMBEDDING_FALLBACK_HAZARD_THRESHOLD", cfg.TAG_EMBEDDING_FALLBACK_THRESHOLD),
        "control_failure": getattr(cfg, "TAG_EMBEDDING_FALLBACK_CONTROL_THRESHOLD", cfg.TAG_EMBEDDING_FALLBACK_THRESHOLD),
        "energy_source": getattr(cfg, "TAG_EMBEDDING_FALLBACK_ENERGY_THRESHOLD", cfg.TAG_EMBEDDING_FALLBACK_THRESHOLD),
        "action_type": getattr(cfg, "TAG_EMBEDDING_FALLBACK_ACTION_THRESHOLD", cfg.TAG_EMBEDDING_FALLBACK_THRESHOLD),
    }
    return float(defaults.get(category, cfg.TAG_EMBEDDING_FALLBACK_THRESHOLD))


def confidence_from_similarity(similarity: float, threshold: float) -> str:
    high_threshold = float(getattr(cfg, "TAG_EMBEDDING_FALLBACK_HIGH_CONFIDENCE_THRESHOLD", 0.46))
    if similarity >= high_threshold:
        return "high"
    if similarity >= threshold:
        return "medium"
    return "low"


def rule_extract(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    text = df[text_column].fillna("").astype(str).map(clean_text_value)
    normalized_text = text.map(normalize_for_rules)
    severe = df.get("severe_actual", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    injury = df.get("any_injury", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    rows = []
    for i, t in enumerate(normalized_text):
        h = find_tags(t, COMPILED_TAG_RULES["hazard_type"])
        c = find_tags(t, COMPILED_TAG_RULES["control_failure"])
        e = find_tags(t, COMPILED_TAG_RULES["energy_source"])
        a = find_tags(t, COMPILED_TAG_RULES["action_type"])
        label, score = consequence(t, bool(severe.iloc[i]), bool(injury.iloc[i]))
        rows.append({
            "hazard_tags": pipe_join(h),
            "control_failure_tags": pipe_join(c),
            "energy_source_tags": pipe_join(e),
            "action_type_tags": pipe_join(a),
            "consequence_potential": label,
            "consequence_score": score,
            "risk_theme_candidate": choose_theme_candidate(h, c),
            "hazard_tag_method": "rule" if h else "none",
            "hazard_tag_confidence": "high" if h else "low",
            "hazard_tag_similarity": np.nan,
            "control_failure_tag_method": "rule" if c else "none",
            "control_failure_tag_confidence": "high" if c else "low",
            "control_failure_tag_similarity": np.nan,
            "energy_source_tag_method": "rule" if e else "none",
            "energy_source_tag_confidence": "high" if e else "low",
            "energy_source_tag_similarity": np.nan,
            "action_type_tag_method": "rule" if a else "none",
            "action_type_tag_confidence": "high" if a else "low",
            "action_type_tag_similarity": np.nan,
        })
    return pd.DataFrame(rows, index=df.index)


def load_embedding_summary() -> dict:
    """Load Step 01 embedding metadata/artifacts.

    Step 02 should reuse Step 01 outputs. The newer
    embedding_artifacts.json contains explicit paths; the older
    01_embedding_summary.json is kept as a backward-compatible fallback.
    """
    for path in [cfg.STEP_01_DIR / "embedding_artifacts.json", cfg.STEP_01_DIR / "01_embedding_summary.json"]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["artifact_metadata_path"] = str(path)
            return data
    return {}


def encode_texts_for_definitions(texts: List[str], provider: str, model_name: str) -> np.ndarray:
    provider = provider.lower()
    if provider == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        x = model.encode(texts, batch_size=min(64, max(1, len(texts))), show_progress_bar=False, normalize_embeddings=True)
        return normalize_embeddings(np.asarray(x, dtype=np.float32))
    if provider == "azure_openai":
        from openai import AzureOpenAI

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if not endpoint or not api_key:
            raise EnvironmentError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for Azure OpenAI embedding fallback.")
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        response = client.embeddings.create(model=model_name, input=texts)
        vectors = [d.embedding for d in response.data]
        return normalize_embeddings(np.asarray(vectors, dtype=np.float32))
    if provider == "tfidf_svd":
        from joblib import load

        artifacts = load_embedding_summary()
        model_path_raw = artifacts.get("tfidf_svd_pipeline_path")
        model_path = Path(model_path_raw) if model_path_raw else cfg.STEP_01_DIR / "models" / "embedding_model_tfidf_svd.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Step 01 TF-IDF/SVD embedding model: {model_path}")
        pipe = load(model_path)
        return normalize_embeddings(np.asarray(pipe.transform(texts), dtype=np.float32))
    raise ValueError(f"Unsupported embedding provider for fallback: {provider}")


def load_aligned_event_embeddings(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], dict]:
    artifacts = load_embedding_summary()
    embeddings_path = Path(artifacts.get("embeddings_path", cfg.TEXT_EMBEDDINGS_PATH))
    id_map_path = Path(artifacts.get("event_id_map_path", cfg.TEXT_EMBEDDING_EVENT_IDS_PATH))
    if not embeddings_path.exists() or not id_map_path.exists():
        return None, {
            "available": False,
            "reason": "Step 01 embedding files not found",
            "embeddings_path": str(embeddings_path),
            "event_id_map_path": str(id_map_path),
        }
    # mmap_mode avoids copying the full array until the aligned rows are needed.
    x = normalize_embeddings(np.load(embeddings_path, mmap_mode="r").astype(np.float32))
    id_map = load_table(id_map_path)
    if "event_id" not in df.columns or "event_id" not in id_map.columns:
        return None, {"available": False, "reason": "event_id column missing from data or embedding id map"}
    if len(x) != len(id_map):
        return None, {"available": False, "reason": f"embedding rows {len(x)} != id map rows {len(id_map)}"}

    event_to_pos = {str(eid): i for i, eid in enumerate(id_map["event_id"].astype(str).tolist())}
    positions = df["event_id"].astype(str).map(event_to_pos)
    valid = positions.notna().to_numpy()
    if valid.sum() == 0:
        return None, {"available": False, "reason": "no event IDs matched embeddings"}
    aligned = np.full((len(df), x.shape[1]), np.nan, dtype=np.float32)
    aligned[valid] = x[positions[valid].astype(int).to_numpy()]
    return aligned, {
        "available": True,
        "embedding_rows": int(len(x)),
        "matched_rows": int(valid.sum()),
        "unmatched_rows": int((~valid).sum()),
        "embedding_dim": int(x.shape[1]),
        "embeddings_path": str(embeddings_path),
        "event_id_map_path": str(id_map_path),
        "artifact_metadata_path": artifacts.get("artifact_metadata_path"),
    }


def build_definition_tables(provider: str, model_name: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    definition_tables: Dict[str, pd.DataFrame] = {}
    definition_embeddings: Dict[str, np.ndarray] = {}
    for category, defs in TAG_DEFINITIONS.items():
        labels = [x[0] for x in defs]
        texts = [f"Safety tag: {label.replace('_', ' ')}. Definition: {desc}" for label, desc in defs]
        definition_tables[category] = pd.DataFrame({
            "tag_category": category,
            "tag": labels,
            "definition": [x[1] for x in defs],
            "definition_text_used_for_embedding": texts,
        })

    # Cache tag-definition embeddings. This keeps Step 02 directly tied to the
    # Step 01 vector space while avoiding repeated definition encoding on reruns.
    safe_provider = re.sub(r"[^A-Za-z0-9_]+", "_", provider)
    safe_model = re.sub(r"[^A-Za-z0-9_]+", "_", model_name or "model")[:80]
    cache_path = cfg.STEP_02_DIR / f"tag_definition_embeddings_{safe_provider}_{safe_model}.npz"
    meta_path = cfg.STEP_02_DIR / f"tag_definition_embeddings_{safe_provider}_{safe_model}.json"
    expected_counts = {cat: len(tbl) for cat, tbl in definition_tables.items()}

    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("provider") == provider and meta.get("model_name") == model_name and meta.get("category_counts") == expected_counts:
                cached = np.load(cache_path)
                for category in definition_tables:
                    definition_embeddings[category] = normalize_embeddings(cached[category].astype(np.float32))
                return definition_tables, definition_embeddings
        except Exception:
            pass

    for category, tbl in definition_tables.items():
        emb = encode_texts_for_definitions(tbl["definition_text_used_for_embedding"].tolist(), provider, model_name)
        definition_embeddings[category] = emb

    np.savez_compressed(cache_path, **definition_embeddings)
    save_json({
        "provider": provider,
        "model_name": model_name,
        "category_counts": expected_counts,
        "cache_path": str(cache_path),
        "created_from_step_01_artifacts": load_embedding_summary(),
    }, meta_path)
    return definition_tables, definition_embeddings

def apply_embedding_fallback(df: pd.DataFrame, tag_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    requested = bool(getattr(cfg, "TAG_EMBEDDING_FALLBACK_ENABLED", True))
    summary = {"requested": requested, "available": False, "applied_rows_by_category": {}}
    if not requested:
        log_progress("Layer 2 embedding fallback is disabled in config.")
        summary["reason"] = "disabled in config"
        return tag_df, summary

    log_progress("Layer 2: loading and aligning Step 01 embeddings for fallback tagging...")
    event_embeddings, emb_info = load_aligned_event_embeddings(df)
    summary["event_embedding_info"] = emb_info
    if event_embeddings is None:
        summary["reason"] = emb_info.get("reason", "event embeddings unavailable")
        log_progress(f"Layer 2 skipped: {summary['reason']}")
        return tag_df, summary

    log_progress(
        "Layer 2 embeddings aligned: "
        f"{emb_info.get('matched_rows', 0):,} matched rows; "
        f"{emb_info.get('unmatched_rows', 0):,} unmatched rows."
    )
    embedding_summary = load_embedding_summary()
    provider = str(embedding_summary.get("provider") or cfg.EMBEDDING_PROVIDER)
    model_name = str(embedding_summary.get("model_name") or embedding_summary.get("model") or cfg.EMBEDDING_MODEL_NAME)
    try:
        log_progress(f"Layer 2: building/loading tag-definition embeddings using provider={provider}, model={model_name}...")
        definition_tables, definition_embeddings = build_definition_tables(provider, model_name)
    except Exception as exc:
        summary["reason"] = f"could not encode tag definitions: {exc}"
        log_progress(f"Layer 2 skipped: {summary['reason']}")
        return tag_df, summary

    tag_definition_library = pd.concat(definition_tables.values(), ignore_index=True)
    save_csv(tag_definition_library, cfg.STEP_02_DIR / "tag_definition_library.csv")

    top_k = int(getattr(cfg, "TAG_EMBEDDING_FALLBACK_TOP_K", 1))
    fill_only_empty = bool(getattr(cfg, "TAG_EMBEDDING_FALLBACK_FILL_ONLY_EMPTY", True))
    fallback_rows_total = 0

    valid_embedding_mask = ~np.isnan(event_embeddings).any(axis=1)
    log_progress("Layer 2: applying embedding fallback by tag category...")
    for category, def_emb in definition_embeddings.items():
        tag_col, method_col, conf_col, sim_col = CATEGORY_TO_OUTPUT[category]
        threshold = get_category_threshold(category)
        if fill_only_empty:
            needs_fallback = tag_df[tag_col].fillna("").astype(str).eq("").to_numpy()
        else:
            needs_fallback = np.ones(len(tag_df), dtype=bool)
        needs_fallback = needs_fallback & valid_embedding_mask
        idx = np.where(needs_fallback)[0]
        if len(idx) == 0:
            summary["applied_rows_by_category"][category] = 0
            continue

        sims = event_embeddings[idx] @ def_emb.T
        labels = np.asarray(definition_tables[category]["tag"].tolist(), dtype=object)

        if top_k <= 1:
            best_pos = np.argmax(sims, axis=1)
            best_scores = sims[np.arange(len(idx)), best_pos]
            pass_mask = best_scores >= threshold
            pass_idx = idx[pass_mask]
            if len(pass_idx) > 0:
                selected_labels = labels[best_pos[pass_mask]]
                selected_scores = best_scores[pass_mask]
                target_index = tag_df.index[pass_idx]
                tag_df.loc[target_index, tag_col] = selected_labels
                tag_df.loc[target_index, method_col] = "embedding_fallback"
                tag_df.loc[target_index, sim_col] = np.round(selected_scores.astype(float), 6)
                tag_df.loc[target_index, conf_col] = [confidence_from_similarity(float(v), threshold) for v in selected_scores]
            applied = int(len(pass_idx))
        else:
            # top_k > 1 is kept for compatibility, but is slower and rarely needed.
            applied = 0
            for local_pos, row_idx in enumerate(idx):
                row_sims = sims[local_pos]
                order = np.argsort(-row_sims)[:top_k]
                selected = [(str(labels[j]), float(row_sims[j])) for j in order if float(row_sims[j]) >= threshold]
                if not selected:
                    continue
                existing = split_pipe(tag_df.at[tag_df.index[row_idx], tag_col])
                merged = existing + [tag for tag, _ in selected]
                tag_df.at[tag_df.index[row_idx], tag_col] = pipe_join(merged)
                tag_df.at[tag_df.index[row_idx], method_col] = "embedding_fallback" if not existing else "rule_plus_embedding_fallback"
                max_sim = max(score for _, score in selected)
                tag_df.at[tag_df.index[row_idx], conf_col] = confidence_from_similarity(max_sim, threshold)
                tag_df.at[tag_df.index[row_idx], sim_col] = round(max_sim, 6)
                applied += 1
        summary["applied_rows_by_category"][category] = int(applied)
        fallback_rows_total += applied
        log_progress(f"Layer 2: {category} fallback assigned {applied:,} rows.")

    tag_df["risk_theme_candidate"] = [
        choose_theme_candidate(split_pipe(h), split_pipe(c))
        for h, c in zip(tag_df["hazard_tags"], tag_df["control_failure_tags"])
    ]
    log_progress(f"Layer 2 complete: {fallback_rows_total:,} total fallback assignments.")
    summary.update({
        "available": True,
        "provider": provider,
        "model_name": model_name,
        "top_k": top_k,
        "fill_only_empty": fill_only_empty,
        "fallback_assignments_total": int(fallback_rows_total),
    })
    return tag_df, summary


# Optional Azure OpenAI extraction. Not used by default.
LLM_SYSTEM_PROMPT = """You are an EHS safety analyst. Extract structured safety risk information from one record.
Return only valid JSON with these keys:
- hazard_tags: array of concise snake_case labels.
- control_failure_tags: array of concise snake_case labels.
- energy_source_tags: array of concise snake_case labels.
- action_type_tags: array of concise snake_case labels.
- consequence_potential: one of unknown_or_low, low_potential, moderate_potential, serious_potential, fatality_potential, actual_injury, actual_severe.
- consequence_score: integer from 0 to 5.
- risk_theme_candidate: concise snake_case theme using the main hazard and control failure.
Do not invent facts beyond the record text.
"""


def normalize_llm_list(value: object) -> str:
    if isinstance(value, list):
        return pipe_join(str(x).strip() for x in value if str(x).strip())
    if value is None:
        return ""
    return pipe_join(str(value).replace(",", "|").split("|"))


def llm_extract_rows(df: pd.DataFrame, text: pd.Series, severe: pd.Series, injury: pd.Series, model_name: str, sleep_seconds: float = 0.0) -> pd.DataFrame:
    try:
        from openai import AzureOpenAI
    except Exception as exc:
        raise ImportError("openai package is required for --backend azure_openai") from exc
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    if not endpoint or not api_key:
        raise EnvironmentError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for --backend azure_openai.")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    outputs = []
    for i, t in enumerate(text):
        record = f"Record ID: {df.iloc[i].get('event_id')}\nSource: {df.iloc[i].get('source_type')}\nSevere actual injury: {bool(severe.iloc[i])}\nAny injury: {bool(injury.iloc[i])}\n\n{str(t)[:12000]}"
        parsed = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": LLM_SYSTEM_PROMPT}, {"role": "user", "content": record}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(resp.choices[0].message.content or "{}")
                break
            except Exception:
                if attempt == 2:
                    parsed = None
                else:
                    time.sleep(2 + attempt)
        if not parsed:
            normalized = normalize_for_rules(t)
            h = find_tags(normalized, COMPILED_TAG_RULES["hazard_type"])
            c = find_tags(normalized, COMPILED_TAG_RULES["control_failure"])
            e = find_tags(normalized, COMPILED_TAG_RULES["energy_source"])
            a = find_tags(normalized, COMPILED_TAG_RULES["action_type"])
            label, score = consequence(normalized, bool(severe.iloc[i]), bool(injury.iloc[i]))
            parsed = {
                "hazard_tags": h,
                "control_failure_tags": c,
                "energy_source_tags": e,
                "action_type_tags": a,
                "consequence_potential": label,
                "consequence_score": score,
                "risk_theme_candidate": choose_theme_candidate(h, c),
            }
        outputs.append({
            "hazard_tags": normalize_llm_list(parsed.get("hazard_tags")),
            "control_failure_tags": normalize_llm_list(parsed.get("control_failure_tags")),
            "energy_source_tags": normalize_llm_list(parsed.get("energy_source_tags")),
            "action_type_tags": normalize_llm_list(parsed.get("action_type_tags")),
            "consequence_potential": str(parsed.get("consequence_potential", "unknown_or_low")),
            "consequence_score": pd.to_numeric(parsed.get("consequence_score", 0), errors="coerce"),
            "risk_theme_candidate": str(parsed.get("risk_theme_candidate", "unclassified")),
            "hazard_tag_method": "azure_openai" if parsed.get("hazard_tags") else "none",
            "hazard_tag_confidence": "medium" if parsed.get("hazard_tags") else "low",
            "hazard_tag_similarity": np.nan,
            "control_failure_tag_method": "azure_openai" if parsed.get("control_failure_tags") else "none",
            "control_failure_tag_confidence": "medium" if parsed.get("control_failure_tags") else "low",
            "control_failure_tag_similarity": np.nan,
            "energy_source_tag_method": "azure_openai" if parsed.get("energy_source_tags") else "none",
            "energy_source_tag_confidence": "medium" if parsed.get("energy_source_tags") else "low",
            "energy_source_tag_similarity": np.nan,
            "action_type_tag_method": "azure_openai" if parsed.get("action_type_tags") else "none",
            "action_type_tag_confidence": "medium" if parsed.get("action_type_tags") else "low",
            "action_type_tag_similarity": np.nan,
        })
        if sleep_seconds:
            time.sleep(sleep_seconds)
        if (i + 1) % 100 == 0:
            print(f"LLM-tagged {i + 1}/{len(text)} records")
    return pd.DataFrame(outputs, index=df.index)


def pct(n: int, denom: int) -> float:
    return float(n / denom) if denom else 0.0


def make_coverage_summary(df: pd.DataFrame, backend: str, embedding_summary: dict) -> dict:
    n = len(df)
    category_summary = {}
    for category, (tag_col, method_col, conf_col, sim_col) in CATEGORY_TO_OUTPUT.items():
        assigned = df[tag_col].fillna("").astype(str).ne("")
        category_summary[category] = {
            "assigned_count": int(assigned.sum()),
            "assigned_pct": pct(int(assigned.sum()), n),
            "unassigned_count": int((~assigned).sum()),
            "unassigned_pct": pct(int((~assigned).sum()), n),
            "method_counts": df[method_col].fillna("none").value_counts().to_dict(),
            "confidence_counts": df[conf_col].fillna("low").value_counts().to_dict(),
            "top_tags": dict(tag_count(df[tag_col]).most_common(30)),
        }
    no_label = df["no_label_assigned"].fillna(False).astype(bool)
    return {
        "row_count": int(n),
        "backend": backend,
        "tag_layers": {
            "layer_1_rules": True,
            "layer_2_embedding_fallback": embedding_summary,
            "layer_3_unmatched_text_discovery": bool(getattr(cfg, "TAG_UNKNOWN_DISCOVERY_ENABLED", True)),
        },
        "no_label_assigned_count": int(no_label.sum()),
        "no_label_assigned_pct": pct(int(no_label.sum()), n),
        "category_coverage": category_summary,
        "consequence_potential_counts": df["consequence_potential"].fillna("unknown_or_low").value_counts().to_dict(),
    }


def top_terms_from_texts(texts: List[str], n: int = 12) -> str:
    if not texts:
        return ""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=1,
            max_df=0.90,
            ngram_range=(1, 2),
            stop_words="english",
            strip_accents="unicode",
        )
        x = vectorizer.fit_transform(texts)
        scores = np.asarray(x.sum(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        order = np.argsort(-scores)[:n]
        return "|".join(terms[order].tolist())
    except Exception:
        words = Counter()
        for text in texts:
            for word in re.findall(r"\b[a-zA-Z][a-zA-Z]{2,}\b", normalize_for_rules(text)):
                if word not in {"the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "not", "task", "audit", "inspection"}:
                    words[word] += 1
        return "|".join([w for w, _ in words.most_common(n)])


def save_unmatched_discovery(df: pd.DataFrame, output_dir: Path, text_column: str) -> dict:
    enabled = bool(getattr(cfg, "TAG_UNKNOWN_DISCOVERY_ENABLED", True))
    if not enabled:
        log_progress("Layer 3 unknown-text discovery is disabled in config.")
        return {"enabled": False}

    ensure_dir(output_dir)
    sample_size = int(getattr(cfg, "TAG_UNKNOWN_EXPORT_SAMPLE_SIZE", 5000))
    no_label = df["no_label_assigned"].fillna(False).astype(bool)
    unknown_df = df.loc[no_label].copy()
    log_progress(f"Layer 3: found {len(unknown_df):,} rows with no assigned label.")
    if unknown_df.empty:
        return {"enabled": True, "no_label_rows": 0}

    export_cols = [
        c for c in [
            "event_id", "source_type", "source_id", "event_date", "site", "department", text_column,
            "hazard_tags", "control_failure_tags", "energy_source_tags", "action_type_tags",
            "consequence_potential", "risk_theme_candidate",
        ]
        if c in unknown_df.columns
    ]
    sample = unknown_df.sample(n=min(sample_size, len(unknown_df)), random_state=cfg.RANDOM_STATE) if len(unknown_df) > sample_size else unknown_df
    unknown_sample_path = output_dir / "unassigned_any_label_records_sample.csv.gz"
    save_csv(sample[export_cols], unknown_sample_path)
    log_progress(f"Layer 3: saved no-label sample to {unknown_sample_path}")

    category_paths = {}
    for category, (tag_col, _, _, _) in CATEGORY_TO_OUTPUT.items():
        mask = df[tag_col].fillna("").astype(str).eq("")
        cat_df = df.loc[mask]
        if cat_df.empty:
            continue
        cat_sample = cat_df.sample(n=min(sample_size, len(cat_df)), random_state=cfg.RANDOM_STATE) if len(cat_df) > sample_size else cat_df
        path = output_dir / f"unassigned_{category}_sample.csv.gz"
        save_csv(cat_sample[export_cols], path)
        category_paths[category] = str(path)

    # Optional unknown clustering, using existing event embeddings if available.
    cluster_info = {"enabled": False}
    try:
        event_embeddings, emb_info = load_aligned_event_embeddings(df)
        if event_embeddings is not None:
            discovery_sample_size = int(getattr(cfg, "TAG_UNKNOWN_DISCOVERY_SAMPLE_SIZE", 50000))
            cluster_count = int(getattr(cfg, "TAG_UNKNOWN_CLUSTER_COUNT", 25))
            unknown_idx_all = np.where(no_label.to_numpy() & ~np.isnan(event_embeddings).any(axis=1))[0]
            if len(unknown_idx_all) >= max(10, cluster_count):
                rng = np.random.default_rng(cfg.RANDOM_STATE)
                if len(unknown_idx_all) > discovery_sample_size:
                    unknown_idx = rng.choice(unknown_idx_all, size=discovery_sample_size, replace=False)
                else:
                    unknown_idx = unknown_idx_all
                n_clusters = min(cluster_count, max(2, len(unknown_idx) // 50))
                from sklearn.cluster import MiniBatchKMeans

                km = MiniBatchKMeans(n_clusters=n_clusters, random_state=cfg.RANDOM_STATE, batch_size=2048, n_init="auto")
                labels = km.fit_predict(event_embeddings[unknown_idx])
                rows = []
                unknown_sample_df = df.iloc[unknown_idx].copy()
                unknown_sample_df["unknown_cluster"] = labels
                for lab in sorted(np.unique(labels)):
                    cluster_rows = unknown_sample_df[unknown_sample_df["unknown_cluster"].eq(lab)]
                    texts = cluster_rows[text_column].fillna("").astype(str).head(500).tolist()
                    source_mix = cluster_rows.get("source_type", pd.Series(dtype=str)).fillna("").astype(str).value_counts().head(10).to_dict()
                    examples = cluster_rows.head(10)["event_id"].astype(str).tolist() if "event_id" in cluster_rows.columns else []
                    rows.append({
                        "unknown_cluster": int(lab),
                        "cluster_size_in_sample": int(len(cluster_rows)),
                        "top_terms": top_terms_from_texts(texts, n=15),
                        "source_type_mix": json.dumps(source_mix),
                        "example_event_ids": "|".join(examples),
                    })
                cluster_path = output_dir / "unassigned_text_discovery_clusters.csv"
                save_csv(pd.DataFrame(rows), cluster_path)
                log_progress(f"Layer 3: saved unknown-text discovery clusters to {cluster_path}")
                cluster_info = {
                    "enabled": True,
                    "cluster_path": str(cluster_path),
                    "cluster_count": int(n_clusters),
                    "clustered_sample_rows": int(len(unknown_idx)),
                    "embedding_info": emb_info,
                }
            else:
                cluster_info = {"enabled": True, "reason": "not enough unknown rows with embeddings", "embedding_info": emb_info}
    except Exception as exc:
        cluster_info = {"enabled": True, "reason": f"unknown clustering failed: {exc}"}

    return {
        "enabled": True,
        "no_label_rows": int(len(unknown_df)),
        "no_label_sample_path": str(unknown_sample_path),
        "category_sample_paths": category_paths,
        "unknown_cluster_discovery": cluster_info,
    }


def main() -> None:
    log_progress("Starting safety tag extraction.")
    parser = argparse.ArgumentParser(description="Extract safety tags from text using rules, embedding fallback, optional Azure OpenAI, and unknown discovery outputs.")
    parser.add_argument("--input", default=cfg.SAFETY_TEXT_EVENT_PATH)
    parser.add_argument("--output-dir", default=cfg.STEP_02_DIR)
    parser.add_argument("--text-column", default=cfg.TEXT_COLUMN)
    parser.add_argument("--backend", choices=["rules", "azure_openai"], default=cfg.TAG_BACKEND)
    parser.add_argument("--model-name", default=cfg.AZURE_OPENAI_CHAT_DEPLOYMENT)
    parser.add_argument("--sleep-seconds", type=float, default=cfg.TAG_LLM_SLEEP_SECONDS)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    log_progress(f"Loading input table from {args.input} ...")
    df = load_table(args.input)
    log_progress(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' is missing from input file: {args.input}")

    text = df[args.text_column].fillna("").astype(str).map(clean_text_value)
    severe = df.get("severe_actual", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    injury = df.get("any_injury", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    log_progress(f"Layer 1: extracting safety tags with backend={args.backend}...")
    if args.backend == "azure_openai":
        tag_df = llm_extract_rows(df, text, severe, injury, args.model_name, args.sleep_seconds)
    else:
        tag_df = rule_extract(df, args.text_column)
    log_progress("Layer 1 complete.")

    df = df.copy()
    for col in TAG_OUTPUT_COLUMNS + METHOD_COLUMNS:
        df[col] = tag_df[col].values

    log_progress("Copying Layer 1 tag columns back to the event table...")

    # Layer 2: semantic fallback to tag definitions, using Step 01 embeddings.
    df_tag_subset = df[TAG_OUTPUT_COLUMNS + METHOD_COLUMNS].copy()
    df_tag_subset, embedding_fallback_summary = apply_embedding_fallback(df, df_tag_subset)
    for col in TAG_OUTPUT_COLUMNS + METHOD_COLUMNS:
        df[col] = df_tag_subset[col].values

    log_progress("Computing tag coverage flags...")
    # Booleans should reflect real labels only. Empty means still unassigned.
    df["has_hazard_tag"] = df["hazard_tags"].fillna("").astype(str).ne("")
    df["has_control_failure_tag"] = df["control_failure_tags"].fillna("").astype(str).ne("")
    df["has_energy_source_tag"] = df["energy_source_tags"].fillna("").astype(str).ne("")
    df["has_action_type_tag"] = df["action_type_tags"].fillna("").astype(str).ne("")
    df["no_hazard_tag"] = ~df["has_hazard_tag"]
    df["no_control_failure_tag"] = ~df["has_control_failure_tag"]
    df["no_energy_source_tag"] = ~df["has_energy_source_tag"]
    df["no_action_type_tag"] = ~df["has_action_type_tag"]
    df["no_label_assigned"] = (
        df["no_hazard_tag"]
        & df["no_control_failure_tag"]
        & df["no_energy_source_tag"]
        & df["no_action_type_tag"]
        & pd.to_numeric(df["consequence_score"], errors="coerce").fillna(0).eq(0)
    )

    output_path = output_dir / "safety_text_event_tagged.csv.gz"
    log_progress(f"Saving tagged output to {output_path} ...")
    save_csv(df, output_path)
    log_progress("Tagged output saved.")

    log_progress("Layer 3: creating unmatched-text discovery outputs...")
    unknown_discovery = save_unmatched_discovery(df, output_dir, args.text_column)
    summary = make_coverage_summary(df, args.backend, embedding_fallback_summary)
    summary["output_path"] = str(output_path)
    summary["unmatched_text_discovery"] = unknown_discovery
    summary_path = output_dir / "02_safety_tag_summary.json"
    log_progress(f"Saving summary JSON to {summary_path} ...")
    save_json(summary, summary_path)
    log_progress("Step 02 completed successfully.")
    print(json.dumps(summary, indent=2)[:6000])


if __name__ == "__main__":
    main()
