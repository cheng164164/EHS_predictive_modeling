"""Location hierarchy construction."""

from __future__ import annotations

import pandas as pd

from lookups import add_listitem_columns, prepare_listitems
from utils import standardize_columns


LOCATION_TYPE_TO_OUTPUT = {
    "Global": "global_name",
    "Corporate": "corporate_name",
    "Buisiness Unit": "business_unit_name",  # spelling comes from source system
    "Business Unit": "business_unit_name",
    "Region": "region_name",
    "Country": "country_name",
    "Site": "site_name",
    "Department": "department_name",
    "Department 2": "department_2_name",
    "Reporting location level 9": "reporting_location_level_9_name",
}


def build_location_hierarchy(location_raw: pd.DataFrame, listitems: pd.DataFrame) -> pd.DataFrame:
    """Build one row per LOCATIONID with reporting hierarchy fields.

    The source has parent-child relationships rather than a flat site/department table. This
    function walks each node's parents and extracts the named reporting levels.
    """
    loc = standardize_columns(location_raw)
    loc = add_listitem_columns(loc, listitems, specs={
        "location_category_id": ("locationcategory", "location_category"),
        "location_status_id": ("locationstatus", "location_status"),
        "location_type_id": ("locationtype", "location_type"),
    })

    for col in ["location_id", "parent_location_id", "location_type_id", "location_tree_level"]:
        if col in loc.columns:
            loc[col] = pd.to_numeric(loc[col], errors="coerce").astype("Int64")

    base_cols = [
        "location_id", "parent_location_id", "location", "location_code", "location_type_id",
        "location_type_name", "location_category_id", "location_category_name",
        "location_status_id", "location_status_name", "location_tree_level", "active", "archived"
    ]
    for col in base_cols:
        if col not in loc.columns:
            loc[col] = pd.NA

    loc_small = loc[base_cols].drop_duplicates("location_id").copy()
    loc_by_id = loc_small.set_index("location_id", drop=False).to_dict("index")

    rows: list[dict] = []
    for raw_id in loc_small["location_id"]:
        if pd.isna(raw_id):
            continue
        location_id = int(raw_id)
        current = loc_by_id.get(location_id)
        ancestors = []
        visited = set()
        safety_counter = 0
        while current is not None and safety_counter < 50:
            cid = current.get("location_id")
            if pd.isna(cid):
                break
            cid_int = int(cid)
            if cid_int in visited:
                break
            visited.add(cid_int)
            ancestors.append(current)
            parent_id = current.get("parent_location_id")
            if pd.isna(parent_id):
                break
            current = loc_by_id.get(int(parent_id))
            safety_counter += 1

        ancestors_root_to_leaf = list(reversed(ancestors))
        row = {"location_id": location_id}
        leaf = loc_by_id[location_id]
        row.update({
            "location_name": leaf.get("location"),
            "location_code": leaf.get("location_code"),
            "parent_location_id": leaf.get("parent_location_id"),
            "location_type_id": leaf.get("location_type_id"),
            "location_type_name": leaf.get("location_type_name"),
            "location_category_id": leaf.get("location_category_id"),
            "location_category_name": leaf.get("location_category_name"),
            "location_status_id": leaf.get("location_status_id"),
            "location_status_name": leaf.get("location_status_name"),
            "location_tree_level": leaf.get("location_tree_level"),
            "location_active": leaf.get("active"),
            "location_archived": leaf.get("archived"),
        })
        for out_col in LOCATION_TYPE_TO_OUTPUT.values():
            row[out_col] = pd.NA
        path_parts = []
        id_path_parts = []
        for anc in ancestors_root_to_leaf:
            loc_name = anc.get("location")
            loc_type = anc.get("location_type_name")
            anc_id = anc.get("location_id")
            if pd.notna(loc_name):
                path_parts.append(str(loc_name))
            if pd.notna(anc_id):
                id_path_parts.append(str(int(anc_id)))
            if loc_type in LOCATION_TYPE_TO_OUTPUT:
                row[LOCATION_TYPE_TO_OUTPUT[loc_type]] = loc_name
        row["location_path"] = " > ".join(path_parts)
        row["location_id_path"] = " > ".join(id_path_parts)
        rows.append(row)

    hierarchy = pd.DataFrame(rows)
    if hierarchy.empty:
        return hierarchy

    # Build fallback grouping keys for modeling when site/department are missing.
    hierarchy["site_name_filled"] = hierarchy["site_name"].fillna(hierarchy["location_name"]).fillna("Unknown")
    hierarchy["department_name_filled"] = (
        hierarchy["department_name"]
        .fillna(hierarchy["department_2_name"])
        .fillna(hierarchy["site_name"])
        .fillna(hierarchy["location_name"])
        .fillna("Unknown")
    )
    hierarchy["business_unit_name_filled"] = hierarchy["business_unit_name"].fillna("Unknown")
    hierarchy["country_name_filled"] = hierarchy["country_name"].fillna("Unknown")
    hierarchy["region_name_filled"] = hierarchy["region_name"].fillna("Unknown")

    return hierarchy.sort_values("location_id").reset_index(drop=True)
