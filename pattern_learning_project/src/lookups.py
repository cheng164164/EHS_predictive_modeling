"""List-item decoding utilities."""

from __future__ import annotations

import pandas as pd

from utils import standardize_columns


LOOKUP_SPECS = {
    # incident
    "incident_category_id": ("incidentcategory", "incident_category"),
    "incident_status_id": ("incidentstatus", "incident_status"),
    # injury, when coded fields are present in future extracts
    "treatment": ("injurytreatment", "injury_treatment"),
    # task
    "task_category_id": ("taskcategory", "task_category"),
    "task_type_id": ("tasktype", "task_type"),
    "task_status_id": ("taskstatus", "task_status"),
    "source_type_id": ("module", "source_type"),
    # audit
    "audit_category_id": ("auditcategory", "audit_category"),
    "audit_type_id": ("audittype", "audit_type"),
    "audit_status_id": ("auditstatus", "audit_status"),
    # audit templates use a different status list in the same column
    "location_category_id": ("locationcategory", "location_category"),
    "location_status_id": ("locationstatus", "location_status"),
    "location_type_id": ("locationtype", "location_type"),
}


def prepare_listitems(listitem_raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean lookup table from LISTITEM_VIEW."""
    li = standardize_columns(listitem_raw)
    li = li.rename(columns={"listitemid": "list_item_id"}) if "listitemid" in li.columns else li
    if "list_item_id" not in li.columns and "list_itemid" in li.columns:
        li = li.rename(columns={"list_itemid": "list_item_id"})
    required = ["list_item_id", "list_type_code", "code", "item", "shortname", "description", "active"]
    for col in required:
        if col not in li.columns:
            li[col] = pd.NA
    li["list_item_id"] = pd.to_numeric(li["list_item_id"], errors="coerce").astype("Int64")
    li["list_type_code"] = li["list_type_code"].astype("string")
    li["item"] = li["item"].astype("string")
    li["code"] = li["code"].astype("string")
    li["shortname"] = li["shortname"].astype("string")
    return li[required].drop_duplicates(subset=["list_item_id", "list_type_code"])


def add_listitem_columns(
    df: pd.DataFrame,
    listitems: pd.DataFrame,
    specs: dict[str, tuple[str, str]] | None = None,
    allow_cross_type_fallback: bool = True,
) -> pd.DataFrame:
    """Decode id columns using LISTITEM_VIEW.

    Parameters
    ----------
    df:
        Input dataframe with snake_case columns.
    listitems:
        Clean listitem dataframe from prepare_listitems.
    specs:
        Mapping of id_column -> (expected_list_type_code, output_prefix).
    allow_cross_type_fallback:
        If True, unresolved values are decoded using LISTITEMID only. This is useful for columns
        like AUDITSTATUSID, where templates and audits share the column but use different list types.
    """
    out = df.copy()
    specs = specs or LOOKUP_SPECS
    li = listitems.copy()
    li["list_item_id"] = pd.to_numeric(li["list_item_id"], errors="coerce").astype("Int64")

    for id_col, (list_type, prefix) in specs.items():
        if id_col not in out.columns:
            continue
        temp = li[li["list_type_code"].eq(list_type)][["list_item_id", "item", "code", "shortname", "list_type_code"]]
        temp = temp.rename(
            columns={
                "list_item_id": id_col,
                "item": f"{prefix}_name",
                "code": f"{prefix}_code",
                "shortname": f"{prefix}_shortname",
                "list_type_code": f"{prefix}_list_type_code",
            }
        )
        out[id_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
        out = out.merge(temp, on=id_col, how="left")

        if allow_cross_type_fallback and out[f"{prefix}_name"].isna().any():
            fallback = li.drop_duplicates("list_item_id")[["list_item_id", "item", "code", "shortname", "list_type_code"]]
            fallback = fallback.rename(
                columns={
                    "list_item_id": id_col,
                    "item": f"{prefix}_name_fallback",
                    "code": f"{prefix}_code_fallback",
                    "shortname": f"{prefix}_shortname_fallback",
                    "list_type_code": f"{prefix}_list_type_code_fallback",
                }
            )
            out = out.merge(fallback, on=id_col, how="left")
            for suffix in ["name", "code", "shortname", "list_type_code"]:
                primary = f"{prefix}_{suffix}"
                fb = f"{prefix}_{suffix}_fallback"
                out[primary] = out[primary].fillna(out[fb])
                out = out.drop(columns=[fb])

    return out


def listitem_dictionary(listitems: pd.DataFrame) -> dict[int, str]:
    """Build a simple LISTITEMID -> ITEM dictionary."""
    li = listitems.dropna(subset=["list_item_id"]).drop_duplicates("list_item_id")
    return dict(zip(li["list_item_id"].astype(int), li["item"].astype("string")))
