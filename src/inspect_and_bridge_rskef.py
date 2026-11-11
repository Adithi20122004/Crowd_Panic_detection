"""
========================================================================================
RSKEF ACTUAL IMPLEMENTATION INSPECTION & VALIDATION BRIDGE
File: inspect_and_bridge_rskef.py
Directory: C:\\crowd_project\\Research_Validation\\inspect_and_bridge_rskef.py

Purpose:
  1. Inspects source files in C:\\crowdpanicdet\\r_skef_core strictly READ-ONLY (no modifications).
  2. Copies r_skef_core into C:\\crowd_project\\Research_Validation\\r_skef_core.
  3. Uses Python inspect to document all exported classes, functions, methods, and config fields.
  4. Audits # RECONSTRUCTED markers in all files.
  5. Audits RskefConfig properties to identify verified defaults vs UNVERIFIED parameters.
  6. Audits GlobalMotionCompensator methods to verify estimate_and_warp() existence.
  7. Audits features, calibration, and detector modules.
  8. Exports:
     - Results\\rskef_api_signature_report.json
     - Results\\rskef_parameter_provenance.json
========================================================================================
"""

import os
import sys
import shutil
import inspect
import json
import re
from typing import Dict, Any

ORIGINAL_PROJECT = r"C:\crowdpanicdet"
VALIDATION_ROOT = r"C:\crowd_project\Research_Validation"
LOCAL_CORE = os.path.join(VALIDATION_ROOT, "r_skef_core")
RESULTS_DIR = os.path.join(VALIDATION_ROOT, "Results")

CORE_FILES = [
    "detector.py",
    "features.py",
    "gmc.py",
    "calibration.py",
    "density.py",
    "config.py"
]

def introspect_module(module_obj) -> Dict[str, Any]:
    info = {"classes": {}, "functions": {}, "variables": []}
    for name, obj in inspect.getmembers(module_obj):
        if name.startswith("_") and name != "__init__":
            continue
        try:
            if inspect.isclass(obj) and obj.__module__ == module_obj.__name__:
                class_methods = {}
                for m_name, m_obj in inspect.getmembers(obj):
                    if not m_name.startswith("_") or m_name == "__init__":
                        if inspect.isfunction(m_obj) or inspect.ismethod(m_obj):
                            try:
                                class_methods[m_name] = str(inspect.signature(m_obj))
                            except Exception:
                                class_methods[m_name] = "unknown_signature"

                fields = []
                if hasattr(obj, "__annotations__"):
                    fields = list(obj.__annotations__.keys())

                info["classes"][name] = {
                    "methods": class_methods,
                    "fields": fields,
                    "doc": inspect.getdoc(obj) or ""
                }
            elif inspect.isfunction(obj) and obj.__module__ == module_obj.__name__:
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    sig = "unknown_signature"
                info["functions"][name] = {
                    "signature": sig,
                    "doc": inspect.getdoc(obj) or ""
                }
            elif not inspect.ismodule(obj) and not name.startswith("_"):
                info["variables"].append(name)
        except Exception:
            continue
    return info


def audit_config_parameters(config_module) -> Dict[str, Any]:
    """Inspects RskefConfig to detect verified defaults vs UNVERIFIED properties."""
    param_audit = {}
    if not hasattr(config_module, "RskefConfig"):
        return {"error": "RskefConfig class not found in config module"}

    cls = getattr(config_module, "RskefConfig")
    try:
        inst = cls()
    except Exception as e:
        param_audit["__instantiation_error__"] = str(e)
        inst = None

    # Audit properties and attributes
    for name, member in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if isinstance(member, property):
            if inst is not None:
                try:
                    val = getattr(inst, name)
                    param_audit[name] = {
                        "type": "property",
                        "status": "VERIFIED_DEFAULT",
                        "value": str(val)
                    }
                except ValueError as ve:
                    param_audit[name] = {
                        "type": "property",
                        "status": "UNVERIFIED_RAISES_VALUE_ERROR",
                        "error_message": str(ve)
                    }
                except Exception as ex:
                    param_audit[name] = {
                        "type": "property",
                        "status": "ERROR_ON_ACCESS",
                        "error_message": str(ex)
                    }
            else:
                param_audit[name] = {"type": "property", "status": "NOT_INSTANTIATED"}
        elif not inspect.isfunction(member) and not inspect.ismethod(member):
            param_audit[name] = {
                "type": "class_attribute",
                "status": "STATIC_VALUE",
                "value": str(member)
            }

    return param_audit


def inspect_and_bridge():
    print("=" * 80)
    print("STEP 1: INSPECTING & BRIDGING ACTUAL RSKEF IMPLEMENTATION")
    print(f"Source Directory (READ-ONLY): {ORIGINAL_PROJECT}")
    print(f"Validation Local Core:        {LOCAL_CORE}")
    print(f"Results Directory:            {RESULTS_DIR}")
    print("=" * 80)

    if not os.path.exists(ORIGINAL_PROJECT):
        raise FileNotFoundError(f"Original project not found at: {ORIGINAL_PROJECT}")

    src_core = os.path.join(ORIGINAL_PROJECT, "r_skef_core")
    if not os.path.exists(src_core):
        raise FileNotFoundError(f"r_skef_core directory not found at: {src_core}")

    os.makedirs(LOCAL_CORE, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    audit_records = {}
    for fname in CORE_FILES:
        src_path = os.path.join(src_core, fname)
        dst_path = os.path.join(LOCAL_CORE, fname)

        if not os.path.exists(src_path):
            print(f"  [MISSING] {fname} not found in {src_core}")
            audit_records[fname] = {"status": "MISSING", "reconstructed": None}
            continue

        shutil.copy2(src_path, dst_path)

        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            code_text = f.read()

        is_reconstructed = bool(re.search(r"reconstruct(ed)?", code_text, re.IGNORECASE))
        audit_records[fname] = {
            "status": "COPIED",
            "reconstructed": is_reconstructed,
            "line_count": len(code_text.splitlines())
        }
        tag = " [TAGGED RECONSTRUCTED IN AUDIT]" if is_reconstructed else " [ORIGINAL]"
        print(f"  Copied {fname:<15} ({audit_records[fname]['line_count']} lines){tag}")

    init_src = os.path.join(src_core, "__init__.py")
    if os.path.exists(init_src):
        shutil.copy2(init_src, os.path.join(LOCAL_CORE, "__init__.py"))
    else:
        with open(os.path.join(LOCAL_CORE, "__init__.py"), "w") as f:
            f.write("# Local RSKEF validation package\n")

    print("\nIntrospecting actual function and class APIs in r_skef_core...")
    if VALIDATION_ROOT not in sys.path:
        sys.path.insert(0, VALIDATION_ROOT)

    api_report = {"file_audit": audit_records, "modules": {}}
    for mod_name in ["config", "features", "gmc", "calibration", "density", "detector"]:
        try:
            imported_mod = __import__(f"r_skef_core.{mod_name}", fromlist=[mod_name])
            mod_info = introspect_module(imported_mod)
            api_report["modules"][mod_name] = mod_info
            print(f"\n  Module r_skef_core.{mod_name}:")
            for cname, cdata in mod_info["classes"].items():
                print(f"    Class: {cname}")
                if cdata["fields"]:
                    print(f"      Fields: {', '.join(cdata['fields'][:8])}")
                for mname, msig in cdata["methods"].items():
                    print(f"      Method: {mname}{msig}")
            for fname, fdata in mod_info["functions"].items():
                print(f"    Function: {fname}{fdata['signature']}")
        except Exception as e:
            print(f"  Failed to introspect r_skef_core.{mod_name}: {e}")
            api_report["modules"][mod_name] = {"error": str(e)}

    # Deep parameter audit for config.py
    try:
        conf_mod = __import__("r_skef_core.config", fromlist=["config"])
        param_audit = audit_config_parameters(conf_mod)
        print("\nConfig Parameters Audit:")
        for p_name, p_info in param_audit.items():
            print(f"  {p_name}: {p_info.get('status')} {p_info.get('value', p_info.get('error_message', ''))}")
    except Exception as e:
        param_audit = {"error": str(e)}

    # Save reports
    report_path = os.path.join(RESULTS_DIR, "rskef_api_signature_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(api_report, f, indent=2)

    prov_path = os.path.join(RESULTS_DIR, "rskef_parameter_provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(param_audit, f, indent=2)

    print(f"\n[SUCCESS] API Signature Report:     {report_path}")
    print(f"[SUCCESS] Parameter Provenance Report: {prov_path}")
    print("=" * 80)


if __name__ == "__main__":
    inspect_and_bridge()
