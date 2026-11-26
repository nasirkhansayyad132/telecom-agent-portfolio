"""Utility tools the agent can call."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

TICKET_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.json"
_ticket_lock = threading.Lock()


def _extract_phone(text: str) -> Optional[str]:
    match = re.search(r"(\+?\d[\d\s-]{6,})", text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_name(text: str) -> Optional[str]:
    match = re.search(r"(?:name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,40})", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().title()
    return None


def _extract_budget(text: str) -> Optional[int]:
    match = re.search(r"(?:budget|afn|afs|balance)\s*([0-9]{2,6})", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # fallback to first plain number
    match = re.search(r"([0-9]{2,6})", text)
    return int(match.group(1)) if match else None


def _extract_usage(text: str) -> Optional[int]:
    match = re.search(r"([0-9]{1,4})\s*gb", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def detect_intent(message: str) -> Tuple[Optional[str], Dict, bool]:
    """Return (tool_name, args, is_ready) based on simple keyword checks."""
    lower = message.lower()
    if "ticket" in lower or "complaint" in lower:
        name = _extract_name(message)
        phone = _extract_phone(message)
        issue = message.strip()
        ready = bool(name and phone and issue)
        return "create_ticket", {"name": name, "phone": phone, "issue": issue}, ready

    if any(keyword in lower for keyword in ["bundle", "plan", "package", "recommend", "balance"]):
        budget = _extract_budget(lower)
        usage = _extract_usage(lower)
        # Ready if we at least have a budget; usage is optional
        ready = budget is not None
        return "recommend_bundle", {"budget_afn": budget, "usage_gb": usage}, ready

    return None, {}, False


def create_ticket(name: str, phone: str, issue: str) -> Dict:
    """Persist a support ticket to a JSON log."""
    TICKET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ticket_id = f"TIC-{uuid.uuid4().hex[:8].upper()}"
    ticket = {
        "ticket_id": ticket_id,
        "name": name,
        "phone": phone,
        "issue": issue,
        "status": "open",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with _ticket_lock:
        if TICKET_LOG_PATH.exists():
            try:
                existing = json.loads(TICKET_LOG_PATH.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except json.JSONDecodeError:
                existing = []
        else:
            existing = []

        existing.append(ticket)
        TICKET_LOG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return ticket


AWCC_PACKAGES = [
    # Data bundles (activation codes from AWCC)
    {"name": "Daily 1GB data", "type": "data", "price_afn": 50, "data_gb": 1, "activation_code": "*477*50*1#"},
    {"name": "Daily 2GB data", "type": "data", "price_afn": 77, "data_gb": 2, "activation_code": "*477*77*1#"},
    {"name": "Daily 3.5GB data", "type": "data", "price_afn": 99, "data_gb": 3.5, "activation_code": "*477*99*1#"},
    {"name": "Weekly 3GB data", "type": "data", "price_afn": 210, "data_gb": 3, "activation_code": "*477*210*1#"},
    {"name": "Monthly 3GB data", "type": "data", "price_afn": 250, "data_gb": 3, "activation_code": "*477*4*1#"},
    {"name": "Monthly 4GB data", "type": "data", "price_afn": 360, "data_gb": 4, "activation_code": "*477*390*1#"},
    {"name": "Monthly 5GB data", "type": "data", "price_afn": 390, "data_gb": 5, "activation_code": "*477*391*1#"},
    {"name": "Monthly 6GB data", "type": "data", "price_afn": 400, "data_gb": 6, "activation_code": "*477*5*1#"},
    {"name": "Monthly 8GB data", "type": "data", "price_afn": 500, "data_gb": 8, "activation_code": "*477*500*1#"},
    {"name": "Monthly 10GB data", "type": "data", "price_afn": 600, "data_gb": 10, "activation_code": "*477*6*1#"},
    {"name": "Monthly 13.5GB data", "type": "data", "price_afn": 800, "data_gb": 13.5, "activation_code": "*477*8*1#"},
    {"name": "Monthly 16GB data", "type": "data", "price_afn": 888, "data_gb": 16, "activation_code": "*477*17*1#"},
    {"name": "Monthly 20GB data", "type": "data", "price_afn": 999, "data_gb": 20, "activation_code": "*477*999*1#"},
    {"name": "Monthly 23GB data", "type": "data", "price_afn": 1100, "data_gb": 23, "activation_code": "*477*10*1#"},
    {"name": "Monthly 30GB data", "type": "data", "price_afn": 1400, "data_gb": 30, "activation_code": "*477*30*1#"},
    {"name": "Monthly 44GB data", "type": "data", "price_afn": 1800, "data_gb": 44, "activation_code": "*477*16*1#"},
    {"name": "Monthly 75GB data", "type": "data", "price_afn": 3000, "data_gb": 75, "activation_code": "*477*27*1#"},
    # Combo bundles
    {"name": "Daily AFN 30 combo", "type": "combo", "price_afn": 30, "data_gb": 0.4, "minutes": "40 on-net", "sms": 40, "activation_code": "*255*33*1#"},
    {"name": "Weekly AFN 90 combo", "type": "combo", "price_afn": 90, "data_gb": 0.512, "minutes": "250 on-net", "sms": 250, "activation_code": "*255*90*1#"},
    {"name": "Weekly AFN 210 combo", "type": "combo", "price_afn": 210, "data_gb": 2, "minutes": "150 on-net + 15 off-net", "sms": 150, "activation_code": "*255*210*1#"},
    {"name": "Monthly AFN 219 combo", "type": "combo", "price_afn": 219, "data_gb": 1, "minutes": "1100 on-net", "sms": 1100, "activation_code": "*255*220*1#"},
    {"name": "Monthly AFN 399 combo", "type": "combo", "price_afn": 399, "data_gb": 2, "minutes": "1500 on-net", "sms": 1500, "activation_code": "*255*399*1#"},
    {"name": "Monthly AFN 499 combo", "type": "combo", "price_afn": 499, "data_gb": 4, "minutes": "1500 on-net", "sms": 1500, "activation_code": "*477*499*1#"},
    {"name": "Monthly AFN 619 combo", "type": "combo", "price_afn": 619, "data_gb": 9, "minutes": "1100 on-net + 1100 off-net", "sms": 1100, "activation_code": "*255*619*1#"},
    {"name": "Monthly AFN 720 combo", "type": "combo", "price_afn": 720, "data_gb": 8, "minutes": "1100 on-net + 80 off-net", "sms": 1100, "activation_code": "*255*720*1#"},
    {"name": "Monthly AFN 290 combo", "type": "combo", "price_afn": 290, "data_gb": 1, "minutes": "1024 on-net + 24 off-net", "sms": 1024, "activation_code": "*255*290*1#"},
    # Voice bundles
    {"name": "Daily 30 min voice", "type": "voice", "price_afn": 25, "minutes": "30 all-net", "activation_code": "*255*20*1#"},
    {"name": "Weekly 100 min voice", "type": "voice", "price_afn": 80, "minutes": "100 all-net", "activation_code": "*255*8*1#"},
    {"name": "Monthly 255 min voice", "type": "voice", "price_afn": 200, "minutes": "255 all-net", "activation_code": "*255*9*1#"},
    {"name": "Monthly 585 min voice", "type": "voice", "price_afn": 450, "minutes": "585 all-net", "activation_code": "*255*3*1#"},
    {"name": "Monthly 1250 min voice", "type": "voice", "price_afn": 900, "minutes": "1250 all-net", "activation_code": "*255*4*1#"},
    {"name": "Pakistan weekly 45 min", "type": "voice", "price_afn": 150, "minutes": "45 to Pakistan", "activation_code": "*888*40*1#"},
]


def _score_package(pkg: Dict, budget_afn: int, usage_gb: Optional[int]) -> float:
    """Score packages to pick the most suitable within budget."""
    if pkg["price_afn"] > budget_afn:
        return -1

    score = pkg["price_afn"]  # prefer spending available budget
    if usage_gb is not None and pkg.get("data_gb"):
        # small bonus for meeting usage
        if pkg["data_gb"] >= usage_gb:
            score += 100
        else:
            score -= (usage_gb - pkg["data_gb"]) * 2

    # slight preference: combo > data > voice
    type_bonus = {"combo": 30, "data": 15, "voice": 5}
    score += type_bonus.get(pkg.get("type", ""), 0)
    return score


def recommend_bundle(budget_afn: int, usage_gb: Optional[int]) -> Dict:
    """Return a bundle recommendation based on budget and optional data need."""
    usage_estimate = usage_gb if usage_gb is not None else 6  # default moderate use
    best = None
    best_score = -1

    for pkg in AWCC_PACKAGES:
        score = _score_package(pkg, budget_afn, usage_gb)
        if score > best_score:
            best = pkg
            best_score = score

    if not best:
        return {
            "name": "No bundle found in budget",
            "price_afn": budget_afn,
            "details": "Try increasing budget or choosing pay-as-you-go.",
            "estimated_usage_gb": usage_estimate,
            "budget_afn": budget_afn,
        }

    # If user asked for usage, surface the closest plan within budget and one that meets usage (even if above budget).
    alt = None
    if usage_gb is not None:
        candidates = [p for p in AWCC_PACKAGES if p.get("data_gb") and p["data_gb"] >= usage_gb]
        if candidates:
            alt = sorted(candidates, key=lambda p: p["price_afn"])[0]

    details = best.get("minutes") or best.get("data_gb")
    detail_text = f"Includes {details}" if isinstance(details, str) else f"{best.get('data_gb', 0)}GB data"

    note = None
    if usage_gb is not None and best.get("data_gb") and best["data_gb"] < usage_gb and alt and alt != best:
        note = f"Closest under budget. For >= {usage_gb}GB, consider {alt['name']} at AFN {alt['price_afn']} (dial {alt.get('activation_code','')})."

    return {
        "name": best["name"],
        "type": best["type"],
        "price_afn": best["price_afn"],
        "details": detail_text,
        "data_gb": best.get("data_gb"),
        "minutes": best.get("minutes"),
        "sms": best.get("sms"),
        "activation_code": best.get("activation_code"),
        "estimated_usage_gb": usage_estimate,
        "budget_afn": budget_afn,
        "note": note,
    }


__all__ = ["create_ticket", "recommend_bundle", "detect_intent"]
