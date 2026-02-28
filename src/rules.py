from dataclasses import dataclass
from datetime import datetime
import csv
import json
import random

from .config import TN_POLICY_PATH, TN_VEHICLE_REGISTRY_PATH


@dataclass
class VehicleProfile:
    plate: str
    vehicle_type: str
    access_category: str


@dataclass
class Decision:
    allowed: bool
    reason: str
    applied_rule: str
    multiplier: float = 1.0


class RulesEngine:
    def __init__(self) -> None:
        self.policy = self._load_policy()
        self.vehicle_profiles = self._load_vehicle_profiles()

    @staticmethod
    def _default_policy() -> dict:
        return {
            "free_minutes": 10,
            "hourly_rate": 2.5,
            "category_multiplier": {
                "visitor": 1.0,
                "subscribed": 0.7,
                "conditional": 1.2,
                "vip": 0.0,
                "blacklist": 999.0,
            },
            "conditional_access": {
                "start_hour": 6,
                "end_hour": 22,
            },
            "vehicle_defaults": {
                "car": "visitor",
            },
        }

    def _load_policy(self) -> dict:
        if TN_POLICY_PATH.exists():
            try:
                with open(TN_POLICY_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                policy = self._default_policy()
                policy.update(loaded)
                return policy
            except Exception:
                return self._default_policy()
        return self._default_policy()

    def _load_vehicle_profiles(self) -> dict[str, VehicleProfile]:
        profiles: dict[str, VehicleProfile] = {}
        if TN_VEHICLE_REGISTRY_PATH.exists():
            try:
                with open(TN_VEHICLE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        plate = (row.get("plate") or "").strip().upper()
                        if not plate:
                            continue
                        vehicle_type = "car"
                        access_category = (row.get("access_category") or "visitor").strip().lower()
                        profiles[plate] = VehicleProfile(plate=plate, vehicle_type=vehicle_type, access_category=access_category)
            except Exception:
                profiles = {}

        if profiles:
            return profiles

        return {
            "111 TU 2222": VehicleProfile("111 TU 2222", "car", "subscribed"),
            "777 TU 7777": VehicleProfile("777 TU 7777", "car", "vip"),
            "999 TU 0001": VehicleProfile("999 TU 0001", "car", "conditional"),
            "123 TU 4567": VehicleProfile("123 TU 4567", "car", "blacklist"),
        }

    def get_policy_snapshot(self) -> dict:
        return self.policy

    def classify_vehicle(self, plate: str) -> VehicleProfile:
        plate = (plate or "UNKNOWN").strip().upper()
        profile = self.vehicle_profiles.get(plate)
        if profile:
            return profile

        default_category = self.policy.get("vehicle_defaults", {}).get("car", "visitor")
        return VehicleProfile(plate=plate, vehicle_type="car", access_category=default_category)

    def evaluate_entry(self, profile: VehicleProfile, at: datetime | None = None) -> Decision:
        timestamp = at or datetime.now()

        category_multiplier = self.policy.get("category_multiplier", {})
        conditional_cfg = self.policy.get("conditional_access", {})
        start_hour = int(conditional_cfg.get("start_hour", 6))
        end_hour = int(conditional_cfg.get("end_hour", 22))
        if profile.access_category == "blacklist":
            return Decision(False, "Plaque blacklistée", "BLOCK_BLACKLIST")

        if profile.access_category == "conditional":
            if timestamp.hour < start_hour or timestamp.hour > end_hour:
                return Decision(False, "Accès conditionnel hors plage horaire", "BLOCK_CONDITIONAL_HOURS")
            return Decision(
                True,
                "Accès conditionnel validé",
                "ALLOW_CONDITIONAL_DAY",
                multiplier=float(category_multiplier.get("conditional", 1.2)),
            )

        if profile.access_category == "vip":
            return Decision(True, "Accès VIP prioritaire", "ALLOW_VIP", multiplier=float(category_multiplier.get("vip", 0.0)))

        if profile.access_category == "subscribed":
            return Decision(
                True,
                "Abonné valide",
                "ALLOW_SUBSCRIBED",
                multiplier=float(category_multiplier.get("subscribed", 0.7)),
            )

        return Decision(True, "Visiteur autorisé", "ALLOW_VISITOR", multiplier=float(category_multiplier.get("visitor", 1.0)))

    def explain_decision(self, decision: Decision, profile: VehicleProfile) -> str:
        return (
            f"Décision: {'AUTORISÉ' if decision.allowed else 'REFUSÉ'} | "
            f"Catégorie: {profile.access_category} | "
            f"Règle: {decision.applied_rule} | Motif: {decision.reason}"
        )
