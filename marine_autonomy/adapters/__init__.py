"""Adapters package — bridges between external data formats and internal contracts."""
from .ais_adapter import parse_ais_contact, contacts_to_perception, ego_state_from_nmea

__all__ = [
    "parse_ais_contact",
    "contacts_to_perception",
    "ego_state_from_nmea",
]
