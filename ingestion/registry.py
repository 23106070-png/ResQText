from ingestion.gdacs import GdacsAdapter
from ingestion.reliefweb import ReliefWebAdapter
from ingestion.sachet import SachetAdapter


def default_adapters() -> dict:
    return {a.name: a for a in (SachetAdapter(), GdacsAdapter(), ReliefWebAdapter())}  # India first
