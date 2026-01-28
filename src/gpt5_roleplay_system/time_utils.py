from datetime import datetime
from zoneinfo import ZoneInfo

def get_pacific_time(ts: float | None = None) -> datetime:
    """Get a datetime object in Pacific Time."""
    if ts is None:
        dt = datetime.now(ZoneInfo("UTC"))
    else:
        dt = datetime.fromtimestamp(ts, ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("America/Los_Angeles"))

def format_pacific_time(ts: float | None = None) -> str:
    """Format a timestamp into a human-readable Pacific Time string."""
    dt = get_pacific_time(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_pacific_timestamp(ts: float | None = None) -> int:
    """Get a numerical timestamp shifted to Pacific Time digits."""
    dt = get_pacific_time(ts)
    # Create a UTC datetime using the Pacific Time digits
    utc_dt = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, tzinfo=ZoneInfo("UTC"))
    return int(utc_dt.timestamp())
