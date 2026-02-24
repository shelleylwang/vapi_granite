# The MCP server that gives the model calendar access to manage appointments.

from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import os

# ============================================================
# EMOJI MAPPING FOR SERVICES
# ============================================================

SERVICE_EMOJIS = {
    "haircut": "💇",
    "color": "🎨",
    "cut_and_color": "💇🎨",
    "treatment": "💆",
    "bridal": "👰",
}

SERVICE_DURATIONS = {  # in minutes
    "haircut": 45,
    "color": 90,
    "cut_and_color": 120,
    "treatment": 60,
    "bridal": 180,
}

# ============================================================
# GOOGLE CALENDAR SETUP
# ============================================================

def get_calendar_service():
    """Connect to Google Calendar API."""
    
    # Get these from Google Cloud Console
    creds = Credentials.from_authorized_user_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=creds)

# ============================================================
# TOOL: SEARCH APPOINTMENTS
# ============================================================

def search_appointments(client_name: str, date_hint: str = None, 
                        date_range_start: str = None, date_range_end: str = None):
    """
    Find appointments for a client.
    
    The model calls this when it needs to look up existing appointments.
    Handles fuzzy dates like "April 8" or "next Tuesday".
    """
    
    service = get_calendar_service()
    
    # Parse date range
    if date_hint:
        # Single date: search that day
        target_date = parse_fuzzy_date(date_hint)
        start = target_date.replace(hour=0, minute=0)
        end = target_date.replace(hour=23, minute=59)
    elif date_range_start and date_range_end:
        # Date range: search between dates
        start = parse_fuzzy_date(date_range_start)
        end = parse_fuzzy_date(date_range_end)
    else:
        # No date: search next 30 days
        start = datetime.now()
        end = start + timedelta(days=30)
    
    # Search Google Calendar
    events = service.events().list(
        calendarId="primary",
        timeMin=start.isoformat() + "Z",
        timeMax=end.isoformat() + "Z",
        q=client_name,  # search by client name
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    
    results = []
    for event in events.get("items", []):
        # Parse the emoji to get service type
        title = event.get("summary", "")
        service_type = emoji_to_service(title)
        
        results.append({
            "id": event["id"],
            "client_name": extract_client_name(title),
            "date": event["start"].get("dateTime", event["start"].get("date")),
            "service": service_type,
            "title": title,
        })
    
    if not results:
        return {"found": False, "message": f"No appointments found for {client_name}"}
    
    return {"found": True, "appointments": results}


# ============================================================
# TOOL: GET AVAILABLE SLOTS
# ============================================================

def get_available_slots(date: str, service_type: str):
    """
    Find open time slots on a given date.
    
    The model calls this when booking or rescheduling.
    """
    
    service = get_calendar_service()
    target_date = parse_fuzzy_date(date)
    duration = SERVICE_DURATIONS.get(service_type, 60)
    
    # Business hours: 9am - 6pm
    business_start = target_date.replace(hour=9, minute=0)
    business_end = target_date.replace(hour=18, minute=0)
    
    # Get existing events
    events = service.events().list(
        calendarId="primary",
        timeMin=business_start.isoformat() + "Z",
        timeMax=business_end.isoformat() + "Z",
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    
    # Find gaps
    busy_times = []
    for event in events.get("items", []):
        start = datetime.fromisoformat(event["start"]["dateTime"].replace("Z", ""))
        end = datetime.fromisoformat(event["end"]["dateTime"].replace("Z", ""))
        busy_times.append((start, end))
    
    # Find available slots
    available = []
    current = business_start
    
    for busy_start, busy_end in sorted(busy_times):
        if current + timedelta(minutes=duration) <= busy_start:
            available.append(current.strftime("%I:%M %p"))
        current = max(current, busy_end)
    
    # Check end of day
    if current + timedelta(minutes=duration) <= business_end:
        available.append(current.strftime("%I:%M %p"))
    
    return {
        "date": target_date.strftime("%B %d, %Y"),
        "service": service_type,
        "duration_minutes": duration,
        "available_times": available,
    }


# ============================================================
# TOOL: RESCHEDULE APPOINTMENT
# ============================================================

def reschedule_appointment(original_date: str, new_date: str, new_time: str,
                           client_name: str, new_service: str = None):
    """
    Move an appointment to a new date/time, optionally changing service type.
    """
    
    service = get_calendar_service()
    
    # Find the original appointment
    original = search_appointments(client_name, date_hint=original_date)
    
    if not original.get("found"):
        return {"success": False, "error": "Could not find original appointment"}
    
    appointment = original["appointments"][0]
    old_event_id = appointment["id"]
    old_service = appointment["service"]
    
    # Use new service or keep old one
    service_type = new_service or old_service
    emoji = SERVICE_EMOJIS.get(service_type, "📅")
    duration = SERVICE_DURATIONS.get(service_type, 60)
    
    # Parse new date/time
    new_start = parse_fuzzy_date(f"{new_date} {new_time}")
    new_end = new_start + timedelta(minutes=duration)
    
    # Delete old event
    service.events().delete(
        calendarId="primary",
        eventId=old_event_id,
    ).execute()
    
    # Create new event
    new_event = {
        "summary": f"{emoji} {client_name}",
        "start": {"dateTime": new_start.isoformat(), "timeZone": "America/New_York"},
        "end": {"dateTime": new_end.isoformat(), "timeZone": "America/New_York"},
    }
    
    created = service.events().insert(
        calendarId="primary",
        body=new_event,
    ).execute()
    
    return {
        "success": True,
        "old_date": original_date,
        "new_date": new_start.strftime("%B %d, %Y"),
        "new_time": new_start.strftime("%I:%M %p"),
        "service": service_type,
        "client": client_name,
    }


# ============================================================
# TOOL: BOOK NEW APPOINTMENT
# ============================================================

def book_appointment(date: str, time: str, client_name: str, 
                     service_type: str, phone: str = None):
    """
    Create a new appointment.
    """
    
    service = get_calendar_service()
    
    emoji = SERVICE_EMOJIS.get(service_type, "📅")
    duration = SERVICE_DURATIONS.get(service_type, 60)
    
    start = parse_fuzzy_date(f"{date} {time}")
    end = start + timedelta(minutes=duration)
    
    event = {
        "summary": f"{emoji} {client_name}",
        "description": f"Phone: {phone}" if phone else "",
        "start": {"dateTime": start.isoformat(), "timeZone": "America/New_York"},
        "end": {"dateTime": end.isoformat(), "timeZone": "America/New_York"},
    }
    
    created = service.events().insert(
        calendarId="primary",
        body=event,
    ).execute()
    
    return {
        "success": True,
        "date": start.strftime("%B %d, %Y"),
        "time": start.strftime("%I:%M %p"),
        "service": service_type,
        "client": client_name,
    }


# ============================================================
# TOOL: CANCEL APPOINTMENT
# ============================================================

def cancel_appointment(client_name: str, date: str):
    """
    Cancel an existing appointment.
    """
    
    service = get_calendar_service()
    
    # Find the appointment
    result = search_appointments(client_name, date_hint=date)
    
    if not result.get("found"):
        return {"success": False, "error": "Could not find appointment"}
    
    appointment = result["appointments"][0]
    
    # Delete it
    service.events().delete(
        calendarId="primary",
        eventId=appointment["id"],
    ).execute()
    
    return {
        "success": True,
        "cancelled_date": date,
        "client": client_name,
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_fuzzy_date(text: str) -> datetime:
    """
    Parse human-friendly dates like 'April 8', 'next Tuesday', 'tomorrow'.
    In production, use a library like dateparser.
    """
    import dateparser
    
    parsed = dateparser.parse(text)
    if parsed:
        return parsed
    
    # Fallback
    return datetime.now()


def emoji_to_service(title: str) -> str:
    """Extract service type from calendar event title."""
    
    for service, emoji in SERVICE_EMOJIS.items():
        if emoji in title:
            return service
    return "unknown"


def extract_client_name(title: str) -> str:
    """Remove emoji from title to get client name."""
    
    for emoji in SERVICE_EMOJIS.values():
        title = title.replace(emoji, "")
    return title.strip()


# ============================================================
# MCP TOOL REGISTRY
# ============================================================

TOOLS = {
    "search_appointments": {
        "function": search_appointments,
        "description": "Search for a client's appointments",
        "parameters": {
            "client_name": "The client's name to search for",
            "date_hint": "A specific date to check (optional)",
            "date_range_start": "Start of date range (optional)",
            "date_range_end": "End of date range (optional)",
        }
    },
    "get_available_slots": {
        "function": get_available_slots,
        "description": "Get available appointment times on a given date",
        "parameters": {
            "date": "The date to check",
            "service_type": "Type of service: haircut, color, cut_and_color, treatment, bridal",
        }
    },
    "reschedule_appointment": {
        "function": reschedule_appointment,
        "description": "Move an appointment to a new date/time",
        "parameters": {
            "original_date": "The current appointment date",
            "new_date": "The desired new date",
            "new_time": "The desired new time",
            "client_name": "The client's name",
            "new_service": "New service type if changing (optional)",
        }
    },
    "book_appointment": {
        "function": book_appointment,
        "description": "Book a new appointment",
        "parameters": {
            "date": "The appointment date",
            "time": "The appointment time",
            "client_name": "The client's name",
            "service_type": "Type of service",
            "phone": "Client's phone number (optional)",
        }
    },
    "cancel_appointment": {
        "function": cancel_appointment,
        "description": "Cancel an existing appointment",
        "parameters": {
            "client_name": "The client's name",
            "date": "The appointment date to cancel",
        }
    },
}


def execute_tool(tool_name: str, arguments: dict):
    """Run a tool and return the result."""
    
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}
    
    tool = TOOLS[tool_name]
    return tool["function"](**arguments)