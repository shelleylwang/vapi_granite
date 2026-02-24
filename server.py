# What Vapi talks to. Connects Vapi → Granite → Calendar.

from flask import Flask, request, Response
import json
import requests
from calendar_mcp import TOOLS, execute_tool

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Where our Granite model is running (Ollama)
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "client-salon"  # Our fine-tuned model in Ollama

# System prompt for the model
SYSTEM_PROMPT = """You are a friendly receptionist for Client's Hair Salon.

Your job is to help customers:
- Book new appointments
- Reschedule existing appointments  
- Cancel appointments
- Answer questions about services and pricing

Services and prices:
- Haircut (💇): $45, 45 minutes
- Color (🎨): $85, 90 minutes
- Cut and Color (💇🎨): $120, 2 hours
- Treatment (💆): $55, 1 hour
- Bridal Styling (👰): $200, 3 hours

When you need to check or modify the calendar, use the available tools.
Be warm, friendly, and conversational. Use the customer's name when you know it.

If someone got the date wrong, offer to search nearby dates.
Always confirm changes before making them.
"""

# Tool definitions (what Granite knows it can call)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_appointments",
            "description": "Find a client's existing appointments",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {"type": "string", "description": "The client's name"},
                    "date_hint": {"type": "string", "description": "Specific date to check"},
                    "date_range_start": {"type": "string", "description": "Start of date range"},
                    "date_range_end": {"type": "string", "description": "End of date range"},
                },
                "required": ["client_name"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": "Check available appointment times on a date",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "The date to check"},
                    "service_type": {
                        "type": "string",
                        "enum": ["haircut", "color", "cut_and_color", "treatment", "bridal"],
                        "description": "Type of service",
                    },
                },
                "required": ["date", "service_type"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Move an appointment to a new date/time",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_date": {"type": "string"},
                    "new_date": {"type": "string"},
                    "new_time": {"type": "string"},
                    "client_name": {"type": "string"},
                    "new_service": {"type": "string"},
                },
                "required": ["original_date", "new_date", "new_time", "client_name"],
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "book_appointment",
            "description": "Create a new appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "client_name": {"type": "string"},
                    "service_type": {"type": "string"},
                    "phone": {"type": "string"},
                },
                "required": ["date", "time", "client_name", "service_type"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment", 
            "description": "Cancel an existing appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["client_name", "date"],
            }
        }
    },
]


# ============================================================
# THE MAIN ENDPOINT VAPI CALLS
# ============================================================

@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    """
    This is the OpenAI-compatible endpoint that Vapi calls.
    
    Flow:
    1. Vapi sends us the conversation
    2. We send it to Granite
    3. If Granite wants to use a tool, we execute it and send result back
    4. We stream Granite's response back to Vapi
    """
    
    data = request.json
    messages = data.get("messages", [])
    
    # Add system prompt if not present
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    # Call Granite via Ollama
    response = call_granite(messages)
    
    # Check if Granite wants to use a tool
    if response.get("tool_calls"):
        # Execute the tool
        tool_results = []
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            print(f"🔧 Executing tool: {tool_name}")
            print(f"   Arguments: {arguments}")
            
            result = execute_tool(tool_name, arguments)
            
            print(f"   Result: {result}")
            
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            })
        
        # Add assistant's tool call and our results to the conversation
        messages.append({
            "role": "assistant",
            "content": response.get("content", ""),
            "tool_calls": response["tool_calls"],
        })
        messages.extend(tool_results)
        
        # Call Granite again with the tool results
        response = call_granite(messages)
    
    # Stream the response back to Vapi
    return stream_response(response)


def call_granite(messages: list) -> dict:
    """Send messages to Granite and get response."""
    
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "stream": False,  # We'll handle streaming ourselves
        },
        timeout=30,
    )
    
    result = response.json()
    return result["choices"][0]["message"]


def stream_response(message: dict) -> Response:
    """Stream the response in SSE format that Vapi expects."""
    
    def generate():
        content = message.get("content", "")
        
        # Send content in chunks (Vapi expects SSE format)
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send done signal
        done_chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk", 
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    print("🚀 Starting Client's Salon Voice Agent Server...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Ollama: {OLLAMA_URL}")
    app.run(host="0.0.0.0", port=5000, debug=True)