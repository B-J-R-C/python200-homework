import os
import json
import pandas as pd
from scipy.stats import pearsonr
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# smolagents imports
from smolagents import tool, ToolCallingAgent, CodeAgent, OpenAIServerModel

# --- Setup: Load keys and create a dummy CSV for testing ---
load_dotenv()
client = OpenAI()

os.makedirs("outputs", exist_ok=True)
df_dummy = pd.DataFrame({
    "avg_traffic_density": [10, 20, 30, 40, 50],
    "avg_speed_kmh": [25, 20, 15, 10, 5],
    "avg_heart_rate": [100, 110, 120, 130, 140],
    "duration_min": [15, 25, 35, 45, 55]
})
df_dummy.to_csv("bike_commute.csv", index=False)
# -----------------------------------------------------------


# ==========================================
# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# ==========================================

# Q1: Function, Schema, and Test Calls
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature in degrees Celsius to convert."
                }
            },
            "required": ["celsius"]
        }
    }
}

print("--- Q1: Celsius to Fahrenheit Tests ---")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# Q2: run_agent (Single Tool)
def get_current_time() -> str:
    return datetime.now().strftime("%I:%M %p")

get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time.",
        "parameters": {"type": "object", "properties": {}}
    }
}

def run_agent_single(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[get_current_time_schema]
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "get_current_time":
                result = get_current_time()
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        final_response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_response.choices[0].message.content
    return msg.content

"""
Q2 Prediction:
Will calling it trigger a tool call? No. The LLM knows that getting the current time won't help it convert temperature.
How many API calls? Just 1. The LLM will immediately respond with its internal knowledge without asking for tools.
"""
print("\n--- Q2: Single Tool Agent ---")
print("Response:", run_agent_single("Convert 100 degrees Celsius to Fahrenheit"))
# Was my prediction correct? Yes, the model answered it directly without invoking the time tool.


# Q3: Extended Multi-Tool run_agent
def run_agent_extended(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[get_current_time_schema, celsius_to_fahrenheit_schema]
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "get_current_time":
                result = get_current_time()
            elif tool_call.function.name == "celsius_to_fahrenheit":
                args = json.loads(tool_call.function.arguments)
                result = celsius_to_fahrenheit(args["celsius"])
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        final_response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final_response.choices[0].message.content
    return msg.content

print("\n--- Q3: Extended Multi-Tool Agent ---")
response_a = run_agent_extended("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Q3 Comment A: A tool WAS called. The model recognized it had a specific math tool to handle this conversion accurately and used it.

response_b = run_agent_extended("What is the boiling point of water in plain English?")
print("Response B:", response_b)
# Q3 Comment B: A tool WAS NOT called. The LLM already knows the boiling point of water from its training data, so it just answered directly without needing to calculate anything.


# ==========================================
# --- Lesson 03: Multi-Tool Agent (CsvManager) ---
# ==========================================

# Q4: Add compute_correlation
class CsvManager:
    def __init__(self):
        self.df = None

    def load_csv(self, filepath: str):
        try:
            self.df = pd.read_csv(filepath)
            return f"Successfully loaded {filepath}. Columns: {list(self.df.columns)}"
        except Exception as e:
            return f"Error loading CSV: {e}"

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return json.dumps({"error": "No CSV loaded."})
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return json.dumps({"error": f"Columns {col1} or {col2} not found."})
        
        # Drop NAs to prevent math errors
        valid_data = self.df[[col1, col2]].dropna()
        r, p = pearsonr(valid_data[col1], valid_data[col2])
        
        return json.dumps({
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4)
        })

csv_manager = CsvManager()

load_csv_schema = {
    "type": "function",
    "function": {
        "name": "load_csv",
        "description": "Load a CSV file to analyze.",
        "parameters": {
            "type": "object",
            "properties": {"filepath": {"type": "string"}},
            "required": ["filepath"]
        }
    }
}

compute_correlation_schema = {
    "type": "function",
    "function": {
        "name": "compute_correlation",
        "description": "Compute the Pearson correlation between two columns in the loaded CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {"type": "string", "description": "Name of the first column"},
                "col2": {"type": "string", "description": "Name of the second column"}
            },
            "required": ["col1", "col2"]
        }
    }
}

tools_schema = [load_csv_schema, compute_correlation_schema]
node_tools = {
    "load_csv": csv_manager.load_csv,
    "compute_correlation": csv_manager.compute_correlation
}


# Q5: Agent Cycle Setup
def run_agent_cycle(messages, query):
    messages.append({"role": "user", "content": query})
    for _ in range(5): # Limit to 5 tool rounds
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            tools=tools_schema
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg) # Add assistant's tool call request
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                if fn_name in node_tools:
                    res = node_tools[fn_name](**args)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(res)})
        else:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content
    return "Hit tool limit"

print("\n--- Q5: run_agent_cycle ---")
SYSTEM_PROMPT = "You are a helpful data assistant. Use your tools to answer questions."
cycle_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(cycle_messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print("Final Result:", result)


# Q6: Print Messages
print("\n--- Q6: Full Message History ---")
"""
Role Definitions in ReAct:
- 'system': The baseline instructions setting the agent's persona and rules.
- 'user': The human's prompt or question.
- 'assistant': The LLM's response (this can be regular text, or a request to call a tool).
- 'tool': The actual data/result returned by the python function we executed on behalf of the assistant.
"""
print(json.dumps(cycle_messages, indent=2, default=str))


# ==========================================
# --- Lesson 04: smolagents ---
# ==========================================

# Q7: smolagents @tool Decorator
@tool
def compute_correlation_tool(col1: str, col2: str) -> str:
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.
    Returns the correlation coefficient and p-value.
    
    Args:
        col1: The name of the first column.
        col2: The name of the second column.
    """
    return csv_manager.compute_correlation(col1, col2)

@tool
def load_csv_tool(filepath: str) -> str:
    """Loads a CSV file into the CsvManager.
    Args:
        filepath: The path to the CSV file.
    """
    return csv_manager.load_csv(filepath)

print("\n--- Q7: smolagents tool description ---")
print(compute_correlation_tool.description)
"""
Q7 Comment: 
smolagents automatically generated a massive JSON schema just by reading the python function's docstring and type hints! 
To produce a good description, smolagents strictly needs: 
1. Type hints for all parameters (e.g., col1: str).
2. A clear docstring that explains what the function does.
3. An 'Args:' section in the docstring detailing what each specific parameter means.
"""


# Q8: ToolCallingAgent vs CodeAgent
print("\n--- Q8: Agent Showdown ---")
model = OpenAIServerModel(model_id="gpt-4o-mini")
TOOLS = [load_csv_tool, compute_correlation_tool]

tool_agent = ToolCallingAgent(tools=TOOLS, model=model)
# We authorize matplotlib/pandas so CodeAgent can successfully write plot code
code_agent = CodeAgent(tools=TOOLS, model=model, additional_authorized_imports=["pandas", "matplotlib.pyplot", "matplotlib"])

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots. Save the plot to the outputs folder."

print("\n-> Running ToolCallingAgent:")
try:
    response_tool = tool_agent.run(prompt)
    print("Tool Agent Response:", response_tool)
except Exception as e:
    print("Tool Agent failed/errored:", str(e))

print("\n-> Running CodeAgent:")
try:
    response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
    print("Code Agent Response:", response_code)
except Exception as e:
    print("Code Agent failed/errored:", str(e))

"""
Q8 Comment:
What did each produce? 
- The ToolCallingAgent likely hallucinated, apologized, or failed completely because we didn't give it a specific 'plot_graph_tool'. It couldn't change the dot color.
- The CodeAgent succeeded! It actually wrote and executed raw Python code (using matplotlib/pandas) to load the CSV, plot the data with 'color="green"', and save the image. 
This reveals that ToolCallingAgents are rigid and only work if you hand-code a specific tool for every possible scenario. CodeAgents are highly flexible and can solve novel problems on the fly by writing their own scripts.
"""

# Q9: Final Reflection
"""
--- Q9: Conceptual Reflection ---
1. ToolCallingAgent Use Case:
A ToolCallingAgent is a better choice for high-security environments, like interacting with a company's payment processing API (e.g., issuing refunds via Stripe). The property that makes it a good fit is constraint: you want the agent strictly limited to pressing a specific, pre-approved "refund" button, rather than having the freedom to write custom scripts that manipulate the database.

2. CodeAgent Risk:
A major risk of using a CodeAgent is security and unintended consequences. Because it literally generates and executes Python code on the host machine, a poorly prompted agent (or a malicious prompt injection) could accidentally write code that deletes local files, causes an infinite loop that crashes the server, or exposes private environment variables.
"""