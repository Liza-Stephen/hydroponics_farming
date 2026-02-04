"""
LLM-based Natural Language Processor for Digital Twin Simulation
Uses Large Language Models to have conversations about hydroponics and simulation results
"""
import os
from typing import Optional, Dict, List
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_file = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_file, override=False)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

class LLMProcessor:
    """
    Processes natural language queries using Large Language Models
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize LLM processor with Ollama
        
        Args:
            model: Ollama model name (defaults to "llama3.2")
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package is required. Install with: pip install requests")
        
        self.provider = "ollama"
        self.conversation_history: List[Dict[str, str]] = []
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
    
    def _build_system_prompt(self, simulation_result: Optional[pd.DataFrame] = None) -> str:
        """
        Build system prompt with simulation context
        
        Args:
            simulation_result: Optional DataFrame with simulation results
        
        Returns:
            System prompt string
        """
        prompt = """You are a helpful hydroponics farming assistant with expertise in:
- Hydroponic system management and optimization
- pH, TDS (Total Dissolved Solids), temperature, and humidity control
- Equipment operation (pH reducers, nutrient adders, fans, humidifiers)
- Plant nutrition and growth optimization
- System troubleshooting and recommendations

You provide clear, accurate, and actionable advice based on:
1. General hydroponics knowledge
2. Current simulation state (if available)
3. Simulation history and trends (if available)

Key optimal ranges for hydroponic systems:
- pH: 5.5-6.5
- TDS: 800-1200 ppm
- Air Temperature: 20-25°C
- Air Humidity: 50-70%
- Water Level: 60-80%

When simulation data is available, use it to provide specific, contextual answers.
When no simulation data is available, provide general hydroponics guidance.

Be conversational, helpful, and concise. Use bullet points or numbered lists when appropriate for clarity."""
        
        if simulation_result is not None and len(simulation_result) > 0:
            final_state = simulation_result.iloc[-1]
            initial_state = simulation_result.iloc[0]
            
            prompt += f"""

CURRENT SIMULATION STATE:
- pH Level: {final_state.get('ph_level', 'N/A'):.2f} (Optimal: {final_state.get('is_ph_optimal', False)})
- TDS Level: {final_state.get('tds_level', 'N/A'):.0f} ppm (Optimal: {final_state.get('is_tds_optimal', False)})
- Air Temperature: {final_state.get('air_temperature', 'N/A'):.1f}°C (Optimal: {final_state.get('is_temp_optimal', False)})
- Air Humidity: {final_state.get('air_humidity', 'N/A'):.1f}% (Optimal: {final_state.get('is_humidity_optimal', False)})
- Water Temperature: {final_state.get('water_temperature', 'N/A'):.1f}°C
- Water Level: {final_state.get('water_level', 'N/A'):.1f}%
- Environment Optimal: {final_state.get('is_environment_optimal', False)}

SIMULATION HISTORY:
- Duration: {len(simulation_result)} time steps
- Initial pH: {initial_state.get('ph_level', 'N/A'):.2f} → Final pH: {final_state.get('ph_level', 'N/A'):.2f}
- Initial TDS: {initial_state.get('tds_level', 'N/A'):.0f} ppm → Final TDS: {final_state.get('tds_level', 'N/A'):.0f} ppm
- Initial Temperature: {initial_state.get('air_temperature', 'N/A'):.1f}°C → Final Temperature: {final_state.get('air_temperature', 'N/A'):.1f}°C
- Initial Humidity: {initial_state.get('air_humidity', 'N/A'):.1f}% → Final Humidity: {final_state.get('air_humidity', 'N/A'):.1f}%
- Optimal Time Steps: {simulation_result['is_environment_optimal'].sum() if 'is_environment_optimal' in simulation_result.columns else 'N/A'} out of {len(simulation_result)}"""
        
        return prompt
    
    def answer_question(
        self, 
        query: str, 
        simulation_result: Optional[pd.DataFrame] = None,
        system_context: Optional[str] = None,
        include_history: bool = True
    ) -> str:
        """
        Answer questions using LLM with simulation context
        
        Args:
            query: User's question
            simulation_result: Optional DataFrame with simulation results
            include_history: Whether to include conversation history
        
        Returns:
            LLM response string
        """
        system_prompt = self._build_system_prompt(simulation_result)
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if enabled
        if include_history and self.conversation_history:
            messages.extend(self.conversation_history[-10:])  # Keep last 10 exchanges
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        try:
            # Ollama API call
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 5000
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("message", {}).get("content", "")
            
            if not answer:
                raise ValueError("Empty response from Ollama. Make sure the model is installed and Ollama is running.")
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except requests.exceptions.ConnectionError:
            return (
                f"**Connection Error**: Cannot connect to Ollama at {self.ollama_url}\n\n"
                f"**Setup Instructions:**\n"
                f"1. Install Ollama from https://ollama.ai\n"
                f"2. Start Ollama: `ollama serve` (or it may start automatically)\n"
                f"3. Pull a model: `ollama pull {self.model}`\n"
                f"4. Verify Ollama is running: `curl http://localhost:11434/api/tags`\n\n"
                f"**Quick Start:**\n"
                f"```bash\n"
                f"# Install Ollama (macOS/Linux)\n"
                f"curl -fsSL https://ollama.ai/install.sh | sh\n\n"
                f"# Or download from https://ollama.ai for Windows\n\n"
                f"# Pull a model\n"
                f"ollama pull {self.model}\n"
                f"```"
            )
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("message", error_json.get("error", str(error_json)))
                except:
                    error_detail = e.response.text
                error_msg = f"{error_msg}\n\n**Server response:** {error_detail}"
            
            return (
                f"**HTTP Error {e.response.status_code if hasattr(e, 'response') else 'Unknown'}**: {error_msg}\n\n"
                f"**Troubleshooting:**\n"
                f"1. Make sure Ollama is running: `ollama serve`\n"
                f"2. Verify the model '{self.model}' is installed: `ollama list`\n"
                f"3. If the model is missing, pull it: `ollama pull {self.model}`\n"
                f"4. Check that Ollama is accessible at {self.ollama_url}"
            )
        except Exception as e:
            return (
                f"I encountered an error while processing your question.\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Troubleshooting:**\n"
                f"1. Make sure Ollama is installed and running\n"
                f"2. Verify the model '{self.model}' is installed: `ollama pull {self.model}`\n"
                f"3. Check that Ollama is accessible at {self.ollama_url}\n"
                f"4. Try restarting Ollama: `ollama serve`"
            )
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
