"""OpenAI function calling definitions for telecom agent tools."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """Definition of a callable tool for the LLM."""
    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    function: Callable[..., Any]
    requires_confirmation: bool = False


class ToolRegistry:
    """Registry of available tools for function calling."""
    
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        function: Callable[..., Any],
        requires_confirmation: bool = False,
    ) -> None:
        """Register a new tool."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            requires_confirmation=requires_confirmation,
        )
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_openai_tools(self) -> List[dict]:
        """Get tools in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in self._tools.values()
        ]
    
    def execute(self, name: str, arguments: dict) -> dict:
        """Execute a tool and return the result."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        
        try:
            result = tool.function(**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}


# Tool parameter schemas
CREATE_TICKET_PARAMS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Customer's full name",
        },
        "phone": {
            "type": "string",
            "description": "Customer's phone number (with country code preferred)",
        },
        "issue": {
            "type": "string",
            "description": "Description of the customer's issue or complaint",
        },
    },
    "required": ["name", "phone", "issue"],
}

RECOMMEND_BUNDLE_PARAMS = {
    "type": "object",
    "properties": {
        "budget_afn": {
            "type": "integer",
            "description": "Customer's budget in AFN (Afghani)",
        },
        "usage_gb": {
            "type": "integer",
            "description": "Expected monthly data usage in GB (optional)",
        },
    },
    "required": ["budget_afn"],
}

CHECK_BALANCE_PARAMS = {
    "type": "object",
    "properties": {
        "phone_number": {
            "type": "string",
            "description": "Customer's phone number to check balance for",
        },
    },
    "required": ["phone_number"],
}

CHECK_USAGE_PARAMS = {
    "type": "object",
    "properties": {
        "phone_number": {
            "type": "string",
            "description": "Customer's phone number",
        },
        "period_days": {
            "type": "integer",
            "description": "Number of days to get usage history for (default: 30)",
            "default": 30,
        },
    },
    "required": ["phone_number"],
}

CHECK_NETWORK_STATUS_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "Location/area to check network status for",
        },
    },
    "required": ["location"],
}


def create_default_registry() -> ToolRegistry:
    """Create a registry with all default telecom tools."""
    from .tools import create_ticket, recommend_bundle
    
    registry = ToolRegistry()
    
    # Core tools
    registry.register(
        name="create_ticket",
        description="Create a support ticket for a customer issue. Use when customer wants to file a complaint or report a problem.",
        parameters=CREATE_TICKET_PARAMS,
        function=create_ticket,
    )
    
    registry.register(
        name="recommend_bundle",
        description="Recommend a data/voice bundle based on customer's budget and usage needs.",
        parameters=RECOMMEND_BUNDLE_PARAMS,
        function=recommend_bundle,
    )
    
    # Simulated tools (for demo purposes)
    registry.register(
        name="check_balance",
        description="Check current account balance and active bundles for a phone number.",
        parameters=CHECK_BALANCE_PARAMS,
        function=_simulated_check_balance,
    )
    
    registry.register(
        name="check_usage",
        description="Get usage history (data, voice, SMS) for a customer's phone number.",
        parameters=CHECK_USAGE_PARAMS,
        function=_simulated_check_usage,
    )
    
    registry.register(
        name="check_network_status",
        description="Check network status and any ongoing issues in a specific area.",
        parameters=CHECK_NETWORK_STATUS_PARAMS,
        function=_simulated_check_network,
    )
    
    return registry


# Simulated tool implementations
def _simulated_check_balance(phone_number: str) -> dict:
    """Simulated balance check."""
    import random
    return {
        "phone_number": phone_number,
        "main_balance_afn": random.randint(50, 500),
        "data_remaining_gb": round(random.uniform(0.5, 5.0), 2),
        "voice_remaining_min": random.randint(10, 200),
        "active_bundles": [
            {"name": "Monthly 4GB data", "expires": "2024-02-15"},
        ],
        "last_recharge": "2024-01-20",
    }


def _simulated_check_usage(phone_number: str, period_days: int = 30) -> dict:
    """Simulated usage history."""
    import random
    return {
        "phone_number": phone_number,
        "period_days": period_days,
        "data_used_gb": round(random.uniform(3, 15), 2),
        "voice_used_min": random.randint(100, 500),
        "sms_sent": random.randint(20, 100),
        "average_daily_data_mb": random.randint(100, 500),
        "peak_usage_day": "Saturday",
    }


def _simulated_check_network(location: str) -> dict:
    """Simulated network status check."""
    import random
    statuses = ["excellent", "good", "fair"]
    return {
        "location": location,
        "status": random.choice(statuses),
        "4g_coverage": True,
        "3g_coverage": True,
        "active_incidents": [],
        "last_updated": "2024-01-25T10:30:00Z",
        "note": f"Network coverage in {location} is currently stable.",
    }


# Global registry
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = create_default_registry()
    return _registry


__all__ = [
    "ToolDefinition",
    "ToolRegistry", 
    "get_tool_registry",
    "create_default_registry",
]
