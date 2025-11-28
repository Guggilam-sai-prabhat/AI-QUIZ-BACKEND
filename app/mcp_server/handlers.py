"""
MCP Request Handlers
Processes MCP protocol requests
"""
import logging
from typing import Dict, Any
from .server import get_mcp_server

logger = logging.getLogger(__name__)


async def handle_mcp_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle incoming MCP requests
    
    Args:
        request_data: MCP request payload
    
    Returns:
        MCP response payload
    """
    mcp_server = get_mcp_server()
    
    method = request_data.get("method")
    params = request_data.get("params", {})
    
    logger.info(f"🔄 MCP Request: {method}")
    
    # Route to appropriate handler
    if method == "get_server_info":
        return await mcp_server.get_server_info()
    
    elif method == "get_all_material_ids":
        return await mcp_server.get_all_material_ids()
    
    elif method == "get_material_chunks":
        material_id = params.get("material_id")
        include_vectors = params.get("include_vectors", False)
        
        if not material_id:
            return {"error": "material_id parameter is required"}
        
        return await mcp_server.get_material_chunks(material_id, include_vectors)
    
    elif method == "get_material_context":
        material_id = params.get("material_id")
        max_chunks = params.get("max_chunks")
        
        if not material_id:
            return {"error": "material_id parameter is required"}
        
        return await mcp_server.get_material_context(material_id, max_chunks)
    
    elif method == "list_materials":
        skip = params.get("skip", 0)
        limit = params.get("limit", 50)
        
        return await mcp_server.list_materials(skip, limit)
    
    else:
        logger.warning(f"⚠️ Unknown MCP method: {method}")
        return {
            "error": f"Unknown method: {method}",
            "available_methods": [
                "get_server_info",
                "get_all_material_ids",
                "get_material_chunks",
                "get_material_context",
                "list_materials"
            ]
        }