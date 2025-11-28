"""
MCP Server API Routes
FastAPI endpoints for MCP server
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
import logging

from .server import get_mcp_server
from .handlers import handle_mcp_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP Server"])


@router.get(
    "/status",
    summary="MCP Server Status",
    description="Check if MCP server is running and get server information"
)
async def mcp_status():
    """
    Get MCP server status and information
    
    Returns:
        Server status and capabilities
    """
    try:
        mcp_server = get_mcp_server()
        info = await mcp_server.get_server_info()
        
        return {
            "status": "active",
            "message": "MCP server is running",
            "server": info
        }
    except Exception as e:
        logger.error(f"❌ MCP status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MCP server error: {str(e)}"
        )


@router.post(
    "/request",
    summary="MCP Request Handler",
    description="Send MCP protocol requests"
)
async def mcp_request(request_data: Dict[str, Any] = Body(...)):
    """
    Handle MCP protocol requests
    
    Args:
        request_data: MCP request payload with method and params
    
    Returns:
        MCP response
    
    Example:
        ```json
        {
            "method": "get_material_chunks",
            "params": {
                "material_id": "507f1f77bcf86cd799439011"
            }
        }
        ```
    """
    try:
        response = await handle_mcp_request(request_data)
        return response
    except Exception as e:
        logger.error(f"❌ MCP request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MCP request error: {str(e)}"
        )


@router.get(
    "/materials",
    summary="List All Materials",
    description="Get list of all available materials"
)
async def list_materials(
    skip: int = 0,
    limit: int = 50
):
    """
    List all available materials with pagination
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        List of materials
    """
    try:
        mcp_server = get_mcp_server()
        result = await mcp_server.list_materials(skip, limit)
        return result
    except Exception as e:
        logger.error(f"❌ Failed to list materials: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list materials: {str(e)}"
        )


@router.get(
    "/materials/{material_id}/chunks",
    summary="Get Material Chunks",
    description="Get all chunks for a specific material"
)
async def get_material_chunks(
    material_id: str,
    include_vectors: bool = False
):
    """
    Get chunks for a specific material
    
    Args:
        material_id: MongoDB document ID
        include_vectors: Include embedding vectors in response
    
    Returns:
        Material chunks with metadata
    """
    try:
        mcp_server = get_mcp_server()
        result = await mcp_server.get_material_chunks(material_id, include_vectors)
        
        if "error" in result:
            raise HTTPException(
                status_code=404,
                detail=result["error"]
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get material chunks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get material chunks: {str(e)}"
        )


@router.get(
    "/materials/{material_id}/context",
    summary="Get Material Context",
    description="Get material context formatted for LLM consumption"
)
async def get_material_context(
    material_id: str,
    max_chunks: Optional[int] = None
):
    """
    Get material context optimized for LLM
    
    Args:
        material_id: MongoDB document ID
        max_chunks: Optional limit on number of chunks
    
    Returns:
        Formatted context for LLM
    """
    try:
        mcp_server = get_mcp_server()
        result = await mcp_server.get_material_context(material_id, max_chunks)
        
        if "error" in result:
            raise HTTPException(
                status_code=404,
                detail=result["error"]
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get material context: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get material context: {str(e)}"
        )


@router.get(
    "/materials/ids",
    summary="Get All Material IDs",
    description="Get list of all material IDs with basic metadata"
)
async def get_material_ids():
    """
    Get all material IDs
    
    Returns:
        List of material IDs with metadata
    """
    try:
        mcp_server = get_mcp_server()
        result = await mcp_server.get_all_material_ids()
        
        return {
            "total": len(result),
            "materials": result
        }
    except Exception as e:
        logger.error(f"❌ Failed to get material IDs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get material IDs: {str(e)}"
        )