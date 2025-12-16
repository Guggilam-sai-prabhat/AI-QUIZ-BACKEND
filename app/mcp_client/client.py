"""
MCP Client
HTTP client for communicating with MCP Server via RPC endpoint
"""
import logging
from typing import Dict, Any, List, Optional
import httpx

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Base exception for MCP client errors"""
    pass


class MCPClient:
    """
    Client for MCP Server RPC communication
    
    Communicates with the MCP Server via HTTP POST to /mcp/request endpoint
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 30.0):
        """
        Initialize MCP Client
        
        Args:
            base_url: Base URL of the MCP server (without /mcp/request)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.rpc_endpoint = f"{self.base_url}/mcp/request"
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        logger.info(f"🔌 MCP Client initialized: {self.rpc_endpoint}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    async def _call_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make RPC call to MCP Server
        
        Args:
            method: MCP method name
            params: Optional method parameters
        
        Returns:
            Response data from MCP server
        
        Raises:
            MCPClientError: If request fails or returns error
        """
        payload = {
            "method": method,
            "params": params or {}
        }
        
        logger.debug(f"📤 MCP RPC: {method}")
        
        try:
            response = await self.client.post(self.rpc_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Check for MCP-level errors
            if "error" in data:
                raise MCPClientError(f"MCP Error: {data['error']}")
            
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ MCP HTTP error: {e.response.status_code}")
            raise MCPClientError(f"MCP HTTP error: {e.response.text}")
        
        except httpx.RequestError as e:
            logger.error(f"❌ MCP request failed: {e}")
            raise MCPClientError(f"MCP request failed: {e}")
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server information"""
        return await self._call_rpc("get_server_info")
    
    async def get_all_material_ids(self) -> List[Dict[str, str]]:
        """Get all available material IDs with metadata"""
        result = await self._call_rpc("get_all_material_ids")
        return result if isinstance(result, list) else []
    
    async def get_material_chunks(
        self,
        material_id: str,
        include_vectors: bool = False
    ) -> Dict[str, Any]:
        """
        Get all chunks for a specific material
        
        Args:
            material_id: MongoDB document ID
            include_vectors: Whether to include embedding vectors
        
        Returns:
            Material metadata and chunks
        """
        return await self._call_rpc(
            "get_material_chunks",
            {
                "material_id": material_id,
                "include_vectors": include_vectors
            }
        )
    
    async def get_material_context(
        self,
        material_id: str,
        max_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get material context in MCP-compatible format
        
        Args:
            material_id: MongoDB document ID
            max_chunks: Optional limit on number of chunks
        
        Returns:
            Formatted context for LLM consumption
        """
        params = {"material_id": material_id}
        if max_chunks is not None:
            params["max_chunks"] = max_chunks
        
        return await self._call_rpc("get_material_context", params)
    
    async def list_materials(
        self,
        skip: int = 0,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        List materials with pagination
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            Paginated list of materials
        """
        return await self._call_rpc(
            "list_materials",
            {
                "skip": skip,
                "limit": limit
            }
        )


# Singleton instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client(base_url: str = "http://localhost:8000") -> MCPClient:
    """
    Get or create the global MCP client instance
    
    Args:
        base_url: Base URL of MCP server
    
    Returns:
        MCPClient instance
    """
    global _mcp_client
    
    if _mcp_client is None:
        _mcp_client = MCPClient(base_url=base_url)
    
    return _mcp_client


async def close_mcp_client():
    """Close the global MCP client"""
    global _mcp_client
    if _mcp_client:
        await _mcp_client.close()
        _mcp_client = None