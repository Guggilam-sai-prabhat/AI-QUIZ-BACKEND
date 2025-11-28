"""
MCP Server Core
Handles MCP protocol communication and context provision
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from app.db.mongodb import get_collection
from app.services.qdrant_service import get_qdrant_service
from bson import ObjectId

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Model Context Protocol Server
    Provides access to document materials and their chunks
    """
    
    def __init__(self):
        """Initialize MCP Server"""
        self.server_name = "MaterialContextProvider"
        self.version = "1.0.0"
        self.started_at = datetime.now(timezone.utc)
        logger.info(f"🚀 MCP Server initialized: {self.server_name} v{self.version}")
    
    async def get_server_info(self) -> Dict[str, Any]:
        """
        Get MCP server information
        
        Returns:
            Server metadata and status
        """
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        
        return {
            "server_name": self.server_name,
            "version": self.version,
            "protocol": "MCP/1.0",
            "status": "active",
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": round(uptime, 2),
            "capabilities": {
                "get_material_chunks": True,
                "get_all_material_ids": True,
                "list_materials": True,
                "get_material_context": True
            }
        }
    
    async def get_all_material_ids(self) -> List[Dict[str, str]]:
        """
        Get all available material IDs with metadata
        
        Returns:
            List of materials with id, filename, and upload date
        """
        try:
            collection = get_collection("documents")
            materials = []
            
            cursor = collection.find(
                {},
                {"_id": 1, "filename": 1, "uploaded_at": 1, "page_count": 1, "chunk_count": 1}
            ).sort("uploaded_at", -1)
            
            async for doc in cursor:
                materials.append({
                    "material_id": str(doc["_id"]),
                    "filename": doc.get("filename", "unknown"),
                    "uploaded_at": doc.get("uploaded_at", datetime.now(timezone.utc)).isoformat(),
                    "page_count": doc.get("page_count", 0),
                    "chunk_count": doc.get("chunk_count", 0)
                })
            
            logger.info(f"📋 Retrieved {len(materials)} material IDs")
            return materials
            
        except Exception as e:
            logger.error(f"❌ Failed to get material IDs: {e}")
            return []
    
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
        try:
            # Validate material_id format
            if not ObjectId.is_valid(material_id):
                logger.warning(f"⚠️ Invalid material_id format: {material_id}")
                return {
                    "material_id": material_id,
                    "error": "Invalid material ID format"
                }
            
            # Get material metadata from MongoDB
            collection = get_collection("documents")
            material = await collection.find_one({"_id": ObjectId(material_id)})
            
            if not material:
                logger.warning(f"⚠️ Material not found: {material_id}")
                return {
                    "material_id": material_id,
                    "error": "Material not found"
                }
            
            # Get chunks from Qdrant
            qdrant_service = get_qdrant_service()
            chunks = await qdrant_service.get_material_chunks(
                material_id=material_id,
                with_vectors=include_vectors
            )
            
            # Build response
            response = {
                "material_id": material_id,
                "filename": material.get("filename"),
                "page_count": material.get("page_count", 0),
                "chunk_count": len(chunks),
                "uploaded_at": material.get("uploaded_at", datetime.now(timezone.utc)).isoformat(),
                "chunks": chunks
            }
            
            logger.info(
                f"✅ Retrieved {len(chunks)} chunks for material {material_id}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get material chunks: {e}")
            return {
                "material_id": material_id,
                "error": str(e)
            }
    
    async def get_material_context(
        self,
        material_id: str,
        max_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get material context in MCP-compatible format
        Optimized for LLM context windows
        
        Args:
            material_id: MongoDB document ID
            max_chunks: Optional limit on number of chunks to return
        
        Returns:
            Formatted context for LLM consumption
        """
        try:
            chunks_data = await self.get_material_chunks(material_id, include_vectors=False)
            
            if "error" in chunks_data:
                return chunks_data
            
            chunks = chunks_data.get("chunks", [])
            
            # Apply chunk limit if specified
            if max_chunks and max_chunks > 0:
                chunks = chunks[:max_chunks]
            
            # Format context
            context_text = []
            for chunk in chunks:
                chunk_header = f"[Chunk {chunk['chunk_id'] + 1}]"
                context_text.append(f"{chunk_header}\n{chunk['text']}\n")
            
            combined_context = "\n".join(context_text)
            
            return {
                "material_id": material_id,
                "filename": chunks_data.get("filename"),
                "total_chunks": chunks_data.get("chunk_count"),
                "chunks_included": len(chunks),
                "context": combined_context,
                "metadata": {
                    "page_count": chunks_data.get("page_count"),
                    "uploaded_at": chunks_data.get("uploaded_at")
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get material context: {e}")
            return {
                "material_id": material_id,
                "error": str(e)
            }
    
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
        try:
            collection = get_collection("documents")
            
            # Get total count
            total = await collection.count_documents({})
            
            # Get paginated materials
            materials = []
            cursor = collection.find().sort("uploaded_at", -1).skip(skip).limit(limit)
            
            async for doc in cursor:
                materials.append({
                    "material_id": str(doc["_id"]),
                    "filename": doc.get("filename"),
                    "file_size": doc.get("file_size"),
                    "page_count": doc.get("page_count", 0),
                    "chunk_count": doc.get("chunk_count", 0),
                    "uploaded_at": doc.get("uploaded_at", datetime.now(timezone.utc)).isoformat()
                })
            
            return {
                "total": total,
                "skip": skip,
                "limit": limit,
                "count": len(materials),
                "materials": materials
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to list materials: {e}")
            return {
                "error": str(e)
            }


# Singleton instance
_mcp_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """
    Get or create the global MCP server instance
    
    Returns:
        MCPServer instance
    """
    global _mcp_server
    
    if _mcp_server is None:
        _mcp_server = MCPServer()
    
    return _mcp_server