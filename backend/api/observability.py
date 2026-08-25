"""
Observability with Langfuse - Track LLM calls, costs, and performance
Using OpenTelemetry-based Langfuse SDK v3
"""
from langfuse import get_client
import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

logger = logging.getLogger(__name__)

class ObservabilityService:
    """Langfuse integration for LLM observability using OpenTelemetry"""
    
    def __init__(self):
        """Initialize Langfuse client"""
        try:
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            
            if not public_key or not secret_key:
                logger.warning("⚠️  Langfuse keys not set - observability disabled")
                self.enabled = False
                self.langfuse = None
                return
            
            # Get the Langfuse singleton client
            self.langfuse = get_client()
            self.enabled = True
            
            # Test connection
            if self.langfuse.auth_check():
                logger.info("✓ Langfuse observability initialized")
            else:
                logger.warning("⚠️  Langfuse auth check failed")
                self.enabled = False
            
        except Exception as e:
            logger.warning(f"⚠️  Langfuse initialization failed: {e}")
            self.enabled = False
            self.langfuse = None
    
    @contextmanager
    def trace_request(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager to create a Langfuse trace (root span)
        
        Usage:
            with observability.trace_request("ask_question", session_id="123") as trace:
                # Your code here
                pass
        """
        if not self.enabled:
            yield None
            return
        
        try:
            # Create root span with Langfuse trace attributes
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name=name
            ) as trace:
                # Set trace-level attributes
                if session_id:
                    trace.update_trace(session_id=session_id)
                if user_id:
                    trace.update_trace(user_id=user_id)
                if metadata:
                    trace.update_trace(metadata=metadata)
                
                yield trace
        except Exception as e:
            logger.error(f"Failed to create trace: {e}", exc_info=True)
            yield None
    
    @contextmanager
    def log_retrieval(
        self,
        name: str,
        input_query: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for logging retrieval operations (ChromaDB searches)
        
        Usage:
            with observability.log_retrieval("chromadb_search", query) as retrieval:
                # Perform search
                results = search(query)
                retrieval.update(output={"documents": results})
        """
        if not self.enabled:
            yield None
            return
        
        try:
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                input={"query": input_query},
                metadata=metadata or {}
            ) as span:
                yield span
        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}", exc_info=True)
            yield None
    
    @contextmanager
    def log_generation(
        self,
        name: str,
        model: str,
        input_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for logging LLM generations
        
        Usage:
            with observability.log_generation("claude", "claude-3", prompt) as gen:
                response = claude.call(prompt)
                gen.update(
                    output=response.text,
                    usage={"input": response.usage.input_tokens, ...}
                )
        """
        if not self.enabled:
            yield None
            return
        
        try:
            with self.langfuse.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                input=input_text,
                metadata=metadata or {}
            ) as generation:
                yield generation
        except Exception as e:
            logger.error(f"Failed to log generation: {e}", exc_info=True)
            yield None
    
    def flush(self):
        """Flush pending events to Langfuse"""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
                logger.debug("✓ Langfuse events flushed")
            except Exception as e:
                logger.error(f"Failed to flush Langfuse events: {e}")
    
    # Backward compatibility methods (v2 API style)
    def create_trace(self, name: str, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Backward compatibility: Create a trace (returns a dummy object with an id)"""
        class DummyTrace:
            def __init__(self):
                self.id = None
        return DummyTrace()
    
    def log_retrieval(self, trace_id: Optional[str] = None, name: str = "", input_text: str = "", output: Any = None, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Backward compatibility: Log retrieval (no-op for now)"""
        pass
    
    def log_generation(self, trace_id: Optional[str] = None, name: str = "", model: str = "", input_text: str = "", output_text: str = "", usage: Optional[Dict] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Backward compatibility: Log generation (no-op for now)"""
        pass
    
    def log_score(self, trace_id: Optional[str] = None, name: str = "", value: float = 0, comment: str = "", **kwargs):
        """Backward compatibility: Log score (no-op for now)"""
        pass

# Global observability instance
observability = ObservabilityService()