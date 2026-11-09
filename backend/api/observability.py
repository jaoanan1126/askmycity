"""
Observability with Langfuse - Track LLM calls, costs, and performance
Using the OpenTelemetry-based Langfuse SDK (v3+)

Usage pattern (explicit span objects, safe to use from worker threads):

    trace = observability.create_trace("ask_question", session_id=conv_id)
    observability.log_retrieval(trace, "chromadb_search", query, docs)
    observability.log_generation(trace, "claude", model, prompt, answer, usage)
    observability.end_trace(trace, output=answer)

Every method is a no-op when Langfuse is disabled or `trace` is None.
"""
from langfuse import get_client, propagate_attributes
import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

logger = logging.getLogger(__name__)


class ObservabilityService:
    """Langfuse integration for LLM observability using OpenTelemetry"""

    def __init__(self):
        """Initialize Langfuse client"""
        self.enabled = False
        self.langfuse = None
        try:
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")

            if not public_key or not secret_key:
                logger.warning("⚠️  Langfuse keys not set - observability disabled")
                return

            # Get the Langfuse singleton client
            self.langfuse = get_client()

            # Test connection
            if self.langfuse.auth_check():
                self.enabled = True
                logger.info("✓ Langfuse observability initialized")
            else:
                logger.warning("⚠️  Langfuse auth check failed - observability disabled")

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

    def create_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a root span (one trace per request).

        Returns the span object (use `.trace_id` for the trace ID), or None
        when observability is disabled or the span could not be created.
        """
        if not self.enabled:
            return None
        try:
            with propagate_attributes(session_id=session_id, trace_name=name):
                return self.langfuse.start_observation(
                    name=name,
                    as_type="span",
                    input=metadata,
                    metadata={**(metadata or {}), "session_id": session_id},
                )
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None

    def log_retrieval(
        self,
        trace,
        name: str,
        input_query: str,
        output_documents: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a retrieval step (e.g. a ChromaDB search) under `trace`."""
        if trace is None:
            return
        try:
            span = trace.start_observation(
                name=name,
                as_type="retriever",
                input={"query": input_query},
                output={"documents": output_documents},
                metadata=metadata or {},
            )
            span.end()
        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}")

    def log_generation(
        self,
        trace,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an LLM generation under `trace`."""
        if trace is None:
            return
        try:
            generation = trace.start_observation(
                name=name,
                as_type="generation",
                model=model,
                input=input_text,
                output=output_text,
                usage_details=usage or {},
                metadata=metadata or {},
            )
            generation.end()
        except Exception as e:
            logger.error(f"Failed to log generation: {e}")

    def log_score(self, trace, name: str, value: float, comment: str = "") -> None:
        """Attach a score to the whole trace."""
        if trace is None:
            return
        try:
            trace.score_trace(name=name, value=value, comment=comment)
        except Exception as e:
            logger.error(f"Failed to log score: {e}")

    def end_trace(self, trace, output: Any = None, error: Optional[str] = None) -> None:
        """Close the root span, recording its output or error."""
        if trace is None:
            return
        try:
            if error:
                trace.update(level="ERROR", status_message=error)
            elif output is not None:
                trace.update(output=output)
            trace.end()
        except Exception as e:
            logger.error(f"Failed to end trace: {e}")

    def flush(self):
        """Flush pending events to Langfuse"""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
                logger.debug("✓ Langfuse events flushed")
            except Exception as e:
                logger.error(f"Failed to flush Langfuse events: {e}")


# Global observability instance
observability = ObservabilityService()
