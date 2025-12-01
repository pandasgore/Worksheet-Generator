
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time
from typing import List

from dotenv import load_dotenv

# Safety limits to prevent infinite loops
MAX_EVENTS = 100  # Maximum number of events to process
TIMEOUT_SECONDS = 120  # Maximum time for worksheet generation (increased for complex topics)

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    __package__ = "agent"

load_dotenv()

from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from .memory_service import WorksheetMemoryService
from .planner import build_root_agent
from .tools import WorksheetToolset


class WorksheetAgentRuntime:
    """Bootstraps the ADK application, runner, and persistent memory."""

    def __init__(self, *, artifacts_dir: Path | None = None):

        base_dir = Path(__file__).resolve().parent
        artifact_dir = artifacts_dir or (base_dir / "../artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        self.memory_service = WorksheetMemoryService(base_dir / "memory.json")
        self.toolset = WorksheetToolset(artifact_dir=artifact_dir)
        root_agent = build_root_agent(
            toolset=self.toolset,
            teacher_profile=self.memory_service.teacher_profile,
        )
        self.app = App(name="worksheet_agent", root_agent=root_agent)
        self.runner = Runner(
            app=self.app,
            session_service=InMemorySessionService(),
            memory_service=self.memory_service,
        )

    async def _ensure_session(self, user_id: str, session_id: str) -> None:
        service = self.runner.session_service
        session = await service.get_session(
            app_name=self.app.name,
            user_id=user_id,
            session_id=session_id,
        )
        if not session:
            await service.create_session(
                app_name=self.app.name,
                user_id=user_id,
                session_id=session_id,
            )

    def run(
        self,
        prompt: str,
        *,
        user_id: str = "teacher",
        session_id: str = "default",
    ) -> List[str]:
        """Execute the ADK agent and return the textual replies."""

        print(f"[DEBUG] Runtime starting run for prompt: {prompt}")
        # CRITICAL: Clear artifacts from previous runs to avoid phantom success
        self.toolset.clear_artifacts()
        
        asyncio.run(self._ensure_session(user_id, session_id))

        events = self.runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            ),
        )

        replies: List[str] = []
        final_response_text = ""
        
        # Safety counters to prevent infinite loops
        event_count = 0
        start_time = time.time()
        
        # Iterate events to drive execution with safety limits
        for event in events:
            event_count += 1
            elapsed = time.time() - start_time
            
            # Check safety limits
            if event_count > MAX_EVENTS:
                print(f"[WARNING] Exceeded max events ({MAX_EVENTS}). Breaking loop.")
                break
            
            if elapsed > TIMEOUT_SECONDS:
                print(f"[WARNING] Exceeded timeout ({TIMEOUT_SECONDS}s). Breaking loop.")
                break
            
            # Early exit if we already have both artifacts
            artifacts = self.toolset.latest_artifacts()
            if "docx" in artifacts and "pdf" in artifacts:
                print(f"[DEBUG] Both artifacts created after {event_count} events. Exiting early.")
                break
            
            # Debug log event details to identify where things get stuck
            try:
                event_type = type(event).__name__
                author = getattr(event, 'author', 'unknown')
                
                # Check for various event attributes
                has_content = hasattr(event, 'content') and event.content is not None
                has_function_call = hasattr(event, 'function_call') and event.function_call is not None
                has_function_response = hasattr(event, 'function_response') and event.function_response is not None
                has_actions = hasattr(event, 'actions') and event.actions is not None
                is_final = hasattr(event, 'is_final_response') and callable(event.is_final_response) and event.is_final_response()
                
                print(f"[DEBUG] Event #{event_count}: type={event_type}, author={author}, "
                      f"has_content={has_content}, has_func_call={has_function_call}, "
                      f"has_func_resp={has_function_response}, is_final={is_final} "
                      f"(elapsed: {elapsed:.1f}s)")
                
                # Log function call details
                if has_function_call:
                    func_call = event.function_call
                    func_name = getattr(func_call, 'name', str(func_call))
                    print(f"[DEBUG] -> Function call: {func_name}")
                
                # Log function response details
                if has_function_response:
                    func_resp = event.function_response
                    resp_name = getattr(func_resp, 'name', 'unknown')
                    print(f"[DEBUG] -> Function response for: {resp_name}")
                
                # Log actions (escalate, etc.)
                if has_actions:
                    actions = event.actions
                    escalate = getattr(actions, 'escalate', None)
                    if escalate is not None:
                        print(f"[DEBUG] -> Actions: escalate={escalate}")
                        
            except Exception as log_err:
                print(f"[DEBUG] Event #{event_count}: (logging error: {log_err})")
            
            if hasattr(event, "is_final_response") and callable(event.is_final_response) and event.is_final_response():
                text = self._extract_text_from_event(event)
                if text:
                    replies.append(text)
                    final_response_text = text
                    print(f"[DEBUG] -> Final response text captured (length: {len(text)})")
        
        print(f"[DEBUG] Event loop finished: {event_count} events in {time.time() - start_time:.1f}s")

        print(f"[DEBUG] Run complete. Final response len: {len(final_response_text)}")
        print(f"[DEBUG] Current artifacts: {self.toolset.latest_artifacts().keys()}")

        # Check if artifacts were created
        artifacts = self.toolset.latest_artifacts()
        print(f"[DEBUG] Replies before fallback check: {replies}")
        
        # Warn if no artifacts were created
        if not artifacts:
            print("[WARNING] No artifacts were created! The agent did not call build_docx/build_pdf.")
            if not replies:
                replies.append(
                    "Unable to generate worksheet files. The agent did not complete the build process. "
                    "Please try again with a different topic or contact support."
                )
        
        # If the model never produced any natural-language reply but artifacts exist,
        # synthesize a short summary so the UI isn't empty.
        elif not replies and artifacts:
            print("[DEBUG] Synthesizing summary message...")
            plan = self.toolset.latest_plan()
            if plan:
                difficulty = plan.difficulty_plan.get("difficulty_level", "mixed")
                message = (
                    f"Both the DOCX and PDF worksheets have been generated for Grade "
                    f"{plan.grade} on \"{plan.concept_text}\" at {difficulty} difficulty "
                    f"with {plan.num_problems} problems. Use the links below to download."
                )
            else:
                message = (
                    "The worksheet files have been generated successfully. "
                    "Use the links below to download the DOCX and PDF."
                )
            replies.append(message)

        self.memory_service.append_request(
            {"prompt": prompt, "session_id": session_id, "user_id": user_id}
        )
        return replies

    def latest_artifacts(self) -> dict[str, dict]:
        return self.toolset.latest_artifacts()

    @staticmethod
    def _extract_text_from_event(event) -> str:
        if not getattr(event, "content", None):
            return ""
        parts = getattr(event.content, "parts", None)
        if not parts:
            return ""
        texts = [part.text for part in parts if getattr(part, "text", None)]
        return "\n".join(filter(None, (text.strip() for text in texts)))


if __name__ == "__main__":
    runtime = WorksheetAgentRuntime()
