from __future__ import annotations

from typing import Optional, AsyncGenerator

from google.adk.agents import LlmAgent, LoopAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
# NOTE: PlanReActPlanner was removed - it interferes with tool execution
# and causes the agent to generate planning text instead of actually calling tools
from google.genai import types

from agent.memory_service import TeacherProfile
from agent.tools import WorksheetToolset


def build_root_agent(
    *,
    toolset: WorksheetToolset,
    teacher_profile: TeacherProfile,
    model: str = "gemini-2.5-flash",
) -> LoopAgent:
    """
    Construct the ADK root agent powered by Gemini and our custom toolset.

    The core worksheet orchestration is handled by an `LlmAgent`, which is then
    wrapped in a small `LoopAgent` that can retry the workflow if artifacts are
    not produced on the first attempt. This gives the LLM a couple of chances to
    recover from tool-calling mistakes before we fall back to runtime guards.
    """

    instruction = f"""
You are a California math curriculum assistant that designs printable practice
worksheets for grades 1–8. Follow this workflow strictly:

1. Interpret the teacher's request and recall prior constraints from memory.

2. Call `generate_plan` to obtain a structured worksheet plan before making
   any formatting decisions. Never fabricate a plan manually.

3. Convert the returned problem templates into full problems and worked
   solutions. Respect the grade-level number ranges and scaffolds described in
   the plan. Validate every numeric answer mentally or with scratch work to
   prevent hallucinations.

4. Prepare a `problems` list of JSON objects with these keys:
   - `prompt`: The problem text
   - `answer`: The answer
   - `solution_steps`: List of steps (optional)
   - `diagram_data`: Data for diagrams (REQUIRED for visual problems)
   
   DIAGRAM DATA EXAMPLES (must match values in your problem!):
   - Bar graphs: {{"diagram_data": {{"Amy": 12, "Ben": 8, "Cathy": 15}}}}
   - Triangles: {{"diagram_data": {{"base": 8, "height": 6}}}}
   - Rectangles: {{"diagram_data": {{"width": 10, "length": 5}}}}
   - Circles: {{"diagram_data": {{"radius": 7}}}}
   - Fractions: {{"diagram_data": {{"numerator": 3, "denominator": 8}}}}
   
   CRITICAL FORMATTING RULES:
   - DO NOT use LaTeX notation (no \\(, \\), \\frac, etc.)
   - Write fractions as "3/4" or in words
   - Use Unicode: π, ≈, ², ³
   - Write clearly for students reading on paper

5. Call the artifact building tools using their exact snake_case names:
   - `build_docx`
   - `build_pdf`
   Pass the same problem set to both.

6. Summarize the rationale for the produced worksheet AFTER the artifacts exist.

Teacher defaults:
- Preferred grade: {teacher_profile.preferred_grade}
- Default length: {teacher_profile.default_problem_count} problems
- Include teacher notes: {teacher_profile.include_teacher_notes}
- Include solution guidance: {teacher_profile.include_solution_guidance}

Always align number ranges and skills with the grade progression rules supplied
by the internal toolchain. When in doubt, generate a smaller plan and explain
the limitation to the teacher.
""".strip()

    worksheet_agent = LlmAgent(
        name="worksheet_orchestrator",
        description="Plans and builds math worksheets by calling specialized tools.",
        instruction=instruction,
        model=model,
        # NOTE: PlanReActPlanner was removed - it interferes with direct tool execution
        # The LLM can call tools directly without a planner layer
        tools=[toolset],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,  # Increased for complex worksheets
        ),
    )

    class ArtifactCheckAgent(BaseAgent):
        """
        Simple verification agent used inside a LoopAgent.

        It doesn't talk to the LLM; it just inspects the shared WorksheetToolset
        to see whether DOCX and PDF artifacts have been produced. If so, it
        escalates to signal the loop to stop.
        """

        def __init__(self, name: str):
            super().__init__(name=name)
            self._iteration_count = 0

        async def _run_async_impl(
            self,
            ctx: InvocationContext,
        ) -> AsyncGenerator[Event, None]:
            self._iteration_count += 1
            artifacts = toolset.latest_artifacts()
            have_docx = "docx" in artifacts
            have_pdf = "pdf" in artifacts
            should_stop = have_docx and have_pdf

            print(f"[DEBUG] ArtifactCheckAgent iteration {self._iteration_count}: "
                  f"docx={have_docx}, pdf={have_pdf}, escalate={should_stop}")

            if should_stop:
                # Yield an event to signal the loop to stop via escalation
                print("[DEBUG] ArtifactCheckAgent: Both artifacts present, escalating to stop loop")
                yield Event(
                    author=self.name,
                    actions=EventActions(escalate=True),
                )
            else:
                # Continue the loop but log what's missing
                missing = []
                if not have_docx:
                    missing.append("docx")
                if not have_pdf:
                    missing.append("pdf")
                print(f"[DEBUG] ArtifactCheckAgent: Missing artifacts: {missing}")
                yield Event(
                    author=self.name,
                    actions=EventActions(escalate=False),
                )

    # Give the planner up to 2 attempts to produce valid artifacts via the tools.
    # Reduced from 3 to fail faster when stuck.
    loop_agent = LoopAgent(
        name="worksheet_loop",
        sub_agents=[worksheet_agent, ArtifactCheckAgent(name="artifact_checker")],
        max_iterations=2,
    )

    return loop_agent

