import asyncio
import calendar
import os
import random
import sys
from datetime import date

from agent_framework import Agent, tool
from agent_framework.observability import configure_otel_providers
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatClient, OpenAIChatCompletionClient
from agent_framework.orchestrations import MagenticBuilder
from azure.ai.agentserver.agentframework import from_agent_framework
from azure.ai.agentserver.agentframework.models import agent_framework_input_converters
from azure.ai.agentserver.agentframework.persistence import AgentSessionRepository, CheckpointRepository
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv(override=False)

# Export traces to AI Toolkit so orchestration and spans are visible locally.
configure_otel_providers(vs_code_extension_port=4317, enable_sensitive_data=True)


_original_transform_input = agent_framework_input_converters.transform_input


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return " ".join(parts).strip()
    return ""


def _latest_user_only_transform_input(input_item):
    """Keep only the most recent user message to avoid replaying completed Magentic tasks."""
    if isinstance(input_item, list):
        latest_text = ""
        for item in input_item:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            if role is not None and role != "user":
                continue

            # Explicit message: {"type":"message","role":"user","content":...}
            if item.get("type") == "message" and "content" in item:
                candidate = _extract_text(item.get("content"))
                if candidate:
                    latest_text = candidate
                continue

            # Implicit user message: {"content":...}
            if "content" in item and role is None and "type" not in item:
                candidate = _extract_text(item.get("content"))
                if candidate:
                    latest_text = candidate

        if latest_text:
            return _original_transform_input(latest_text)
    return _original_transform_input(input_item)


# Magentic orchestrations expect one task at a time. The playground includes full
# conversation history in each request, so we normalize to the latest user task.
agent_framework_input_converters.transform_input = _latest_user_only_transform_input


class StatelessAgentSessionRepository(AgentSessionRepository):
    """Disable conversation-level session persistence so each request starts fresh."""

    async def get(self, conversation_id: str | None):
        return None

    async def set(self, conversation_id: str | None, session) -> None:
        return


class NoopCheckpointRepository(CheckpointRepository):
    """Disable checkpoint resume behavior so Magentic runs are always fresh."""

    async def get_or_create(self, conversation_id: str | None):
        return None


def build_client() -> FoundryChatClient | OpenAIChatClient | OpenAIChatCompletionClient:
    foundry_endpoint = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
    foundry_model = os.getenv("FOUNDRY_MODEL_DEPLOYMENT_NAME")

    if foundry_endpoint and foundry_model:
        return FoundryChatClient(
            project_endpoint=foundry_endpoint,
            model=foundry_model,
            credential=DefaultAzureCredential(),
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        # Use Chat Completions API for OpenAI fallback to avoid duplicate item-id issues
        # observed in some Magentic reset/progress flows with Responses API.
        return OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=openai_api_key)

    raise RuntimeError(
        "Missing model configuration. Set FOUNDRY_PROJECT_ENDPOINT and "
        "FOUNDRY_MODEL_DEPLOYMENT_NAME, or set OPENAI_API_KEY for local fallback."
    )


@tool(approval_mode="never_require")
def get_random_number() -> str:
    return str(random.randint(0, 100))


@tool(approval_mode="never_require")
def get_today_date() -> str:
    return date.today().isoformat()


@tool(approval_mode="never_require")
def get_random_date_next_month() -> str:
    today = date.today()
    year = today.year + (1 if today.month == 12 else 0)
    month = 1 if today.month == 12 else today.month + 1
    last_day = calendar.monthrange(year, month)[1]
    day = random.randint(1, last_day)
    return date(year, month, day).isoformat()


def build_magentic_workflow():
    client = build_client()

    agent_1_number = Agent(
        client=client,
        name="Agent1Number",
        description="Specialist for generating a random number between 0 and 100.",
        instructions=(
            "You are Agent 1 number specialist. "
            "Use get_random_number when number generation is requested. "
            "Return concise Dutch output and include the generated value."
        ),
        tools=[get_random_number],
    )

    agent_2_date = Agent(
        client=client,
        name="Agent2Date",
        description="Specialist for date retrieval and random next-month date generation.",
        instructions=(
            "You are Agent 2 date specialist. "
            "Use get_today_date for requests about today's date. "
            "Use get_random_date_next_month for requests about a random date next month. "
            "Return concise Dutch output with the date value."
        ),
        tools=[get_today_date, get_random_date_next_month],
    )

    agent_3_topic = Agent(
        client=client,
        name="Agent3Topic",
        description="Specialist for writing information or a story about a topic.",
        instructions=(
            "You are Agent 3 topic specialist. "
            "Write clear, engaging Dutch content about the requested topic. "
            "When a date is provided by another agent, incorporate that exact date in the story."
        ),
    )

    manager_agent = Agent(
        client=client,
        name="MagenticManager",
        description="Orchestrator that coordinates the three specialists dynamically.",
        instructions=(
            "You coordinate Agent1Number, Agent2Date, and Agent3Topic to solve the user's request. "
            "Only involve agents that are needed. "
            "For tasks like 'Geef me de datum van vandaag en laat er een goed verhaal over schrijven', "
            "first delegate to Agent2Date, then pass the resulting date to Agent3Topic. "
            "Synthesize a single final Dutch answer for the user."
        ),
    )

    return MagenticBuilder(
        participants=[agent_1_number, agent_2_date, agent_3_topic],
        manager_agent=manager_agent,
        intermediate_outputs=False,
        max_round_count=10,
        max_stall_count=2,
        max_reset_count=1,
    ).build()


app = from_agent_framework(
    build_magentic_workflow,
    session_repository=StatelessAgentSessionRepository(),
    checkpoint_repository=NoopCheckpointRepository(),
)


if __name__ == "__main__":
    script_name = os.path.basename(sys.argv[0])
    if os.getenv("AGENTDEV_ENABLED") != "1":
        raise SystemExit(
            "Start this hosted workflow with agentdev so Agent Inspector can use /agentdev/version.\n"
            f"Example: agentdev run {script_name} --port 8088"
        )
    asyncio.run(app.run_async())
