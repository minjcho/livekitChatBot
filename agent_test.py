import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero

# uncomment to enable Krisp BVC noise cancellation, currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("multi-agent")

load_dotenv()

common_instructions = (
    "Your name is Alex. You are a person who helps users make reservations via voice."
    "You are curious and friendly, with a sense of humor."
)

@dataclass
class ReservationData:
    # Shared data that's used by the reservation agent.
    # This structure is passed as a parameter to function calls.
    name: Optional[str] = None
    language: Optional[str] = "English"
    reservation_type: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    party_size: Optional[int] = None
    special_requests: Optional[str] = None

class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{common_instructions} Your goal is to gather a few pieces of "
            "information from the user to personalize the interaction."
            "You should ask the user for their name and preferred language."
            "Start the conversation with a short introduction.",
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def information_gathered(
        self,
        context: RunContext[ReservationData],
        name: str,
        language: str,
    ):
        """Called when the user has provided the information needed to make the reservation personalized.

        Args:
            name: The name of the user
            language: The language of the user
        """

        context.userdata.name = name
        context.userdata.language = language

        reservation_agent = ReservationAgent(name, language)
        logger.info(
            "switching to the reservation agent with the provided user data: %s", context.userdata
        )
        return reservation_agent, "Let's start making your reservation!"

class ReservationAgent(Agent):
    def __init__(self, name: str, language: str, *, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=f"{common_instructions} You are now helping the user make a reservation. "
            "Ask for the type of reservation (e.g., restaurant, hotel, event), date, time, party size, and any special requests. "
            "Confirm all details with the user before finalizing the reservation. "
            "Be interactive and clarify any missing information. "
            "Do not end on a statement where the user is not expected to respond. "
            "When interrupted, ask if the user would like to continue or end. "
            f"The user's name is {name}, and they want to speak in {language}.",
            llm=openai.realtime.RealtimeModel(voice="echo"),
            tts=None,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def reservation_details(
        self,
        context: RunContext[ReservationData],
        reservation_type: str,
        date: str,
        time: str,
        party_size: int,
        special_requests: Optional[str] = None,
    ):
        """Called when the user has provided all reservation details.

        Args:
            reservation_type: Type of reservation (restaurant, hotel, etc.)
            date: Date of reservation
            time: Time of reservation
            party_size: Number of people
            special_requests: Any special requests
        """
        context.userdata.reservation_type = reservation_type
        context.userdata.date = date
        context.userdata.time = time
        context.userdata.party_size = party_size
        context.userdata.special_requests = special_requests

        logger.info("Reservation details collected: %s", context.userdata)
        return None, (
            f"Thank you! Here are your reservation details:\n"
            f"Type: {reservation_type}\n"
            f"Date: {date}\n"
            f"Time: {time}\n"
            f"Party Size: {party_size}\n"
            f"Special Requests: {special_requests or 'None'}\n"
            "Would you like to confirm this reservation?"
        )

    @function_tool
    async def reservation_confirmed(self, context: RunContext[ReservationData]):
        """Called when the user confirms the reservation."""
        self.session.interrupt()
        await self.session.generate_reply(
            instructions=f"Confirm the reservation for {context.userdata.name} and say goodbye.",
            allow_interruptions=False
        )
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

    @function_tool
    async def reservation_cancelled(self, context: RunContext[ReservationData]):
        """Called when the user cancels the reservation."""
        self.session.interrupt()
        await self.session.generate_reply(
            instructions=f"Let {context.userdata.name} know the reservation was cancelled and say goodbye.",
            allow_interruptions=False
        )
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession[ReservationData](
        vad=ctx.proc.userdata["vad"],
        # llm=openai.LLM(model="gpt-4.1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        # stt=deepgram.STT(model="nova-3"),
        stt=openai.STT(model="whisper-1"),
        tts=openai.TTS(voice="echo"),
        userdata=ReservationData(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=IntroAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))