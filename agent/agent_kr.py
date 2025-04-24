import logging
from dataclasses import dataclass, field
from typing import List, Optional

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

logger = logging.getLogger("multi-agent")

load_dotenv()

common_instructions = (
    "당신은 드라이브스루 주문을 도와주는 친절한 AI입니다. "
    "항상 한국어로 대화하며, 예의를 갖춰 응답하세요."
)

@dataclass
class OrderItem:
    name: str
    size: Optional[str] = None
    special_requests: Optional[str] = None

@dataclass
class OrderData:
    items: List[OrderItem] = field(default_factory=list)
    current_item_index: int = 0 # Track which item we are asking details for

class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{common_instructions} "
                         "안녕하세요! 스타벅스 드라이브스루입니다. 주문 도와드릴까요? "
                         "주문하실 메뉴들을 말씀해주세요."
        )

    async def on_enter(self):
        # Generate the initial greeting without waiting for user input first
        await self.session.generate_reply(allow_interruptions=True)


    @function_tool
    async def order_started(
        self,
        context: RunContext[OrderData],
        menu_items: str, # Expecting comma-separated or similar list from user speech
    ):
        """
        Call this function when the user lists the menu items they want to order.
        `menu_items` should be the list of items mentioned by the user.
        """
        # Basic parsing, assuming comma-separated items or LLM provides a clean list string
        # A more robust approach might involve the LLM parsing the user's speech into a list directly.
        item_names = [item.strip() for item in menu_items.split(',') if item.strip()]
        if not item_names:
             return None, "죄송합니다, 주문하실 메뉴를 다시 말씀해주시겠어요?"

        context.userdata.items = [OrderItem(name=name) for name in item_names]
        context.userdata.current_item_index = 0
        order_agent = OrderAgent()
        logger.info(
            "Switching to order agent with items: %s", [item.name for item in context.userdata.items]
        )

        # Ask for details of the first item
        first_item_name = context.userdata.items[0].name
        reply = f"{', '.join(item_names)} 주문 확인했습니다. 첫 번째 메뉴인 '{first_item_name}'의 사이즈는 어떻게 하시겠어요?"
        return order_agent, reply

class OrderAgent(Agent):
    def __init__(self, *, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=f"{common_instructions} "
                         "이제 각 메뉴 항목에 대한 세부 정보를 수집합니다. "
                         "현재 처리 중인 메뉴 항목({self.get_current_item_name(context)})의 사이즈와 특별 요청 사항을 물어보세요. "
                         "사용자가 사이즈를 말하면 `add_item_details` 함수를 호출하여 정보를 저장하세요. "
                         "모든 메뉴 항목의 세부 정보가 수집되면, `confirm_full_order` 함수를 호출하여 전체 주문 내역을 요약하고 사용자에게 확인을 요청하세요. "
                         "사용자가 최종 확인하면 `order_confirmed`를, 취소하면 `order_cancelled`를 호출하세요. "
                         "대화는 한국어로 진행됩니다.",
            tts=None, # Inherit TTS from session
            chat_ctx=chat_ctx,
        )

    def get_current_item_name(self, context: RunContext[OrderData]) -> str:
        """Helper for instructions to know the current item."""
        idx = context.userdata.current_item_index
        if 0 <= idx < len(context.userdata.items):
            return context.userdata.items[idx].name
        return "알 수 없는 항목"

    # on_enter might not be needed if IntroAgent's reply already prompts for the first item's size

    @function_tool
    async def add_item_details(
        self,
        context: RunContext[OrderData],
        drink_size: str,
        special_requests: Optional[str] = None,
    ):
        """
        Call this function after the user specifies the size and optionally special requests
        for the current menu item being processed.
        """
        current_index = context.userdata.current_item_index
        if current_index >= len(context.userdata.items):
            logger.warning("Attempted to add details but no more items left or index out of bounds.")
            # Maybe try to recover or confirm order as is
            return await self.confirm_full_order(context)

        current_item = context.userdata.items[current_index]
        current_item.size = drink_size
        current_item.special_requests = special_requests
        logger.info(f"Added details for item '{current_item.name}': Size={drink_size}, Requests={special_requests}")

        context.userdata.current_item_index += 1

        if context.userdata.current_item_index < len(context.userdata.items):
            # Ask for details of the next item
            next_item = context.userdata.items[context.userdata.current_item_index]
            reply = (f"네, '{current_item.name}'은(는) {drink_size} 사이즈"
                     f"{f', 특별 요청: {special_requests}' if special_requests else ''} 맞으시죠? "
                     f"다음 메뉴인 '{next_item.name}'의 사이즈는 어떻게 하시겠어요?")
            return None, reply # Stay in OrderAgent, ask next question
        else:
            # All items processed, move to confirmation by calling the confirmation tool
            # Instruct the LLM to call confirm_full_order next.
            logger.info("All item details collected. Requesting final confirmation.")
            # It's better to have a dedicated tool for confirmation summary
            return await self.confirm_full_order(context)


    @function_tool
    async def confirm_full_order(self, context: RunContext[OrderData]):
        """
        Call this function when all item details have been collected to read back the
        full order to the user and ask for final confirmation.
        """
        if not context.userdata.items:
            return None, "주문 내역이 없습니다. 주문을 다시 시작해주세요."

        order_summary = "주문하신 내역 최종 확인해드리겠습니다:\n"
        for i, item in enumerate(context.userdata.items):
            order_summary += (f"{i+1}. 메뉴: {item.name}, "
                              f"사이즈: {item.size or '미지정'}, "
                              f"특별 요청: {item.special_requests or '없음'}\n")

        logger.info("Confirming full order: %s", context.userdata.items)
        reply = order_summary + "\n이대로 주문을 진행할까요?"
        # Stay in OrderAgent, wait for user's Yes/No to call order_confirmed/order_cancelled
        return None, reply


    @function_tool
    async def order_confirmed(self, context: RunContext[OrderData]):
        """Call this function if the user confirms the final order."""
        logger.info("Order confirmed by user. Final order: %s", context.userdata.items)
        self.session.interrupt() # Stop listening for further input immediately
        await self.session.generate_reply(
            # Use instructions for the final message, not a direct text reply
            instructions="주문이 완료되었습니다. 픽업대로 이동해주세요. 감사합니다.",
            allow_interruptions=False # Don't allow interruption for the final message
        )
        # Clean up the room after the final message is likely sent
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
        # Note: The agent might exit before the TTS finishes completely. Consider adding a small delay if needed.

    @function_tool
    async def order_cancelled(self, context: RunContext[OrderData]):
        """Call this function if the user cancels the order."""
        logger.info("Order cancelled by user. Order details: %s", context.userdata.items)
        self.session.interrupt()
        await self.session.generate_reply(
            instructions="주문이 취소되었습니다. 다음에 또 이용해주세요. 안녕히 계세요.",
            allow_interruptions=False
        )
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

# (Rest of the file remains the same: prewarm, entrypoint, __main__)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession[OrderData](
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        # stt=deepgram.STT(model="nova-3", language="korean"),
        stt=openai.STT(model="whisper-1", language="ko"),
        tts=openai.TTS(voice="alloy"),
        userdata=OrderData(), # Initialize with empty order data
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
        agent=IntroAgent(), # Start with the IntroAgent
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
