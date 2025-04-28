import logging

from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from livekit import api, rtc
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
from livekit.agents.voice.avatar import DataStreamAudioReceiver

import pandas as pd

from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero, elevenlabs

logger = logging.getLogger("multi-agent")
 
def looks_like_order(text: str) -> bool:
    order_keywords = ["아메리카노", "카페라떼", "프라푸치노", "커피", "에스프레소", "차", "주문", "디카페인", "물"]
    return any(keyword in text for keyword in order_keywords)

load_dotenv()

common_instructions = (
    "당신은 드라이브스루 주문을 도와주는 친절한 AI입니다. "
    "항상 한국어로 대화하며, 예의를 갖춰 응답하세요."
)

@dataclass
class OrderData:
    menu_items: Optional[str] = None
    drink_size: Optional[str] = None
    special_requests: Optional[str] = None
    # payment_method: Optional[str] = None  # Removed

class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{common_instructions} "
                         "안녕하세요! 스타벅스 드라이브스루입니다. 주문 도와드릴까요?"
                         "주문과 관련된 이야기만 응답하세요. "
                         "예: 메뉴, 사이즈, 요청사항 등. "
                         "그 외의 질문은 '죄송합니다, 주문과 관련된 것만 말씀드릴 수 있어요.' 라고 대답하세요."

        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def order_started(
        self,
        context: RunContext[OrderData],
        menu_items: str,
    ):
        if not looks_like_order(menu_items):
            return None, "죄송합니다, 주문과 관련된 이야기만 도와드릴 수 있어요."

        context.userdata.menu_items = menu_items
        order_agent = OrderAgent()
        logger.info(
            "switching to order agent with menu items: %s", context.userdata.menu_items
        )
        return order_agent, f"{menu_items} 주문 받았습니다. 음료 사이즈를 알려주세요."


class OrderAgent(Agent):
    def __init__(self, *, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=f"{common_instructions} "
                         "이제 주문 정보를 수집합니다. 음료 사이즈와 특별 요청을 차례로 물어보고 확인하세요. "
                         "대화는 한국어로 진행됩니다.",
            tts=None,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def order_details(
        self,
        context: RunContext[OrderData],
        drink_size: str,
        special_requests: Optional[str] = None,
        # payment_method: Optional[str] = None,  # Removed
    ):
        context.userdata.drink_size = drink_size
        context.userdata.special_requests = special_requests
        # context.userdata.payment_method = payment_method  # Removed

        logger.info("Collected order details: %s", context.userdata)
        return None, (
            f"주문 세부 정보입니다:\n"
            f"메뉴: {context.userdata.menu_items}\n"
            f"사이즈: {drink_size}\n"
            f"특별 요청: {special_requests or '없음'}\n"
            # f"결제 방법: {payment_method or '미정'}\n"  # Removed
            "이대로 주문을 진행할까요?"
        )

    @function_tool
    async def order_confirmed(self, context: RunContext[OrderData]):
        self.session.interrupt()
        await self.session.generate_reply(
            instructions="주문이 완료되었습니다. 픽업대로 이동해주세요. 감사합니다.",
            allow_interruptions=False
        )
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

    # @function_tool
    # async def order_cancelled(self, context: RunContext[OrderData]):
    #     self.session.interrupt()
    #     await self.session.generate_reply(
    #         instructions="주문이 취소되었습니다. 다음에 또 이용해주세요. 안녕히 계세요.",
    #         allow_interruptions=False
    #     )
    #     job_ctx = get_job_context()
    #     await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession[OrderData](
        vad=ctx.proc.userdata["vad"],
        # llm=openai.LLM(model="gpt-4o-mini"),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.1),
        # stt=deepgram.STT(model="nova-3", language="korean"),
        stt=openai.STT(model="whisper-1", language="ko"),
        # tts=openai.TTS(voice="alloy"),
        tts=elevenlabs.TTS(),
        userdata=OrderData(),
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
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
