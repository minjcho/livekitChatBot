"""
Starbucks Drive‑Thru multi‑agent voicebot built on LiveKit Agents SDK.
Inspired by the original restaurant example.
"""
import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

# ---------------------------------------------------------------------------
# logging & env
# ---------------------------------------------------------------------------
logger = logging.getLogger("starbucks-drive-thru")
logger.setLevel(logging.INFO)
load_dotenv()

# ---------------------------------------------------------------------------
# Voice IDs (replace with your ElevenLabs / TTS voice IDs as needed)
# ---------------------------------------------------------------------------
voices = {
    "greeter": "794f9389-aac1-45b6-b726-9d9369183238",   # lane greeter
    "order": "156fb8d2-335b-4950-9cb3-a2d33befec77",     # barista order‑taker
    "payment": "6f84f4b8-58a2-430c-8c79-688dad597532",   # payment agent
    "pickup": "39b376fc-488e-4d0c-8b37-e00b72059fdd",    # pickup notifier
}

# ---------------------------------------------------------------------------
# Dataclass holding conversation state
# ---------------------------------------------------------------------------
@dataclass
class DriveThruData:
    customer_name: Optional[str] = None
    car_model: Optional[str] = None
    license_plate: Optional[str] = None

    order: Optional[list[dict]] = None  # list of {item,size,options}

    pay_method: Optional[str] = None  # "카드", "사전 결제" 등
    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    # internal agent bookkeeping
    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    # helper for LLM context
    def summarize(self) -> str:
        data = {
            "name": self.customer_name or "unknown",
            "car_model": self.car_model or "unknown",
            "license_plate": self.license_plate or "unknown",
            "order": self.order or "unknown",
            "pay_method": self.pay_method or "unknown",
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        return yaml.dump(data, allow_unicode=True)

RunContext_T = RunContext[DriveThruData]

# ---------------------------------------------------------------------------
# Common function‑tools
# ---------------------------------------------------------------------------
@function_tool()
async def update_name(name: Annotated[str, Field(description="고객 이름")], context: RunContext_T) -> str:
    """고객이 이름을 알려줄 때 호출"""
    context.userdata.customer_name = name
    return f"이름을 {name}(으)로 저장했습니다."

@function_tool()
async def update_car_model(model: Annotated[str, Field(description="차종 (예: 'Kia Sorento')")], context: RunContext_T) -> str:
    context.userdata.car_model = model
    return f"차량 모델을 {model}(으)로 저장했습니다."

@function_tool()
async def update_license_plate(plate: Annotated[str, Field(description="차량 번호")], context: RunContext_T) -> str:
    context.userdata.license_plate = plate
    return f"차량 번호를 {plate}(으)로 저장했습니다."

@function_tool()
async def add_item(
    item: Annotated[str, Field(description="메뉴 이름, 예: 'Caffè Latte'")],
    size: Annotated[str, Field(description="사이즈 'Short'|'Tall'|'Grande'|'Venti'")],
    options: Annotated[list[str], Field(description="추가 옵션 목록, 예: ['Decaf','Oat milk']")],
    context: RunContext_T,
) -> str:
    """고객 주문에 아이템 추가"""
    data = context.userdata
    data.order = data.order or []
    data.order.append({"item": item, "size": size, "options": options})
    return f"{size} {item} 추가 완료."

@function_tool()
async def update_pay_method(method: Annotated[str, Field(description="결제 방식, 예: '카드' or '사전 결제'")], context: RunContext_T) -> str:
    context.userdata.pay_method = method
    return f"결제 방식을 {method}(으)로 설정했습니다."

# ---------------------------------------------------------------------------
# Base Agent class with shared helpers
# ---------------------------------------------------------------------------
class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        userdata: DriveThruData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # Bring previous agent chat for continuity
        if isinstance(userdata.prev_agent, Agent):
            truncated = userdata.prev_agent.chat_ctx.copy(exclude_instructions=True, exclude_function_call=False).truncate(max_items=6)
            existing = {m.id for m in chat_ctx.items}
            chat_ctx.items.extend([m for m in truncated.items if m.id not in existing])

        # System message with updated summary
        chat_ctx.add_message(
            role="system",
            content=f"You are {agent_name}. Current drive‑thru data:\n{userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T):
        userdata = context.userdata
        next_agent = userdata.agents[name]
        userdata.prev_agent = context.session.current_agent
        return next_agent, f"{name} 로 이동합니다."

# ---------------------------------------------------------------------------
# Agent implementations
# ---------------------------------------------------------------------------
class LaneGreeter(BaseAgent):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "당신은 스타벅스 드라이브‑스루의 LaneGreeter 입니다. "
                "차가 센서 범위에 들어오면 밝게 인사하고, 오늘의 추천 음료를 간단히 소개하세요. "
                "고객이 주문을 시작하면 BaristaOrderTaker 로 연결하세요."
            ),
            llm=openai.LLM(parallel_tool_calls=False),
            tts=cartesia.TTS(voice=voices["greeter"]),
        )
        self.menu = menu

    @function_tool()
    async def to_order(self, context: RunContext_T):
        """고객이 주문을 시작할 때 호출"""
        return await self._transfer_to_agent("order", context)

class BaristaOrderTaker(BaseAgent):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                f"당신은 BaristaOrderTaker 입니다. 스타벅스 메뉴:\n{menu}\n"
                "고객에게 사이즈, 옵션(디카페인, 우유 종류 등)을 확인하며 주문을 입력하세요. "
                "add_item tool 로 주문을 저장하고, 완료되면 PaymentAgent 로 이동하세요."
            ),
            tools=[add_item, update_name, update_car_model, update_license_plate],
            tts=cartesia.TTS(voice=voices["order"]),
        )

    @function_tool()
    async def order_done(self, context: RunContext_T):
        """고객이 주문 완료라고 할 때 PaymentAgent 로 이동"""
        if not context.userdata.order:
            return "아직 주문이 없습니다. 음료/푸드를 알려주세요."
        return await self._transfer_to_agent("payment", context)

class PaymentAgent(BaseAgent):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "당신은 PaymentAgent 입니다. 총 금액을 계산해 안내하고, 결제 방식을 물어봅니다. "
                "update_pay_method tool 로 결제 방식을 저장하고, 결제 완료 후 PickupNotifier 로 이동하세요."
            ),
            tools=[update_pay_method],
            tts=cartesia.TTS(voice=voices["payment"]),
        )

    @function_tool()
    async def confirm_total(
        self,
        total: Annotated[float, Field(description="총 금액 (₩)")],
        context: RunContext_T,
    ) -> str:
        context.userdata.expense = total
        return f"총 {total:,.0f}원입니다."

    @function_tool()
    async def payment_done(self, context: RunContext_T):
        """결제 완료 시 호출"""
        if not context.userdata.expense or not context.userdata.pay_method:
            return "결제 금액 또는 방식을 먼저 확인해주세요."
        return await self._transfer_to_agent("pickup", context)

class PickupNotifier(BaseAgent):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "당신은 PickupNotifier 입니다. 결제 이후 고객에게 픽업 창구 번호(예: '2번 창구')와 "
                "추가 안내 사항을 전달합니다. 모든 과정이 종료되면 LaneGreeter 로 돌아갑니다."
            ),
            tts=cartesia.TTS(voice=voices["pickup"]),
        )

    @function_tool()
    async def pickup_complete(self, context: RunContext_T):
        context.userdata.checked_out = True
        return await self._transfer_to_agent("greeter", context)

# ---------------------------------------------------------------------------
# Entry point for LiveKit Worker
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    menu = (
        "Espresso: ₩4,500 | Americano: ₩4,800 | Caffè Latte: ₩5,300 | "
        "Cold Brew: ₩5,000 | Caramel Macchiato: ₩5,900 | "
        "Butter Croissant: ₩3,800 | Blueberry Muffin: ₩4,200"
    )

    userdata = DriveThruData()
    userdata.agents.update({
        "greeter": LaneGreeter(menu),
        "order":   BaristaOrderTaker(menu),
        "payment": PaymentAgent(menu),
        "pickup":  PickupNotifier(menu),
    })

    session = AgentSession[DriveThruData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        max_tool_steps=7,
    )

    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
