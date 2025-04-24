from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        # stt=deepgram.STT(model="nova-3", language="multi"),
        stt=openai.STT(model="whisper-1"),
        # stt=openai.STT(model="gpt-4o-mini-transcribe"),
        # llm=openai.LLM(model="gpt-4o-mini"),
        llm=openai.LLM(model="gpt-4o"),
        # tts=cartesia.TTS(),
        tts=openai.TTS(model="gpt-4o-mini-tts"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))









# import os
# import asyncio
# from dotenv import load_dotenv
# from livekit import agents, rtc
# from livekit.agents.tts import SynthesizedAudio, StreamAdapter  # StreamAdapter 임포트
# from livekit.plugins import openai
# from typing import AsyncIterable, List

# # 환경변수 로드
# load_dotenv()

# # 예시 텍스트 스트림 (실제 환경에 맞게 구현)
# async def sample_text_stream(texts: List[str], delay: float = 1.5) -> AsyncIterable[str]:
#     for text in texts:
#         await asyncio.sleep(delay)
#         yield text

# # SynthesizedAudio 스트림을 받아 RTC 오디오 소스에 전송하는 함수
# async def send_audio(audio_stream: AsyncIterable[SynthesizedAudio], audio_source: rtc.AudioSource):
#     async for audio in audio_stream:
#         await audio_source.capture_frame(audio.frame)

# async def entrypoint(ctx: agents.JobContext):
#     await ctx.connect()

#     # RTC 오디오 소스 및 로컬 오디오 트랙 생성 (44100Hz, mono)
#     audio_source = rtc.AudioSource(44100, 1)
#     track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
#     await ctx.room.local_participant.publish_track(track)

#     # OpenAI TTS 인스턴스 생성 (스트리밍은 기본 지원하지 않음)
#     tts = openai.TTS(model="gpt-4o-mini-tts")

#     # StreamAdapter는 생성자에 인자를 받지 않으므로 인자 없이 생성합니다.
#     stream_adapter = StreamAdapter()
#     # 예시로, TTS 인스턴스를 할당하는 방법:
#     # stream_adapter.set_tts(tts)  # 만약 이런 메서드가 있다면 사용
#     # 또는 속성 할당 방식 (버전에 따라 다를 수 있음)
#     stream_adapter.tts = tts

#     # 이제 스트림 인터페이스를 얻습니다.
#     tts_stream = stream_adapter.stream()

#     # 별도 태스크로 TTS 스트림에서 나온 오디오 프레임을 RTC 오디오 소스로 전송
#     ctx.create_task(send_audio(tts_stream, audio_source))

#     # 예시 텍스트 목록 (실제 상황에서는 외부 입력이나 다른 스트림 사용 가능)
#     texts = [
#         "Hello, how are you?",
#         "I'm here to assist you with any questions.",
#         "Feel free to ask if you need help."
#     ]

#     # 텍스트 스트림에서 각 텍스트를 push_text로 전달
#     async for text in sample_text_stream(texts):
#         tts_stream.push_text(text)

#     # 모든 텍스트 입력 후 스트림 종료 신호 전달
#     tts_stream.end_input()

# if __name__ == "__main__":
#     agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))