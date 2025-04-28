import wave
import os
import math

def combine_wavs_in_batches(wav_folder, batch_size=10, output_prefix="combined"):
    # 모든 wav 파일을 정렬된 순서로 가져옴
    files = sorted(
        [f for f in os.listdir(wav_folder) if f.endswith(".wav")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    if not files:
        print("No wav files found.")
        return

    num_batches = math.ceil(len(files) / batch_size)

    for batch_idx in range(num_batches):
        batch_files = files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        output_file = f"{output_prefix}_{batch_idx}.wav"

        # 첫 파일에서 파라미터 추출
        with wave.open(os.path.join(wav_folder, batch_files[0]), 'rb') as wf:
            params = wf.getparams()
            frames = [wf.readframes(wf.getnframes())]

        # 나머지 파일들에서 데이터만 추출
        for fname in batch_files[1:]:
            with wave.open(os.path.join(wav_folder, fname), 'rb') as wf:
                if wf.getparams() != params:
                    raise ValueError(f"Wave params mismatch in {fname}")
                frames.append(wf.readframes(wf.getnframes()))

        # 합쳐서 저장
        with wave.open(output_file, 'wb') as out_wf:
            out_wf.setparams(params)
            for frame in frames:
                out_wf.writeframes(frame)

        print(f"Combined {len(batch_files)} wav files into {output_file}")

# 사용 예시
combine_wavs_in_batches("./wav", batch_size=50, output_prefix="combined")