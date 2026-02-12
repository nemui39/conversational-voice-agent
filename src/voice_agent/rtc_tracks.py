"""WebRTC カスタムオーディオトラック

TTS 音声を WebRTC で送信するための AudioStreamTrack 実装。
音声がないときは無音を送り続け、play() で音声データをキューに入れると再生する。
"""

from __future__ import annotations

import asyncio
import fractions
from collections import deque

import numpy as np
from aiortc import AudioStreamTrack
from av import AudioFrame


class TTSPlaybackTrack(AudioStreamTrack):
    """TTS 音声データを WebRTC オーディオトラックとして送信する。

    - recv() が ~20ms ごとに呼ばれ、960 サンプル(48kHz) のフレームを返す
    - 音声がなければ無音フレームを返す
    - play() で 48kHz int16 PCM をキューに追加すると順次再生
    """

    kind = "audio"
    SAMPLE_RATE = 48000
    PTIME = 20  # ms
    SAMPLES_PER_FRAME = SAMPLE_RATE * PTIME // 1000  # 960

    def __init__(self) -> None:
        super().__init__()
        self._queue: deque[np.ndarray] = deque()
        self._playing = False
        self._on_playback_done: asyncio.Event | None = None

    async def recv(self) -> AudioFrame:
        """次のオーディオフレームを返す。キューに音声があればそこから、なければ無音。"""
        pts, time_base = await self.next_timestamp()

        if self._queue:
            samples = self._queue.popleft()
            self._playing = True
        else:
            samples = np.zeros(self.SAMPLES_PER_FRAME, dtype=np.int16)
            if self._playing:
                self._playing = False
                if self._on_playback_done:
                    self._on_playback_done.set()

        frame = AudioFrame(format="s16", layout="mono", samples=self.SAMPLES_PER_FRAME)
        frame.planes[0].update(samples.tobytes())
        frame.pts = pts
        frame.time_base = fractions.Fraction(1, self.SAMPLE_RATE)
        frame.sample_rate = self.SAMPLE_RATE

        return frame

    def play(self, pcm_48k: np.ndarray) -> asyncio.Event:
        """48kHz int16 PCM データをキューに追加して再生開始。

        Args:
            pcm_48k: 48kHz モノラル int16 numpy 配列

        Returns:
            再生完了時に set される Event
        """
        self._on_playback_done = asyncio.Event()

        # SAMPLES_PER_FRAME ごとに分割してキューに追加
        for i in range(0, len(pcm_48k), self.SAMPLES_PER_FRAME):
            chunk = pcm_48k[i : i + self.SAMPLES_PER_FRAME]
            if len(chunk) < self.SAMPLES_PER_FRAME:
                # 最後のチャンクが端数の場合、ゼロ埋め
                padded = np.zeros(self.SAMPLES_PER_FRAME, dtype=np.int16)
                padded[: len(chunk)] = chunk
                chunk = padded
            self._queue.append(chunk)

        return self._on_playback_done

    @property
    def is_playing(self) -> bool:
        """現在再生中かどうか。"""
        return self._playing or len(self._queue) > 0

    @property
    def queue_duration_ms(self) -> float:
        """キューに残っている音声の長さ（ミリ秒）。"""
        return len(self._queue) * self.PTIME
