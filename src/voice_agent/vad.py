"""Voice Activity Detection モジュール

webrtcvad を使って音声フレームから発話区間を検出する。
aiortc から届く 48kHz PCM フレーム（20ms）を受け取り、
発話開始〜無音 N 秒で「1発話」として PCM データを返す。
"""

from __future__ import annotations

import webrtcvad


class SpeechDetector:
    """48kHz PCM フレームを受け取り、発話区間を検出する。

    process_frame() に 20ms の PCM バイト列を渡し続けると、
    発話終了時（無音が silence_duration 秒続いたとき）に
    発話区間全体の PCM バイト列を返す。
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        silence_duration: float = 0.6,
        aggressiveness: int = 2,
        min_speech_ms: int = 300,
    ):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.min_speech_ms = min_speech_ms
        self._buffer: list[bytes] = []
        self._speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

    def process_frame(
        self, pcm_bytes: bytes, frame_duration_ms: int = 20
    ) -> bytes | None:
        """20ms PCM フレームを処理する。

        Args:
            pcm_bytes: int16 PCM バイト列（20ms 分）
            frame_duration_ms: フレーム長（ms）。webrtcvad は 10/20/30 対応。

        Returns:
            発話が完了した場合、発話区間全体の PCM バイト列。
            まだ発話中または無音なら None。
        """
        is_speech = self.vad.is_speech(pcm_bytes, self.sample_rate)

        if is_speech:
            if not self._speaking:
                self._speaking = True
            self._silence_frames = 0
            self._speech_frames += 1
            self._buffer.append(pcm_bytes)
        elif self._speaking:
            self._silence_frames += 1
            self._buffer.append(pcm_bytes)

            max_silence = int(self.silence_duration * 1000 / frame_duration_ms)
            if self._silence_frames >= max_silence:
                # 最小発話長チェック
                total_speech_ms = self._speech_frames * frame_duration_ms
                if total_speech_ms >= self.min_speech_ms:
                    speech_data = b"".join(self._buffer)
                    self.reset()
                    return speech_data
                else:
                    # 短すぎる → ノイズとして破棄
                    self.reset()
        return None

    @property
    def is_speaking(self) -> bool:
        """現在発話中かどうか。"""
        return self._speaking

    def reset(self) -> None:
        """内部状態をリセットする。"""
        self._buffer.clear()
        self._speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
