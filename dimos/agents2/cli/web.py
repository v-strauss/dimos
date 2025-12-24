#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Web-based input module for agents2 framework with browser text and voice input."""

import queue
from threading import Thread

import reactivex as rx
import reactivex.operators as ops
from reactivex.disposable import Disposable

from dimos.agents2 import Output, Reducer, Stream, skill
from dimos.core import Module, rpc
from dimos.web.robot_web_interface import RobotWebInterface
from dimos.stream.audio.stt.node_whisper import WhisperNode
from dimos.stream.audio.node_normalizer import AudioNormalizer
from dimos.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class WebInput(Module):
    """Web interface input module with text and voice support via STT."""

    running: bool = False
    web_interface: RobotWebInterface = None
    stt_node: WhisperNode = None
    text_queue: queue.Queue = None
    thread: Thread = None

    def __init__(self):
        super().__init__()
        self.text_queue = queue.Queue()

    @rpc
    def start(self) -> None:
        super().start()

        # Create web interface
        audio_subject = rx.subject.Subject()
        agent_response_subject = rx.subject.Subject()

        self.web_interface = RobotWebInterface(
            port=5555,
            text_streams={"agent_responses": agent_response_subject},
            audio_subject=audio_subject,
        )

        # Set up STT for audio transcription
        normalizer = AudioNormalizer()
        self.stt_node = WhisperNode()

        # Connect audio pipeline: browser audio → normalizer → whisper
        normalizer.consume_audio(audio_subject.pipe(ops.share()))
        self.stt_node.consume_audio(normalizer.emit_audio())

        # Subscribe to both text input sources
        # 1. Direct text from web interface
        unsub1 = self.web_interface.query_stream.subscribe(self.text_queue.put)
        self._disposables.add(Disposable(unsub1))

        # 2. Transcribed text from STT
        unsub2 = self.stt_node.emit_text().subscribe(self.text_queue.put)
        self._disposables.add(Disposable(unsub2))

        # Start web server in background thread
        self.thread = Thread(target=self.web_interface.run, daemon=True)
        self.thread.start()

        logger.info("Web interface started at http://localhost:5555")

    @skill(stream=Stream.call_agent, reducer=Reducer.string, output=Output.human)
    def web_human(self):
        """receives human input from web interface (text or voice)"""
        if self.running:
            return "already running"
        self.running = True

        # Yield text from queue (either typed or transcribed from voice)
        for message in iter(self.text_queue.get, None):
            logger.info(f"Web input received: {message}")
            yield message

    @rpc
    def stop(self) -> None:
        if self.web_interface:
            self.web_interface.stop()
        if self.thread:
            self.thread.join(timeout=1.0)
        super().stop()
