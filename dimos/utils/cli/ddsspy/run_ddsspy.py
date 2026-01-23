# Copyright 2025-2026 Dimensional Inc.
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

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET

from rich.text import Text
from textual.app import App, ComposeResult
from textual.color import Color
from textual.widgets import DataTable

from dimos.utils.cli import theme
from dimos.utils.cli.ddsspy.ddsspy import DDSSpy, DDSSpyConfig


def gradient(max_value: float, value: float) -> str:
    ratio = min(value / max_value, 1.0) if max_value > 0 else 0
    cyan = Color.parse(theme.CYAN)
    yellow = Color.parse(theme.YELLOW)
    color = cyan.blend(yellow, ratio)
    return color.hex


def topic_text(topic_name: str) -> Text:
    if "#" in topic_name:
        parts = topic_name.split("#", 1)
        return Text(parts[0], style=theme.BRIGHT_WHITE) + Text("#" + parts[1], style=theme.BLUE)
    return Text(topic_name, style=theme.BRIGHT_WHITE)


def format_age(last_seen: float) -> str:
    if last_seen <= 0:
        return "—"
    age = time.time() - last_seen
    if age < 1:
        return f"{age * 1000:.0f}ms"
    if age < 60:
        return f"{age:.1f}s"
    if age < 3600:
        return f"{age / 60:.1f}m"
    return f"{age / 3600:.1f}h"


class DDSSpyApp(App):  # type: ignore[type-arg]
    CSS_PATH = "../dimos.tcss"

    CSS = f"""
    Screen {{
        layout: vertical;
        background: {theme.BACKGROUND};
    }}
    DataTable {{
        height: 2fr;
        width: 1fr;
        border: solid {theme.BORDER};
        background: {theme.BG};
        scrollbar-size: 0 0;
    }}
    DataTable > .datatable--header {{
        color: {theme.ACCENT};
        background: transparent;
    }}
    """

    refresh_interval: float = 0.5

    BINDINGS = [
        ("q", "quit"),
        ("ctrl+c", "quit"),
    ]

    def __init__(self, config: DDSSpyConfig, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.spy = DDSSpy(config)
        self.table: DataTable | None = None  # type: ignore[type-arg]

    def compose(self) -> ComposeResult:
        self.table = DataTable(zebra_stripes=False, cursor_type=None)  # type: ignore[arg-type]
        self.table.add_column("Topic")
        self.table.add_column("Type")
        self.table.add_column("Writers")
        self.table.add_column("Readers")
        self.table.add_column("Freq (Hz)")
        self.table.add_column("Bandwidth")
        self.table.add_column("Total Traffic")
        self.table.add_column("Last Seen")
        yield self.table

    def on_mount(self) -> None:
        self.spy.start()
        self.set_interval(self.refresh_interval, self.refresh_table)

    async def on_unmount(self) -> None:
        self.spy.stop()

    def refresh_table(self) -> None:
        if not self.table:
            return
        topics = list(self.spy.topic.values())
        topics.sort(key=lambda t: t.total_traffic(), reverse=True)
        self.table.clear(columns=False)

        for t in topics:
            freq = t.freq(5.0)
            kbps = t.kbps(5.0)
            bw_val, bw_unit = t.kbps_hr(5.0)
            total_val, total_unit = t.total_traffic_hr()

            type_name = t.type_name or "—"
            self.table.add_row(
                topic_text(t.name),
                Text(type_name, style=theme.FOREGROUND),
                Text(str(len(t.writers)), style=gradient(10, len(t.writers))),
                Text(str(len(t.readers)), style=gradient(10, len(t.readers))),
                Text(f"{freq:.1f}", style=gradient(10, freq)),
                Text(f"{bw_val} {bw_unit.value}/s", style=gradient(1024 * 3, kbps)),
                Text(f"{total_val} {total_unit.value}"),
                Text(format_age(t.last_seen), style=theme.DIM_WHITE),
            )


def build_cyclonedds_uri(
    domain_id: int | None,
    interface: str | None,
    allow_multicast: bool | None,
    peers: list[str],
) -> str | None:
    if domain_id is None and not interface and allow_multicast is None and not peers:
        return None

    root = ET.Element("CycloneDDS")
    domain = ET.SubElement(root, "Domain")
    if domain_id is not None:
        domain.set("id", str(domain_id))

    if interface or allow_multicast is not None:
        general = ET.SubElement(domain, "General")
        if interface:
            ET.SubElement(general, "NetworkInterfaceAddress").text = interface
        if allow_multicast is not None:
            ET.SubElement(general, "AllowMulticast").text = "true" if allow_multicast else "false"

    if peers:
        discovery = ET.SubElement(domain, "Discovery")
        peers_elem = ET.SubElement(discovery, "Peers")
        for peer in peers:
            ET.SubElement(peers_elem, "Peer", address=peer)

    return ET.tostring(root, encoding="unicode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDS spy tool using Cyclone DDS")
    parser.add_argument("--domain", type=int, default=None, help="DDS domain id")
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
        help="Network interface name or address for Cyclone DDS",
    )
    parser.add_argument(
        "--allow-multicast",
        dest="allow_multicast",
        action="store_true",
        default=None,
        help="Allow multicast for discovery/data",
    )
    parser.add_argument(
        "--no-allow-multicast",
        dest="allow_multicast",
        action="store_false",
        default=None,
        help="Disable multicast for discovery/data",
    )
    parser.add_argument(
        "--peer",
        action="append",
        default=[],
        help="Static peer address (repeatable)",
    )
    parser.add_argument(
        "--cyclonedds-uri",
        type=str,
        default=None,
        help="Raw Cyclone DDS XML config string (overrides other network options)",
    )
    parser.add_argument(
        "--cyclonedds-uri-file",
        type=str,
        default=None,
        help="Path to a Cyclone DDS XML config file (overrides other network options)",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=0.5,
        help="UI refresh interval in seconds",
    )
    return parser.parse_args()


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "web":
        import os as _os

        from textual_serve.server import Server  # type: ignore[import-not-found]

        server = Server(f"python {_os.path.abspath(__file__)}")
        server.serve()
        return

    args = parse_args()

    cyclonedds_uri = None
    if args.cyclonedds_uri_file:
        with open(args.cyclonedds_uri_file, encoding="utf-8") as f:
            cyclonedds_uri = f.read()
    elif args.cyclonedds_uri:
        cyclonedds_uri = args.cyclonedds_uri
    else:
        cyclonedds_uri = build_cyclonedds_uri(
            args.domain,
            args.interface,
            args.allow_multicast,
            args.peer,
        )

    if cyclonedds_uri:
        os.environ["CYCLONEDDS_URI"] = cyclonedds_uri

    config = DDSSpyConfig(
        domain_id=args.domain,
        poll_interval=max(args.refresh_interval / 2, 0.1),
        cyclonedds_uri=cyclonedds_uri,
    )
    DDSSpyApp(config).run()


if __name__ == "__main__":
    main()
