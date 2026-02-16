import argparse
import asyncio
import logging
import re
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

from gpt5_roleplay_system.config import load_config
from gpt5_roleplay_system.llm import ExtractedFact, LLMClient, LLMResponseBundle
from gpt5_roleplay_system.models import Action, CommandType, InboundChat
from gpt5_roleplay_system.server import GPT5RoleplayServer, _build_knowledge_store, _build_tracer


LOG = logging.getLogger("viewer_command_gui")
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
XYZ_RE = re.compile(r"[-+]?\d*\.?\d+")


class MockLLMClient(LLMClient):
    async def is_addressed_to_me(self, *args, **kwargs) -> bool:
        return True

    async def generate_bundle(self, *args, **kwargs) -> LLMResponseBundle:
        return LLMResponseBundle(text="", actions=[])

    async def generate_autonomous_bundle(self, *args, **kwargs) -> LLMResponseBundle:
        return LLMResponseBundle(text="", actions=[])

    async def summarize(self, summary: str, messages: List[InboundChat]) -> str:
        return summary

    async def extract_facts(self, *args, **kwargs) -> List[ExtractedFact]:
        return []


@dataclass
class EntityRecord:
    source: str
    name: str
    uuid: str
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    distance: Optional[float]
    raw: Dict[str, object] = field(default_factory=dict)

    @property
    def pos_text(self) -> str:
        if self.x is None or self.y is None or self.z is None:
            return "?"
        return f"{self.x:.2f}, {self.y:.2f}, {self.z:.2f}"

    @property
    def distance_text(self) -> str:
        if self.distance is None:
            return "?"
        return f"{self.distance:.2f}"


@dataclass
class BackendSnapshot:
    connected: bool = False
    sessions: List[str] = field(default_factory=list)
    selected_session: str = ""
    location: str = ""
    avatar_position: Optional[Tuple[float, float, float]] = None
    agents: List[EntityRecord] = field(default_factory=list)
    objects: List[EntityRecord] = field(default_factory=list)
    updated_at: float = 0.0


class AsyncBackend:
    def __init__(self, host_override: Optional[str], port_override: Optional[int]) -> None:
        self._host_override = host_override
        self._port_override = port_override
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[GPT5RoleplayServer] = None
        self._server_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._selected_session_id = ""
        self._snapshot = BackendSnapshot()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="viewer-command-gui-backend")
        self._thread.start()
        if not self._ready.wait(timeout=8):
            raise RuntimeError("Backend failed to start")

    def stop(self) -> None:
        if not self._loop:
            return

        def _shutdown() -> None:
            if self._poll_task:
                self._poll_task.cancel()
            if self._server_task:
                self._server_task.cancel()
            self._loop.stop()

        self._loop.call_soon_threadsafe(_shutdown)
        if self._thread:
            self._thread.join(timeout=3)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._server_task = loop.create_task(self._run_server())
        self._poll_task = loop.create_task(self._poll_state())
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _run_server(self) -> None:
        config = load_config()
        if self._host_override:
            config.host = self._host_override
        if self._port_override is not None:
            config.port = self._port_override
        # Keep the GUI test harness lightweight and local-only.
        config.wandb.enabled = False
        config.neo4j.uri = ""
        config.neo4j.user = ""
        config.neo4j.password = ""
        config.autonomy.enabled = False
        config.episode.persist_state = False

        LOG.info("Starting test server on %s:%s", config.host, config.port)
        knowledge_store = _build_knowledge_store(config)
        tracer = _build_tracer(config)
        llm = MockLLMClient()
        self._server = GPT5RoleplayServer(config.host, config.port, config, knowledge_store, llm, tracer)
        await self._server.start()

    async def _poll_state(self) -> None:
        while True:
            with self._lock:
                selected = self._selected_session_id

            snapshot = BackendSnapshot(updated_at=time.time())
            server = self._server
            if server:
                sessions = list(server._sessions.keys())
                snapshot.sessions = sessions
                snapshot.connected = bool(sessions)

                if selected and selected in server._sessions:
                    session_id = selected
                elif sessions:
                    session_id = sessions[0]
                else:
                    session_id = ""
                snapshot.selected_session = session_id

                if session_id:
                    session = server._sessions[session_id]
                    env = session.controller._pipeline._environment  # noqa: SLF001
                    avatar_pos = _parse_xyz_from_string(env.avatar_position)
                    snapshot.avatar_position = avatar_pos
                    snapshot.location = env.location
                    snapshot.agents = _build_entities(env.agents, "agent", avatar_pos)
                    snapshot.objects = _build_entities(env.objects, "object", avatar_pos)

            with self._lock:
                self._snapshot = snapshot
                if snapshot.selected_session and snapshot.selected_session != self._selected_session_id:
                    self._selected_session_id = snapshot.selected_session

            await asyncio.sleep(0.5)

    def get_snapshot(self) -> BackendSnapshot:
        with self._lock:
            return self._snapshot

    def set_selected_session(self, session_id: str) -> None:
        with self._lock:
            self._selected_session_id = session_id

    def send_action(self, action: Action, timeout_seconds: float = 4.0) -> Tuple[bool, str]:
        if not self._loop:
            return False, "Backend loop not running"
        fut = asyncio.run_coroutine_threadsafe(self._send_action_async(action), self._loop)
        try:
            session_id = fut.result(timeout=timeout_seconds)
            return True, f"Sent {action.command_type.value} to session {session_id}"
        except Exception as exc:  # pragma: no cover - runtime/UI path
            return False, str(exc)

    async def _send_action_async(self, action: Action) -> str:
        if not self._server:
            raise RuntimeError("Server not initialized")
        sessions = self._server._sessions
        if not sessions:
            raise RuntimeError("No viewer connected")

        with self._lock:
            selected = self._selected_session_id
        if selected and selected in sessions:
            session = sessions[selected]
        else:
            session = sessions[next(iter(sessions))]

        await session.send_actions([action])
        return session.session_id


def _parse_xyz_from_string(text: str) -> Optional[Tuple[float, float, float]]:
    if not text:
        return None
    matches = XYZ_RE.findall(text)
    if len(matches) < 3:
        return None
    try:
        return float(matches[0]), float(matches[1]), float(matches[2])
    except ValueError:
        return None


def _extract_uuid_and_name(raw_name: str, entry: Dict[str, object]) -> Tuple[str, str]:
    uuid_value = str(entry.get("uuid") or entry.get("target_uuid") or "").strip()
    name = raw_name or "Unknown"
    if uuid_value:
        return uuid_value, name
    if "|" in name:
        candidate_name, candidate_uuid = name.rsplit("|", 1)
        candidate_uuid = candidate_uuid.strip()
        if UUID_RE.match(candidate_uuid):
            return candidate_uuid, candidate_name.strip()
    return "", name


def _extract_xyz(entry: Dict[str, object]) -> Optional[Tuple[float, float, float]]:
    has_xyz = all(key in entry for key in ("x", "y", "z"))
    if has_xyz:
        try:
            return float(entry["x"]), float(entry["y"]), float(entry["z"])
        except (TypeError, ValueError):
            pass
    position = str(entry.get("position") or "")
    return _parse_xyz_from_string(position)


def _build_entities(raw_entries: List[Dict[str, object]], source: str, avatar_pos: Optional[Tuple[float, float, float]]) -> List[EntityRecord]:
    out: List[EntityRecord] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        name_raw = str(entry.get("name") or "Unknown")
        uuid_value, display_name = _extract_uuid_and_name(name_raw, entry)
        xyz = _extract_xyz(entry)

        distance = None
        if xyz and avatar_pos:
            dx = xyz[0] - avatar_pos[0]
            dy = xyz[1] - avatar_pos[1]
            dz = xyz[2] - avatar_pos[2]
            distance = (dx * dx + dy * dy + dz * dz) ** 0.5

        out.append(
            EntityRecord(
                source=source,
                name=display_name,
                uuid=uuid_value,
                x=xyz[0] if xyz else None,
                y=xyz[1] if xyz else None,
                z=xyz[2] if xyz else None,
                distance=distance,
                raw=entry,
            )
        )
    out.sort(key=lambda item: item.distance if item.distance is not None else 1e9)
    return out


class ViewerCommandGUI:
    def __init__(self, backend: AsyncBackend) -> None:
        self._backend = backend
        self._root = tk.Tk()
        self._root.title("Viewer Command GUI")
        self._root.geometry("1120x760")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._agents_by_iid: Dict[str, EntityRecord] = {}
        self._objects_by_iid: Dict[str, EntityRecord] = {}

        self._session_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="Starting...")
        self._location_var = tk.StringVar(value="Location: ?")
        self._avatar_var = tk.StringVar(value="Avatar: ?")
        self._standoff_var = tk.StringVar(value="0.5")
        self._x_var = tk.StringVar(value="")
        self._y_var = tk.StringVar(value="")
        self._z_var = tk.StringVar(value="")
        self._chat_var = tk.StringVar(value="")
        self._emote_var = tk.StringVar(value="")

        self._build_ui()
        self._refresh_loop()

    def run(self) -> None:
        self._root.mainloop()

    def _build_ui(self) -> None:
        top = ttk.Frame(self._root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Session:").pack(side=tk.LEFT)
        self._session_combo = ttk.Combobox(top, textvariable=self._session_var, state="readonly", width=28)
        self._session_combo.pack(side=tk.LEFT, padx=(6, 10))
        self._session_combo.bind("<<ComboboxSelected>>", self._on_session_selected)

        ttk.Label(top, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self._location_var).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Label(top, textvariable=self._avatar_var).pack(side=tk.RIGHT)

        lists = ttk.Frame(self._root, padding=(8, 0, 8, 0))
        lists.pack(fill=tk.BOTH, expand=True)

        agents_frame, self._agents_tree = self._make_tree(lists, "Agents")
        agents_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self._agents_tree.bind("<<TreeviewSelect>>", self._on_agents_select)

        objects_frame, self._objects_tree = self._make_tree(lists, "Objects")
        objects_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self._objects_tree.bind("<<TreeviewSelect>>", self._on_objects_select)

        controls = ttk.Frame(self._root, padding=8)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Standoff (m)").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(controls, textvariable=self._standoff_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=(6, 16))

        ttk.Button(controls, text="Move To Selected", command=self._move_to_selected).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Touch Selected", command=self._touch_selected).grid(row=0, column=3, padx=4)
        ttk.Button(controls, text="Sit Selected", command=self._sit_selected).grid(row=0, column=4, padx=4)
        ttk.Button(controls, text="Face Selected", command=self._face_selected).grid(row=0, column=5, padx=4)
        ttk.Button(controls, text="Stand", command=self._stand).grid(row=0, column=6, padx=4)

        ttk.Label(controls, text="X").grid(row=1, column=0, sticky=tk.E, pady=(8, 0))
        ttk.Entry(controls, textvariable=self._x_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=(6, 2), pady=(8, 0))
        ttk.Label(controls, text="Y").grid(row=1, column=2, sticky=tk.E, pady=(8, 0))
        ttk.Entry(controls, textvariable=self._y_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=(6, 2), pady=(8, 0))
        ttk.Label(controls, text="Z").grid(row=1, column=4, sticky=tk.E, pady=(8, 0))
        ttk.Entry(controls, textvariable=self._z_var, width=10).grid(row=1, column=5, sticky=tk.W, padx=(6, 8), pady=(8, 0))
        ttk.Button(controls, text="Move To XYZ", command=self._move_to_xyz).grid(row=1, column=6, padx=4, pady=(8, 0))

        ttk.Label(controls, text="Chat").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(controls, textvariable=self._chat_var, width=60).grid(row=2, column=1, columnspan=4, sticky=tk.W, padx=(6, 8), pady=(8, 0))
        ttk.Button(controls, text="Send Chat", command=self._send_chat).grid(row=2, column=5, columnspan=2, padx=4, pady=(8, 0))

        ttk.Label(controls, text="Emote").grid(row=3, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(controls, textvariable=self._emote_var, width=60).grid(row=3, column=1, columnspan=4, sticky=tk.W, padx=(6, 8), pady=(8, 0))
        ttk.Button(controls, text="Send Emote", command=self._send_emote).grid(row=3, column=5, columnspan=2, padx=4, pady=(8, 0))

        log_frame = ttk.Frame(self._root, padding=(8, 0, 8, 8))
        log_frame.pack(fill=tk.BOTH, expand=False)
        ttk.Label(log_frame, text="Log").pack(anchor=tk.W)
        self._log = tk.Text(log_frame, height=10)
        self._log.pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _make_tree(parent: tk.Widget, title: str) -> Tuple[ttk.LabelFrame, ttk.Treeview]:
        frame = ttk.LabelFrame(parent, text=title, padding=4)
        columns = ("name", "uuid", "distance", "position")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=16)
        tree.heading("name", text="Name")
        tree.heading("uuid", text="UUID")
        tree.heading("distance", text="Dist")
        tree.heading("position", text="Position")
        tree.column("name", width=220, anchor=tk.W)
        tree.column("uuid", width=260, anchor=tk.W)
        tree.column("distance", width=70, anchor=tk.E)
        tree.column("position", width=180, anchor=tk.W)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        return frame, tree

    def _refresh_loop(self) -> None:
        snap = self._backend.get_snapshot()
        self._status_var.set("Connected" if snap.connected else "Waiting for viewer...")
        self._location_var.set(f"Location: {snap.location or '?'}")

        if snap.avatar_position:
            ax, ay, az = snap.avatar_position
            self._avatar_var.set(f"Avatar: {ax:.2f}, {ay:.2f}, {az:.2f}")
        else:
            self._avatar_var.set("Avatar: ?")

        current = self._session_var.get()
        if snap.sessions != list(self._session_combo["values"]):
            self._session_combo["values"] = snap.sessions
        if snap.selected_session and (current != snap.selected_session):
            self._session_var.set(snap.selected_session)

        self._repopulate_tree(self._agents_tree, self._agents_by_iid, snap.agents)
        self._repopulate_tree(self._objects_tree, self._objects_by_iid, snap.objects)
        self._root.after(500, self._refresh_loop)

    @staticmethod
    def _repopulate_tree(tree: ttk.Treeview, mapping: Dict[str, EntityRecord], entities: List[EntityRecord]) -> None:
        selected = tree.selection()
        selected_entity = mapping.get(selected[0]) if selected else None
        selected_identity = ViewerCommandGUI._entity_identity(selected_entity) if selected_entity else None

        tree.delete(*tree.get_children())
        mapping.clear()
        identity_to_iid: Dict[str, str] = {}
        for idx, entity in enumerate(entities):
            iid = ViewerCommandGUI._entity_iid(entity, idx)
            mapping[iid] = entity
            identity_to_iid[ViewerCommandGUI._entity_identity(entity)] = iid
            tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(entity.name, entity.uuid or "-", entity.distance_text, entity.pos_text),
            )
        if selected_identity and selected_identity in identity_to_iid:
            tree.selection_set(identity_to_iid[selected_identity])

    @staticmethod
    def _entity_iid(entity: EntityRecord, idx: int) -> str:
        if entity.uuid:
            return entity.uuid
        return f"{idx}:{ViewerCommandGUI._entity_identity(entity)}"

    @staticmethod
    def _entity_identity(entity: EntityRecord) -> str:
        if entity.uuid:
            return f"uuid:{entity.uuid}"
        return f"fallback:{entity.source}|{entity.name}|{entity.pos_text}"

    def _on_agents_select(self, _event: object) -> None:
        self._objects_tree.selection_remove(*self._objects_tree.selection())

    def _on_objects_select(self, _event: object) -> None:
        self._agents_tree.selection_remove(*self._agents_tree.selection())

    def _on_session_selected(self, _event: object) -> None:
        sid = self._session_var.get().strip()
        self._backend.set_selected_session(sid)
        self._log_line(f"Selected session {sid}")

    def _selected_entity(self) -> Optional[EntityRecord]:
        agent_sel = self._agents_tree.selection()
        if agent_sel:
            return self._agents_by_iid.get(agent_sel[0])
        obj_sel = self._objects_tree.selection()
        if obj_sel:
            return self._objects_by_iid.get(obj_sel[0])
        return None

    def _read_standoff(self) -> float:
        try:
            return float(self._standoff_var.get().strip() or "0.5")
        except ValueError:
            return 0.5

    def _move_to_selected(self) -> None:
        entity = self._selected_entity()
        if not entity:
            messagebox.showwarning("No selection", "Select an agent or object first.")
            return
        if entity.x is None or entity.y is None or entity.z is None:
            messagebox.showwarning("Missing position", "Selected item has no coordinates.")
            return
        standoff = self._read_standoff()
        params = {
            "x": str(entity.x),
            "y": str(entity.y),
            "z": str(entity.z),
            "standoff": str(standoff),
        }
        action = Action(
            command_type=CommandType.MOVE,
            x=entity.x,
            y=entity.y,
            z=entity.z,
            target_uuid=entity.uuid,
            parameters=params,
        )
        self._send_action(action)

    def _touch_selected(self) -> None:
        entity = self._selected_entity()
        if not entity or not entity.uuid:
            messagebox.showwarning("Missing UUID", "Select an item with a valid UUID for TOUCH.")
            return
        action = Action(command_type=CommandType.TOUCH, target_uuid=entity.uuid)
        self._send_action(action)

    def _sit_selected(self) -> None:
        entity = self._selected_entity()
        if not entity or not entity.uuid:
            messagebox.showwarning("Missing UUID", "Select an item with a valid UUID for SIT.")
            return
        action = Action(command_type=CommandType.SIT, target_uuid=entity.uuid)
        self._send_action(action)

    def _stand(self) -> None:
        action = Action(command_type=CommandType.STAND)
        self._send_action(action)

    def _face_selected(self) -> None:
        entity = self._selected_entity()
        if not entity:
            messagebox.showwarning("No selection", "Select an agent or object first.")
            return

        has_coords = entity.x is not None and entity.y is not None and entity.z is not None
        has_uuid = bool(entity.uuid)
        if not has_coords and not has_uuid:
            messagebox.showwarning("Missing target", "Selected item has neither UUID nor coordinates.")
            return

        x = float(entity.x) if entity.x is not None else 0.0
        y = float(entity.y) if entity.y is not None else 0.0
        z = float(entity.z) if entity.z is not None else 0.0

        params: Dict[str, str] = {}
        if has_coords:
            params["x"] = str(x)
            params["y"] = str(y)
            params["z"] = str(z)

        action = Action(
            command_type=CommandType.FACE_TARGET,
            target_uuid=entity.uuid,
            x=x,
            y=y,
            z=z,
            parameters=params,
        )
        self._send_action(action)

    def _move_to_xyz(self) -> None:
        try:
            x = float(self._x_var.get().strip())
            y = float(self._y_var.get().strip())
            z = float(self._z_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid coordinates", "Enter numeric X/Y/Z values.")
            return
        standoff = self._read_standoff()
        params = {"x": str(x), "y": str(y), "z": str(z), "standoff": str(standoff)}
        action = Action(command_type=CommandType.MOVE, x=x, y=y, z=z, parameters=params)
        self._send_action(action)

    def _send_chat(self) -> None:
        text = self._chat_var.get().strip()
        if not text:
            return
        action = Action(command_type=CommandType.CHAT, content=text, parameters={"content": text})
        self._send_action(action)

    def _send_emote(self) -> None:
        text = self._emote_var.get().strip()
        if not text:
            return
        action = Action(command_type=CommandType.EMOTE, content=text, parameters={"content": text})
        self._send_action(action)

    def _send_action(self, action: Action) -> None:
        ok, msg = self._backend.send_action(action)
        prefix = "OK" if ok else "ERR"
        self._log_line(f"[{prefix}] {msg}")

    def _log_line(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._log.insert(tk.END, f"{stamp} {text}\n")
        self._log.see(tk.END)

    def _on_close(self) -> None:
        try:
            self._backend.stop()
        finally:
            self._root.destroy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local GUI for viewer command testing without full AI.")
    parser.add_argument("--host", default=None, help="Override server bind host")
    parser.add_argument("--port", type=int, default=None, help="Override server bind port")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    backend = AsyncBackend(host_override=args.host, port_override=args.port)
    backend.start()
    gui = ViewerCommandGUI(backend)
    gui.run()


if __name__ == "__main__":
    main()
