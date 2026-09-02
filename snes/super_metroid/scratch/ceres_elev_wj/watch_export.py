"""Watch + export Ceres station from first control (no Nintendo/title intro).

Headed window at 1x. MP4 under recordings/ceres/. Scratch only.
"""

from __future__ import annotations

from pathlib import Path

from retro_harness.headed import attach_headed, idle_headed
from retro_harness.video import VideoCaptureConfig, VideoRecorder
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.paths import RECORDINGS_DIR
from super_metroid.progression import MORPH_GRAPH
from super_metroid.ram import parse_state
from super_metroid.routes.kpdr.ceres.geometry import CERES_DATA_DIR
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
)
from super_metroid.routes.runtime import RouteSession

PIN = CERES_DATA_DIR / "ceres_first_control.state"
OUT = RECORDINGS_DIR / "ceres" / "ceres_elev_climb.mp4"


def _hud(env) -> str:
    st = parse_state(env.get_ram(), frame=0)
    return (
        f"x={int(st.samus_x)} y={int(st.samus_y)} "
        f"p={int(st.pose)} gs={int(st.game_state)} "
        f"room=0x{int(st.room_id):04X}"
    )


def main() -> None:
    if not PIN.is_file():
        raise SystemExit(f"missing pin: {PIN}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    assist.attach_env(env)
    pygame_mod = None
    writer: VideoRecorder | None = None
    try:
        boot_from_state(env, PIN, settle_frames=0)
        pygame_mod = attach_headed(
            env, title="SM Ceres from control", hud=_hud, speed=1.0
        )
        obs = env.render()
        writer = VideoRecorder(
            OUT,
            width=int(obs.shape[1]),
            height=int(obs.shape[0]),
            config=VideoCaptureConfig(
                fps=60, scale=3, crf=18, audio=True, footer=True
            ),
            audio_rate=int(env.em.get_audio_rate()),
        )
        session = RouteSession(
            env, writer=writer, assist=assist, graph=MORPH_GRAPH
        )
        session._capture_frame(obs, None)
        print("control", _hud(env), flush=True)
        play_ceres_outbound_to_ridley(session)
        print(f"ridley {session.frame} {_hud(env)}", flush=True)
        play_ceres_escape_to_landing(session)
        print(f"landing {session.frame} {_hud(env)}", flush=True)
        print(f"video {OUT} frames={writer.frames_written}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if pygame_mod is not None:
            idle_headed(env, pygame_mod, frames=120)
        env.close()


if __name__ == "__main__":
    main()
