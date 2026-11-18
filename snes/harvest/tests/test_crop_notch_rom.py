"""Single-bag experiments for mirrored side notches, separate from campaign policy."""
from pathlib import Path

import pytest

from harvest.tasks.nav import make_action, get_tile_at
from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState


@pytest.mark.rom
@pytest.mark.parametrize("layout,opening", [("eight_tile_ring", (0, 0)), ("eight_east_notch", (1, 0)), ("eight_west_notch", (-1, 0)), ("seven_east_access", (1, 0)), ("seven_west_access", (-1, 0))])
def test_eight_crop_notch_can_be_sown_from_a_natural_morning(layout, opening):
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.buy_seeds import BuyShelfItemTask
    from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask, EnsureCropSeedsTask
    from harvest.tasks.crop_planter import CropWaterTask
    from harvest.core.tile_catalog import CROP_TILE_RANGE

    pin = "PR_spring_final_Y1_Spring_D3"
    pins = Path(__file__).resolve().parents[1] / "custom_integrations/HarvestMoon-Snes"
    if not (pins / f"{pin}.state").exists():
        pytest.skip(f"missing local pin {pin}")
    env = AuditedEnv(make_harvest_env(pin, lean_step=True), capabilities=AuditCapabilities.all("harvest.crop_notch"))
    try:
        env.reset()
        def run(task, frames=18000):
            task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
            for frame in range(frames):
                result = task.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
                if result.status != TaskStatus.RUNNING:
                    break
                env.step(result.action.action if result.action else make_action())
            assert result.status == TaskStatus.SUCCESS, (task.name, frame, result.reason)
            return frame

        run(BuyShelfItemTask(stock_field="turnip_seeds"))
        run(EnsureCarryToolTask(tool_id=2, exit_when_done=True))
        run(EnsureCropSeedsTask(seed_type="turnip"))
        task = CropWaterTask(seed_type="turnip", layout_name=layout, work_mode="establish", bounds=(7, 33, 11, 37))
        # Fix only the plot choice: all navigation, hoeing and sowing run on the ROM.
        task._plan_new_plot_centers = lambda ram: [(9, 35)]
        plant_frames = run(task)
        crop_tiles = {(x, y) for y in range(34, 37) for x in range(8, 11)
                      if get_tile_at(env.get_ram(), x, y) in CROP_TILE_RANGE}
        assert len(crop_tiles) == (7 if layout.startswith("seven") else 8), crop_tiles
        assert ((9, 35) in crop_tiles) == (layout.startswith("eight_") and layout != "eight_tile_ring")
        assert (9 + opening[0], 35 + opening[1]) not in crop_tiles
        run(EnsureCarryToolTask(tool_id=16, exit_when_done=True))
        water_frames = run(CropWaterTask(work_mode="water", bounds=(7, 33, 11, 37)))
        assert all(get_tile_at(env.get_ram(), *tile) % 2 == 1 for tile in crop_tiles)
        from harvest.tasks.grass_planter import plan_stands, PASTURE_BOUNDS
        from harvest.maps.farm_pond import crop_working_tiles

        pasture = plan_stands(env.get_ram(), PASTURE_BOUNDS, 6, reach=None)
        assert pasture
        assert all((cx + dx, cy + dy) not in crop_working_tiles()
                   for cx, cy in pasture for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        print(f"{layout}: plant={plant_frames} frames water={water_frames} frames")
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()
