import subprocess
from pathlib import Path
from typing import Tuple


# deprecated
# only run on desktop
class Render:
    def __init__(
            self,
            world_dir: Path,
            lims: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        ) -> None:
        self.world_dir = world_dir
        self.xlim = lims[0]
        self.ylim = lims[1]
        self.zlim = lims[2]

    def make_render_script(self, obj_path):
        from params import ExportParams
        xlim ,ylim, zlim = self.xlim, self.ylim, self.zlim
        with open(ExportParams.scripts_dir / "export_obj.mwscript", 'w') as f:
            f.write("Save Log file: export_obj.log\n")
            f.write("Set render type: Wavefront OBJ absolute indices\n")
            f.write('Minecraft world: ' + str(self.world_dir.name) + "\n")
            f.write(f"Selection location min to max: {xlim[0]}, {ylim[0]}, {zlim[0]} to {xlim[1]}, {ylim[1]}, {zlim[1]}\n")
            f.write("Scale model by making each block 100 cm high\n")
            f.write(f"Export for Rendering: {str(obj_path.absolute())}")
        return ExportParams.scripts_dir / "export_obj.mwscript"

    def export_obj(self, obj_path: Path):
        from params import ExportParams
        commands = [
            "wine",
            str(Path(ExportParams.mineways_path).absolute()),
            "-m", "-s",
            str(self.world_dir.parent.absolute()),
            self.make_render_script(obj_path)
        ]
        process = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)
        return obj_path