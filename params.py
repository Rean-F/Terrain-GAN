import os
from pathlib import Path
import yaml


with open("./plain_params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

class CommonParams:
    _tmp = params["common"]
    worlds_dir = Path(_tmp["worlds_dir"])
    world_dir = worlds_dir / _tmp["world_name"]
    xlim, ylim, zlim = _tmp["xlim"], _tmp["ylim"], _tmp["zlim"]
    lims_name = f"{xlim[0]}:{xlim[1]}|{ylim[0]}:{ylim[1]}|{zlim[0]}:{zlim[1]}"

    _model_dir = Path(_tmp["models_dir"]) / _tmp["world_name"] / lims_name
    _log_dir = Path(_tmp["logs_dir"]) / _tmp["world_name"] / lims_name
    _out_dir = Path(_tmp["output_dir"]) / _tmp["world_name"] / lims_name
    map_dir = _out_dir / "maps"

    @staticmethod
    def check_dir() -> None:
        for tmp in [CommonParams.map_dir, ]:
            if not tmp.is_dir():
                os.makedirs(tmp)
        return

class Block2VecParams(CommonParams):
    model_dir = CommonParams._model_dir / "block2vec"
    log_dir = CommonParams._log_dir / "block2vec"
    out_dir = CommonParams._out_dir / "block2vec"

    embed_dim = params["block2vec"]["embed_dim"]
    epochs = params["block2vec"]["epochs"]

    @staticmethod
    def check_dir() -> None:
        for tmp in [Block2VecParams.model_dir, Block2VecParams.log_dir, Block2VecParams.out_dir, Block2VecParams.map_dir]:
            if not tmp.is_dir():
                os.makedirs(tmp)
        return

class SinGanParams(CommonParams):
    model_dir = CommonParams._model_dir / "singan"
    log_dir = CommonParams._log_dir / "singan"
    out_dir = CommonParams._out_dir / "singan"

    num_scales = params["singan"]["num_scales"]
    scales = params["singan"]["scales"]
    lr = float(params["singan"]["learning_rate"])
    n_iters = params["singan"]["n_iters"]
    noise_w = params["singan"]["noise_w"]

    @staticmethod
    def check_dir() -> None:
        for tmp in [SinGanParams.model_dir, SinGanParams.log_dir, SinGanParams.out_dir, SinGanParams.map_dir]:
            if not tmp.is_dir():
                os.makedirs(tmp)
        return

class ExportParams(CommonParams):
    empty_world_dir = Path(params["export"]["empty_world_dir"])

CommonParams.check_dir()
Block2VecParams.check_dir()
SinGanParams.check_dir()
ExportParams.check_dir()
