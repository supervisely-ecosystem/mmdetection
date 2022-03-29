import os
import pathlib
import sys
import shutil
import pkg_resources
import supervisely as sly
import zipfile
from dotenv import load_dotenv

root_source_path = str(pathlib.Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

source_path = str(pathlib.Path(sys.argv[0]).parents[0])
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)

ui_sources_dir = os.path.join(source_path, "ui")
sly.logger.info(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

models_configs_dir = os.path.join(root_source_path, "models")
sys.path.append(models_configs_dir)

debug_env_path = os.path.join(root_source_path, "serve", "debug.env")
secret_debug_env_path = os.path.join(root_source_path, "serve", "secret_debug.env")
# @TODO: for debug
# load_dotenv(debug_env_path)
# load_dotenv(secret_debug_env_path, override=True)

my_app = sly.AppService()
api = my_app.public_api

TASK_ID = my_app.task_id
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

meta: sly.ProjectMeta = None
model = None
local_weights_path = None
model_config_local_path = None
cfg = None
device = None

configs_dir = os.path.join(root_source_path, "configs")
mmdet_ver = pkg_resources.get_distribution("mmdet").version
zip_path = f"/tmp/mmdet/v{mmdet_ver}.zip"
if os.path.exists(zip_path) and os.path.isfile(zip_path) and not os.path.exists(configs_dir):
    sly.logger.info(f"Getting model configs of current mmdetection version {mmdet_ver}...")
    copied_zip_path = os.path.join(my_app.data_dir, f"v{mmdet_ver}.zip")
    shutil.copyfile(zip_path, copied_zip_path)
    with zipfile.ZipFile(copied_zip_path, 'r') as zip_ref:
        zip_ref.extractall(my_app.data_dir)
    unzipped_dir = os.path.join(my_app.data_dir, f"mmdetection-{mmdet_ver}")
    if os.path.isdir(unzipped_dir):
        shutil.move(os.path.join(unzipped_dir, "configs"), configs_dir)
    if os.path.isdir(configs_dir):
        shutil.rmtree(unzipped_dir)
        os.remove(copied_zip_path)
os.makedirs(configs_dir, exist_ok=True)
config_folders_cnt = len(os.listdir(configs_dir)) - 1
sly.logger.info(f"Found {config_folders_cnt} folders in {configs_dir} directory.")