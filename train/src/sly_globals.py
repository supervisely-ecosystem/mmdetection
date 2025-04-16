import os
from pathlib import Path
import sys
import supervisely as sly
import shutil
import zipfile
import pkg_resources
from supervisely.nn.artifacts.mmdetection import MMDetection

# from dotenv import load_dotenv

root_source_dir = str(Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)
source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)
ui_sources_dir = os.path.join(source_path, "ui")
sly.logger.info(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

debug_env_path = os.path.join(root_source_dir, "train", "debug.env")
secret_debug_env_path = os.path.join(root_source_dir, "train", "secret_debug.env")
# @TODO: for debug
# load_dotenv(debug_env_path)
# load_dotenv(secret_debug_env_path, override=True)

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])
project_info = api.project.get_info_by_id(project_id)

sly_mmdet = MMDetection(team_id)

# @TODO: for debug
# sly.fs.clean_dir(my_app.data_dir)  

project_dir = os.path.join(my_app.data_dir, "sly_project")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
project_fs: sly.Project = None

data_dir = sly.app.get_synced_data_dir()
artifacts_dir = os.path.join(data_dir, "artifacts")
sly.fs.mkdir(artifacts_dir)
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
sly.fs.mkdir(checkpoints_dir)

configs_dir = os.path.join(root_source_dir, "configs")
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

sly_mmdet_generated_metadata = None # for project Workflow purposes
train_size, val_size = None, None