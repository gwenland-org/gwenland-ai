"""Download one named evidence archive, without downloading build caches."""
import argparse
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

parser = argparse.ArgumentParser()
parser.add_argument('kernel')
parser.add_argument('filename')
parser.add_argument('destination', type=Path)
args = parser.parse_args()
owner, slug = args.kernel.split('/')
api = KaggleApi()
api.authenticate()
token = None
with api.build_kaggle_client() as client:
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        api._set_paging(request, 100, token)
        page = client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in page.files or []:
            if item.file_name == args.filename:
                response = requests.get(item.url, timeout=120)
                response.raise_for_status()
                args.destination.parent.mkdir(parents=True, exist_ok=True)
                args.destination.write_bytes(response.content)
                print(f'Downloaded {len(response.content)} bytes to {args.destination}')
                raise SystemExit(0)
        token = page.next_page_token
        if not token:
            raise SystemExit(f'Archive not found: {args.filename}')
