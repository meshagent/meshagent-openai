import base64
import io
from pathlib import PurePosixPath
from typing import Protocol

from PIL import Image

from meshagent.agents.images_dataset import ImagesDataset
from meshagent.api import RoomException
from meshagent.api.messaging import FileContent
from meshagent.tools.storage import StorageToolkit
from meshagent.tools.tool import FunctionTool
from meshagent.tools.toolkit import ToolContext, Toolkit

from meshagent.openai.proxy.proxy import get_client

DEFAULT_IMAGE_GENERATION_MODEL = "gpt-image-2"


class ImageGenerationClient(Protocol):
    @property
    def images(self): ...


def _extension_format(path: str) -> tuple[str, str, str]:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".png":
        return path, "PNG", "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return path, "JPEG", "image/jpeg"
    normalized = str(PurePosixPath(path).with_suffix(".png"))
    return normalized, "PNG", "image/png"


def _convert_image(data: bytes, image_format: str) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as source:
            output = io.BytesIO()
            image = source
            if image_format == "JPEG":
                if source.mode in {"RGBA", "LA"} or (
                    source.mode == "P" and "transparency" in source.info
                ):
                    background = Image.new("RGB", source.size, "white")
                    rgba = source.convert("RGBA")
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    image = background
                else:
                    image = source.convert("RGB")
            image.save(output, format=image_format)
            return output.getvalue()
    except Exception as error:
        raise RoomException(f"unable to decode image: {error}") from error


class _ImageGenerationBackend:
    def __init__(
        self,
        *,
        images_dataset: ImagesDataset,
        storage_toolkit: StorageToolkit,
        client: ImageGenerationClient,
        model: str,
    ) -> None:
        self.images_dataset = images_dataset
        self.storage_toolkit = storage_toolkit
        self.client = client
        self.model = model

    async def generate(
        self,
        *,
        prompt: str,
        referenced_image_ids: list[str] | None,
        created_by: str,
    ) -> dict:
        references = referenced_image_ids or []
        if references:
            image_files = []
            for image_id in references:
                record = await self.images_dataset.read_record(image_id=image_id)
                if record is None:
                    raise RoomException(f"image not found: {image_id}")
                extension = ".jpg" if record.mime_type == "image/jpeg" else ".png"
                image_files.append(
                    (
                        f"{image_id}{extension}",
                        record.data,
                        record.mime_type,
                    )
                )
            response = await self.client.images.edit(
                image=image_files,
                prompt=prompt,
                model=self.model,
                background="auto",
                quality="auto",
                size="auto",
                output_format="png",
            )
        else:
            response = await self.client.images.generate(
                prompt=prompt,
                model=self.model,
                background="auto",
                quality="auto",
                size="auto",
                output_format="png",
            )
        if not response.data or response.data[0].b64_json is None:
            raise RoomException("image generation returned no image data")
        try:
            data = base64.b64decode(response.data[0].b64_json, validate=True)
        except ValueError as error:
            raise RoomException(
                "image generation returned invalid image data"
            ) from error
        saved = await self.images_dataset.save(
            data=data,
            mime_type="image/png",
            created_by=created_by,
            annotations={"prompt": prompt},
        )
        return {"saved_image_id": saved.id, "mime_type": saved.mime_type}

    async def read(self, *, image_id: str) -> FileContent:
        record = await self.images_dataset.read_record(image_id=image_id)
        if record is None:
            raise RoomException(f"image not found: {image_id}")
        extension = ".jpg" if record.mime_type == "image/jpeg" else ".png"
        return FileContent(
            data=record.data,
            name=f"{image_id}{extension}",
            mime_type=record.mime_type,
        )

    async def delete(self, *, image_id: str) -> dict:
        deleted = await self.images_dataset.delete(image_id=image_id)
        if not deleted:
            raise RoomException(f"image not found: {image_id}")
        return {"deleted_image_id": image_id}

    async def export(self, *, image_id: str, destination_path: str) -> dict:
        record = await self.images_dataset.read_record(image_id=image_id)
        if record is None:
            raise RoomException(f"image not found: {image_id}")
        actual_path, image_format, mime_type = _extension_format(destination_path)
        data = _convert_image(record.data, image_format)
        await self.storage_toolkit.write_bytes(
            path=actual_path,
            data=data,
            overwrite=True,
        )
        return {
            "exported_image_id": image_id,
            "destination_path": actual_path,
            "mime_type": mime_type,
        }

    async def import_image(self, *, source_path: str, created_by: str) -> dict:
        content = await self.storage_toolkit.read_file(path=source_path)
        _, image_format, mime_type = _extension_format(source_path)
        data = _convert_image(content.data, image_format)
        saved = await self.images_dataset.save(
            data=data,
            mime_type=mime_type,
            created_by=created_by,
            annotations={"source_path": source_path},
        )
        return {
            "saved_image_id": saved.id,
            "source_path": source_path,
            "mime_type": saved.mime_type,
        }


class ImageGenerateTool(FunctionTool):
    def __init__(self, backend: _ImageGenerationBackend):
        self._backend = backend
        super().__init__(
            name="imagegen",
            title="generate image",
            description="Generate or edit an image and save it in the images dataset.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["prompt", "referenced_image_ids"],
                "properties": {
                    "prompt": {"type": "string"},
                    "referenced_image_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "Image IDs from the images dataset to edit. "
                            "Use an empty array to generate a new image."
                        ),
                    },
                },
            },
        )

    async def execute(
        self,
        context: ToolContext,
        *,
        prompt: str,
        referenced_image_ids: list[str] | None = None,
    ):
        return await self._backend.generate(
            prompt=prompt,
            referenced_image_ids=referenced_image_ids,
            created_by=context.caller.id,
        )


class ReadImageTool(FunctionTool):
    def __init__(self, backend: _ImageGenerationBackend):
        self._backend = backend
        super().__init__(
            name="read_image",
            title="read image",
            description="Read image content from the images dataset.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["id"],
                "properties": {"id": {"type": "string"}},
            },
        )

    async def execute(self, context: ToolContext, *, id: str):
        del context
        return await self._backend.read(image_id=id)


class DeleteImageTool(FunctionTool):
    def __init__(self, backend: _ImageGenerationBackend):
        self._backend = backend
        super().__init__(
            name="delete_image",
            title="delete image",
            description="Delete an image from the images dataset.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["id"],
                "properties": {"id": {"type": "string"}},
            },
        )

    async def execute(self, context: ToolContext, *, id: str):
        del context
        return await self._backend.delete(image_id=id)


class ExportImageTool(FunctionTool):
    def __init__(self, backend: _ImageGenerationBackend):
        self._backend = backend
        super().__init__(
            name="export_image",
            title="export image",
            description="Export an image from the images dataset to configured storage.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "destination_path"],
                "properties": {
                    "id": {"type": "string"},
                    "destination_path": {"type": "string"},
                },
            },
        )

    async def execute(
        self,
        context: ToolContext,
        *,
        id: str,
        destination_path: str,
    ):
        del context
        return await self._backend.export(
            image_id=id,
            destination_path=destination_path,
        )


class ImportImageTool(FunctionTool):
    def __init__(self, backend: _ImageGenerationBackend):
        self._backend = backend
        super().__init__(
            name="import_image",
            title="import image",
            description="Import an image from configured storage into the images dataset.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["source_path"],
                "properties": {"source_path": {"type": "string"}},
            },
        )

    async def execute(self, context: ToolContext, *, source_path: str):
        return await self._backend.import_image(
            source_path=source_path,
            created_by=context.caller.id,
        )


class ImageGenerationToolkit(Toolkit):
    def __init__(
        self,
        *,
        images_dataset: ImagesDataset,
        storage_toolkit: StorageToolkit,
        model: str = DEFAULT_IMAGE_GENERATION_MODEL,
        client: ImageGenerationClient | None = None,
        api_key: str | None = None,
    ):
        backend = _ImageGenerationBackend(
            images_dataset=images_dataset,
            storage_toolkit=storage_toolkit,
            client=client or get_client(api_key=api_key),
            model=model,
        )
        super().__init__(
            name="image-generation",
            title="image generation",
            description="Generate, read, delete, import, and export dataset images.",
            tools=[
                ImageGenerateTool(backend),
                ReadImageTool(backend),
                DeleteImageTool(backend),
                ExportImageTool(backend),
                ImportImageTool(backend),
            ],
        )
